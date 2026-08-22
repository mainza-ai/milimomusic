"""Agent Runtime core tests — hermetic (no network, no real providers).

Covers:
  * ResiliencePolicy chain ordering + typed failover (quota/auth/parse)
  * AllProvidersFailedError carrying attempts
  * Experiencer schemas round-trip + brief validation bounds
  * ExperiencerAgent message construction (system persona, history, target)
"""
from __future__ import annotations

import json
from typing import List

import pytest
from pydantic import ValidationError

from app.agents.experiencer.agent import ExperiencerAgent
from app.agents.experiencer.schemas import AlbumBrief, ExperiencerVision
from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import ModelProfile, ResiliencePolicy
from app.core.llm_contracts import (
    AllProvidersFailedError,
    LLMAuthError,
    LLMParseError,
    LLMQuotaError,
    LLMResult,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeProvider:
    """Scripted provider: pops one behavior per call."""

    def __init__(self, behaviors: List[dict]):
        self._behaviors = list(behaviors)
        self.calls: List[List[dict]] = []

    def generate_chat(self, messages, model, **kwargs) -> LLMResult:
        self.calls.append(messages)
        behavior = self._behaviors.pop(0)
        if "raise" in behavior:
            raise behavior["raise"]
        content = behavior["content"]
        return LLMResult(
            content=content, provider="fake", model=model,
            latency_ms=1, usage={"prompt_tokens": 11, "completion_tokens": 7},
        )


VALID_VISION_JSON = json.dumps({
    "journey_title": "Neon Exodus",
    "concept_statement": "A driver leaves everything familiar and drives through the night toward a city that may not want them.",
    "life_journey_narrative": (
        "The tank was half full and the decision was already made. "
        "Streetlights strobed across the dashboard like a countdown. "
        "Every mile marker was a small goodbye; every rest stop, a rehearsal "
        "for the person they were about to become."
    ),
    "emotional_arc": [
        {"position": 1, "label": "Departure", "intensity": 0.4, "description": "engine idle, heart loud"},
        {"position": 2, "label": "Friction", "intensity": 0.8},
        {"position": 3, "label": "Arrival", "intensity": 0.6},
    ],
    "song_seeds": [
        {"working_title": "Half Full Tank", "mood": "resolute ache",
         "story_seed": "Leaving at 2am with the voicemail still unplayed, garage lights humming.",
         "suggested_style_tags": ["synthwave", "night drive"], "energy": 0.6,
         "placement_hint": "opener"},
        {"working_title": "Toll Booth Confessional", "mood": "wry guilt",
         "story_seed": "Paying exact change to a stranger who will never know your name or your crime.",
         "suggested_style_tags": ["indie rock"], "energy": 0.5,
         "placement_hint": "mid"},
    ],
    "recurring_motifs": ["unplayed voicemail", "mile markers"],
    "listener_experience_notes": "You should arrive feeling forgiven but not fixed.",
})


@pytest.fixture()
def patch_provider_factory(monkeypatch):
    def _install(providers_by_name):
        from app.services import llm_service as svc
        monkeypatch.setattr(
            svc.LLMService, "_get_provider",
            staticmethod(lambda override_config=None: providers_by_name[(override_config or {}).get("provider")]),
        )
    return _install


def _profiles(*names):
    return [ModelProfile(provider=n, model="test-model") for n in names]


@pytest.mark.asyncio
async def test_chain_order_first_success_wins(patch_provider_factory):
    first = FakeProvider([{"content": VALID_VISION_JSON}])
    second = FakeProvider([])  # must never be called
    patch_provider_factory({"alpha": first, "beta": second})

    outcome = await ResiliencePolicy().run_structured(
        [{"role": "user", "content": "go"}],
        ExperiencerVision,
        profiles=_profiles("alpha", "beta"),
    )
    assert outcome.structured.journey_title == "Neon Exodus"
    assert len(outcome.attempts) == 1 and outcome.attempts[0].ok is True
    assert second.calls == []  # never reached


@pytest.mark.asyncio
async def test_quota_failure_fails_over_to_next_provider(patch_provider_factory):
    failing = FakeProvider([{"raise": LLMQuotaError("429 rate limit exceeded")}])
    good = FakeProvider([{"content": VALID_VISION_JSON}])
    patch_provider_factory({"alpha": failing, "beta": good})

    outcome = await ResiliencePolicy().run_structured(
        [{"role": "user", "content": "go"}], ExperiencerVision,
        profiles=_profiles("alpha", "beta"),
    )
    assert outcome.structured.concept_statement.startswith("A driver")
    types = [(a.provider, a.error_type if not a.ok else "ok") for a in outcome.attempts]
    assert types == [("alpha", "LLMQuotaError"), ("beta", "ok")]


@pytest.mark.asyncio
async def test_parse_failure_fails_over_like_transient(patch_provider_factory):
    bad_json = FakeProvider([{"content": "I am sorry, I cannot comply. {broken"}])
    good = FakeProvider([{"content": VALID_VISION_JSON}])
    patch_provider_factory({"alpha": bad_json, "beta": good})

    outcome = await ResiliencePolicy().run_structured(
        [{"role": "user", "content": "go"}], ExperiencerVision,
        profiles=_profiles("alpha", "beta"),
    )
    assert outcome.attempts[0].ok is False
    assert outcome.attempts[0].error_type == "LLMParseError"
    assert outcome.structured.listener_experience_notes != ""


@pytest.mark.asyncio
async def test_auth_error_fails_over_without_backoff_delay(patch_provider_factory, monkeypatch):
    sleeps: List[float] = []
    monkeypatch.setattr("app.agents.runtime.policy.asyncio.sleep",
                        lambda s: (_ for _ in ()).throw(AssertionError(f"should not sleep {s}")))
    failing = FakeProvider([{"raise": LLMAuthError("401 invalid api key")}])
    good = FakeProvider([{"content": VALID_VISION_JSON}])
    patch_provider_factory({"alpha": failing, "beta": good})

    outcome = await ResiliencePolicy().run_structured(
        [{"role": "user", "content": "go"}], ExperiencerVision,
        profiles=_profiles("alpha", "beta"),
    )
    assert outcome.structured.journey_title == "Neon Exodus"
    assert sleeps == []


@pytest.mark.asyncio
async def test_all_providers_failed_carries_attempts(patch_provider_factory):
    a = FakeProvider([{"raise": LLMAuthError("401 nope")}])
    b = FakeProvider([{"raise": LLMQuotaError("429 slow down")}])
    patch_provider_factory({"alpha": a, "beta": b})

    with pytest.raises(AllProvidersFailedError) as excinfo:
        await ResiliencePolicy().run_structured(
            [{"role": "user", "content": "go"}], ExperiencerVision,
            profiles=_profiles("alpha", "beta"),
        )
    codes = [att["provider"] for att in excinfo.value.attempts]
    assert codes == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Experiencer contracts
# ---------------------------------------------------------------------------
def test_brief_rejects_out_of_bounds_track_target():
    with pytest.raises(ValidationError):
        AlbumBrief(album_title="X", album_concept="Y", track_target=0)
    with pytest.raises(ValidationError):
        AlbumBrief(album_title="X", album_concept="Y", track_target=31)


def test_vision_requires_minimum_arc_and_seed_content():
    payload = json.loads(VALID_VISION_JSON)
    payload["song_seeds"] = [dict(payload["song_seeds"][0], story_seed="too short")]
    with pytest.raises(ValidationError):
        ExperiencerVision.model_validate(payload)


def test_agent_build_messages_embeds_persona_history_and_target():
    agent = ExperiencerAgent()
    brief = AlbumBrief(album_title="Neon Exodus", album_concept="Night drive out of an old life.",
                       artist_name="Vela", artist_bio="Ex-radio host turned insomniac.",
                       tags="synthwave", track_target=7, extra_direction="more rain")
    ctx = RunContext(agent_name="experiencer", history=[
        {"role": "user", "content": "earlier thought"},
        {"role": "producer", "content": "earlier producer note"},
    ])
    msgs = agent.build_messages(brief, ctx)
    assert msgs[0]["role"] == "system" and "THE EXPERIENCER" in msgs[0]["content"]
    joined = "\n".join(m["content"] for m in msgs)
    assert "Neon Exodus" in joined and "Vela" in joined and "more rain" in joined
    assert "Produce exactly 7 song_seeds" in joined
    # history mapped: producer → assistant, and both precede the live ask
    assert msgs[1] == {"role": "user", "content": "earlier thought"}
    assert msgs[2] == {"role": "assistant", "content": "earlier producer note"}
    assert msgs[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_experiencer_run_reports_shortfall(patch_provider_factory):
    short_json = json.loads(VALID_VISION_JSON)
    short_json["song_seeds"] = short_json["song_seeds"][:1]
    provider = FakeProvider([{"content": json.dumps(short_json)}])
    patch_provider_factory({"only": provider})

    agent = ExperiencerAgent()
    brief = AlbumBrief(album_title="Neon Exodus", album_concept="Night drive out of an old life.",
                       track_target=10)
    result = await agent.run(brief, RunContext(agent_name="experiencer"),
                             policy=ResiliencePolicy(chain=_profiles("only")))
    assert isinstance(result.vision, ExperiencerVision)
    assert "Requested 10 song seeds but the model returned 1" in result.shortfall_notes
    assert result.tokens_in == 11 and result.tokens_out == 7  # usage captured, not dropped
