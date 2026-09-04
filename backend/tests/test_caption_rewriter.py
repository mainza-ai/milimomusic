"""
Caption rewriter tests (production behavior).

Pure-logic tests are deterministic. The failure-path test uses a REAL provider
configuration pointing at an unreachable endpoint (no monkeypatching of classes)
to prove the rewriter degrades gracefully instead of raising.
"""

import pytest

from app.services import llm_service as L
from app.services.llm_service import LLMService


def test_rank_families_routes_genre_tokens():
    ranked = L._rank_caption_families("disco funk pop with an electronic dance feel")
    assert ranked, "expected at least one routed family"
    assert "dance-pop-disco-funk" in ranked[:2] or "club-edm-house-trance" in ranked[:2], ranked


def test_pick_caption_templates_matches_routed_family():
    families = L._rank_caption_families("disco funk pop electronic")
    templates = L._pick_caption_templates("disco funk pop 122 bpm female vocals", families, k=3)
    assert templates, "expected template matches"
    for t in templates:
        assert t.endswith(".txt")
        assert L._read_caption_file(t).strip(), f"{t} unreadable"


def test_parse_caption_response_requires_all_three_sections():
    good = {
        "global_metadata": "Genre X.",
        "vocal_details": "Vocal Y.",
        "arrangement": "Arrangement Z.",
    }
    assert L.LLMService._parse_caption_response(good) == good
    partial = {"global_metadata": "only one section"}
    assert L.LLMService._parse_caption_response(partial) is None
    assert L.LLMService._parse_caption_response("not a dict") is None


def test_constructed_caption_is_complete_and_honest():
    cap = L.LLMService._constructed_caption("A nostalgic night drive", "Synthwave, 118 BPM")
    assert len(cap) == 3
    assert "Synthwave" in cap["global_metadata"]
    assert "Vocal Gender" in cap["vocal_details"]
    assert "Instrument Lifecycle" in cap["arrangement"]


@pytest.mark.asyncio
async def test_rewrite_caption_falls_back_when_llm_unreachable():
    # REAL provider config pointed at a dead endpoint — the rewriter must not
    # raise and must return a complete, honestly-flagged fallback caption.
    dead = {"provider": "ollama", "ollama": {"base_url": "http://127.0.0.1:1"}}
    result = await L.LLMService.rewrite_caption(
        concept="A nostalgic synthwave night drive",
        lyrics="[Verse 1]\nStars over the freeway\n[Chorus]\nWe drive all night",
        tags="Synthwave, 118 BPM",
        provider_config=dead,
    )
    assert result["rewritten"] is False
    assert result["fallback_reason"], "expected an honest reason"
    assert result["structured_caption"] is not None
    assert len(result["structured_caption"]) == 3
