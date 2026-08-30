"""ResiliencePolicy — THE single failover authority for agent LLM calls.

Replaces the audit's "triplicated folklore" (three diverging hardcoded chains
in llm_service) with one policy object that:

  * resolves an ordered provider chain from live config (active provider
    first, then key-bearing cloud providers, then local engines);
  * executes each attempt OFF the event loop (asyncio.to_thread) — fixing
    gap G2, the biggest scalability defect;
  * classifies every failure into the typed hierarchy (G7) and reacts by
    kind: auth/bad-model skip immediately, quota/timeout back off;
  * treats a schema-parse failure as a provider-level failure too, so a model
    that can't emit valid JSON fails over to one that can (parse-failover);
  * records every attempt for usage accounting (G5).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from app.core.llm_contracts import (
    AllProvidersFailedError,
    LLMAuthError,
    LLMBadModelError,
    LLMParseError,
    LLMQuotaError,
    LLMResult,
    extract_json_object,
)
from app.agents.runtime.usage import AttemptRecord, RunUsage


class _RetrySameProvider(Exception):
    """Internal: parse failed but repair budget remains on THIS provider."""


@dataclass
class ModelProfile:
    """One candidate slot in the failover chain."""

    provider: str
    model: Optional[str] = None          # None → configured/default model
    temperature: float = 0.8
    timeout: float = 60.0

    def to_dict(self) -> Dict:
        return {"provider": self.provider, "model": self.model or "(configured)",
                "temperature": self.temperature}


@dataclass
class PolicyOutcome:
    content: str
    structured: Optional[BaseModel]
    result: LLMResult                    # the winning attempt's envelope
    attempts: List[AttemptRecord] = field(default_factory=list)


# Stable candidate order after the active provider. Cloud first (key-gated),
# local engines as the universal safety net.
_CLOUD_ORDER = ["nvidia", "deepseek", "openai", "gemini", "openrouter", "opencode"]
_LOCAL_ORDER = ["omlx", "ollama", "lmstudio"]


class ResiliencePolicy:
    def __init__(
        self,
        chain: Optional[Sequence[ModelProfile]] = None,
        parse_retries: int = 2,
        chain_head: Optional[ModelProfile] = None,
    ):
        self._explicit_chain: Optional[List[ModelProfile]] = list(chain) if chain else None
        # Crew override (per-artist assignment/profile): pinned as the FIRST
        # candidate; the auto-resolved global chain remains the failover.
        self._chain_head: Optional[ModelProfile] = chain_head
        self.parse_retries = max(0, parse_retries)

    # ------------------------------------------------------------------
    # Chain resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _has_credentials(cfg: Dict[str, Any], provider: str) -> bool:
        block = cfg.get(provider, {}) or {}
        api_key = (block.get("api_key") or "").strip()
        env_key = ""
        if provider == "nvidia":
            env_key = os.environ.get("NVIDIA_API_KEY", "")
        elif provider == "opencode":
            env_key = os.environ.get("OPENCODE_API_KEY", "")
        if provider in ("ollama", "lmstudio", "omlx"):
            return True  # local engines need no key
        return bool(api_key or env_key)

    def resolve_chain(self) -> List[ModelProfile]:
        if self._explicit_chain is not None:
            return list(self._explicit_chain)

        from app.services.llm_service import LLMService  # late import: no cycle at module load

        base = ConfigManager_safe_config()
        active = (base.get("provider") or "").strip()
        chain: List[ModelProfile] = []

        def push(provider: str, model: Optional[str] = None) -> None:
            if not provider or any(p.provider == provider for p in chain):
                return
            block = base.get(provider, {}) or {}
            chain.append(ModelProfile(
                provider=provider,
                model=(model or block.get("model") or "").strip() or None,
            ))

        # Crew override rides first; push() keeps the rest of the chain
        # duplicate-free (same-provider global entries are skipped).
        if self._chain_head is not None:
            head = self._chain_head
            block = base.get(head.provider, {}) or {}
            chain.append(ModelProfile(
                provider=head.provider,
                model=(head.model or block.get("model") or "").strip() or None,
            ))

        if active and (active in ("ollama", "lmstudio", "omlx") or self._has_credentials(base, active)):
            push(active)
        for p in _CLOUD_ORDER:
            if p != active and self._has_credentials(base, p):
                push(p)
        for p in _LOCAL_ORDER:
            if p != active:
                push(p)

        if not chain:  # absolute last resort — local engine always exists
            push("ollama")
        return chain

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_override(profile: ModelProfile) -> Dict[str, Any]:
        from app.services.config_manager import ConfigManager
        base = ConfigManager_safe_config()
        block = dict(base.get(profile.provider, {}) or {})
        if profile.model:
            block["model"] = profile.model
        return {**base, "provider": profile.provider, profile.provider: block}

    def _instantiate(self, profile: ModelProfile):
        from app.services.llm_service import LLMService
        return LLMService._get_provider(self._build_override(profile))

    def _model_for(self, profile: ModelProfile) -> str:
        if profile.model:
            return profile.model
        from app.services.config_manager import ConfigManager
        base = ConfigManager_safe_config()
        return ((base.get(profile.provider, {}) or {}).get("model") or "").strip()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def run_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        *,
        profiles: Optional[Sequence[ModelProfile]] = None,
        temperature: Optional[float] = None,
        max_providers: int = 5,
        timeout: Optional[float] = None,
    ) -> PolicyOutcome:
        # Per-attempt ceiling. Large instruct models (120B-class) need real
        # headroom for long structured output — env-configurable so testing
        # and production can tune independently of code.
        if timeout is None:
            try:
                timeout = float(os.environ.get("MILIMO_AGENT_TIMEOUT", "60"))
            except ValueError:
                timeout = 60.0
        """Run `messages` through the chain until one provider returns JSON
        that validates against `schema`. Parse failures consume a per-provider
        repair budget (corrective round-trip) BEFORE failing over; every call
        is recorded for usage accounting."""
        chain = list(profiles) if profiles is not None else self.resolve_chain()
        chain = chain[:max(1, max_providers)]
        usage = RunUsage()

        for index, profile in enumerate(chain):
            model = self._model_for(profile)
            if not model:
                usage.add(AttemptRecord(
                    provider=profile.provider, model="(no-model)", ok=False,
                    error_type="LLMBadModelError",
                    error_message="No model configured for this provider.",
                ))
                continue

            provider = self._instantiate(profile)
            started = asyncio.get_event_loop().time()
            call_messages = messages
            repairs_left = self.parse_retries

            while True:
                try:
                    # OFF the event loop (gap G2): sync SDK call in a worker
                    # thread; kwargs pass straight through to generate_chat.
                    result: LLMResult = await asyncio.to_thread(
                        provider.generate_chat,
                        call_messages,
                        model,
                        options={"temperature": temperature or profile.temperature},
                        timeout=timeout,
                        force_json=True,  # constrained decoding at the provider
                    )
                    latency_ms = int((asyncio.get_event_loop().time() - started) * 1000)

                    try:
                        parsed = schema.model_validate(extract_json_object(result.content))
                    except Exception as pe:
                        usage.add(AttemptRecord(
                            provider=profile.provider, model=model, ok=False,
                            latency_ms=latency_ms, error_type="LLMParseError",
                            error_message=f"{profile.provider}: {pe}",
                        ))
                        if repairs_left > 0:
                            repairs_left -= 1
                            call_messages = list(call_messages) + [
                                {"role": "assistant", "content": result.content},
                                {"role": "user", "content":
                                    "Your previous response was not valid JSON "
                                    f"({type(pe).__name__}). Return ONLY the corrected JSON object — no prose, "
                                    "no markdown fences, escape all quotes inside string values."},
                            ]
                            continue  # corrective round-trip: SAME provider
                        break  # repair budget exhausted → next provider

                    attempt = AttemptRecord(
                        provider=profile.provider, model=model, ok=True,
                        latency_ms=result.latency_ms or latency_ms,
                        tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                    )
                    usage.add(attempt)
                    return PolicyOutcome(
                        content=result.content,
                        structured=parsed,
                        result=result,
                        attempts=usage.attempts,
                    )

                except Exception as exc:  # noqa: BLE001 — classified below
                    latency_ms = int((asyncio.get_event_loop().time() - started) * 1000)
                    typed = exc if isinstance(exc, (
                        LLMAuthError, LLMBadModelError, LLMQuotaError
                    )) else _classify(exc, profile.provider)
                    usage.add(AttemptRecord(
                        provider=profile.provider, model=model, ok=False,
                        latency_ms=latency_ms,
                        error_type=type(typed).__name__,
                        error_message=str(typed),
                    ))
                    if isinstance(typed, (LLMAuthError, LLMBadModelError)):
                        break  # provider cannot work — next provider NOW
                    await asyncio.sleep(min(0.4 * (index + 1), 2.0))
                    break  # transient — next provider

        raise AllProvidersFailedError(
            "All LLM providers in the resilience chain failed.",
            attempts=[a.to_dict() for a in usage.attempts],
        )


def _classify(exc: Exception, provider: str):
    """Local classifier shim so policy doesn't import adapter internals."""
    from app.core.llm_contracts import classify_llm_error
    return classify_llm_error(provider, exc)


def ConfigManager_safe_config() -> Dict[str, Any]:
    """Config snapshot with env overlays applied (the same view the UI sees)."""
    from app.services.config_manager import ConfigManager
    return ConfigManager().get_config()


__all__ = [
    "ModelProfile", "PolicyOutcome", "ResiliencePolicy",
]
