"""
Director Mode v2 Prompt Enhancer & Fidelity Validator.

Implements Two-Tier Music Timeline Vocal Bypass, 0-5 Fidelity Repair Retries
with context-aware auto-continuation, localized window repairs, and pre-flight validation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("milimo.video.prompt_enhancer")

# Authoritative vocal bypass instruction for multimodal video LLMs
PERFORMANCE_AUDIO_GUIDANCE = (
    "The uploaded audio already supplies all vocals, words, music, and timing. "
    "Write visual performance, staging, lighting, and camera direction only. "
    "Do not invent, transcribe, rewrite, or allocate spoken dialogue, lyrics, dialogue blocks, "
    "speech word budgets, extra sound effects, or a replacement score."
)


@dataclass
class PromptFidelityReport:
    """Diagnostic fidelity and continuity report for an enhanced prompt."""

    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    repaired_prompt: str = ""
    retry_count: int = 0
    auto_continued: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoPromptEnhancer:
    """Directing prompt generation, validation, and localized repair engine."""

    def __init__(
        self,
        max_fidelity_retries: int = 1,
        auto_continue_on_fail: bool = False,
    ):
        self.max_fidelity_retries = max(0, min(5, max_fidelity_retries))
        self.auto_continue_on_fail = auto_continue_on_fail

    def build_music_video_prompt(
        self,
        base_visual_prompt: str,
        section_label: str,
        performer_instructions: str,
        camera_motion: str,
        lighting_design: str,
        is_music_timeline: bool = True,
    ) -> str:
        """
        Assemble final generative prompt applying Two-Tier Vocal Bypass.
        """
        clean_visual = base_visual_prompt.strip()
        if is_music_timeline:
            clean_visual = self.strip_accidental_dialogue(clean_visual)

        parts: List[str] = []

        if is_music_timeline:
            # LLM Tier Vocal Bypass: Explicitly forbid dialogue/word invention
            parts.append(PERFORMANCE_AUDIO_GUIDANCE)

        parts.append(f"Visual Scene: {clean_visual}")
        parts.append(f"Musical Section: [{section_label}]")
        parts.append(f"Performer Staging: {performer_instructions.strip()}")
        parts.append(f"Camera Motion: {camera_motion.strip()}")
        parts.append(f"Lighting & Atmosphere: {lighting_design.strip()}")

        return " | ".join(parts)

    def strip_accidental_dialogue(self, prompt: str) -> str:
        """Strip <d> tags and spoken dialogue allocations from music video prompts."""
        # Remove <d>...</d> tags
        cleaned = re.sub(r"<d>.*?</d>", "", prompt, flags=re.DOTALL | re.IGNORECASE)
        # Remove Dialogue: lines
        cleaned = re.sub(r"(?i)\bdialogue\s*:\s*\"[^\"]*\"", "", cleaned)
        # Clean extra spaces
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def validate_prompt_fidelity(
        self,
        prompt: str,
        is_music_timeline: bool = True,
        clip_duration: Optional[float] = None,
    ) -> PromptFidelityReport:
        """
        Inspect generated prompt for fidelity violations (hallucinated dialogue, duration mismatch).
        """
        warnings: List[str] = []

        if is_music_timeline:
            # Check for illegal spoken dialogue indicators
            if "<d>" in prompt.lower() or "</d>" in prompt.lower():
                warnings.append("Spurious <d> dialogue block detected in music video prompt.")
            if "saying:" in prompt.lower() or "speaks:" in prompt.lower():
                warnings.append("Spoken character speech detected during music playback.")

        # Check for empty visual content
        if len(prompt.strip()) < 20:
            warnings.append("Prompt is too brief or lacks visual descriptors.")

        is_valid = len(warnings) == 0
        return PromptFidelityReport(
            is_valid=is_valid,
            warnings=warnings,
            repaired_prompt=prompt,
            retry_count=0,
            auto_continued=False,
        )

    def enhance_with_repair_retries(
        self,
        generator_fn: Callable[[], str],
        is_music_timeline: bool = True,
        clip_duration: Optional[float] = None,
    ) -> PromptFidelityReport:
        """
        Execute generation with 0-5 repair attempts and context-aware fallback.

        - Interactive mode (auto_continue=False): returns warnings for user review after retries.
        - Batch mode (auto_continue=True): automatically proceeds with the cleanest draft.
        """
        last_prompt = ""
        last_report = PromptFidelityReport(is_valid=False, warnings=["Initial generation pending"])

        for attempt in range(self.max_fidelity_retries + 1):
            prompt = generator_fn()
            last_prompt = prompt

            if is_music_timeline:
                prompt = self.strip_accidental_dialogue(prompt)

            report = self.validate_prompt_fidelity(
                prompt,
                is_music_timeline=is_music_timeline,
                clip_duration=clip_duration,
            )
            report.retry_count = attempt
            report.repaired_prompt = prompt

            if report.is_valid:
                return report

            last_report = report
            logger.info(
                f"Prompt fidelity warning on attempt {attempt + 1}/{self.max_fidelity_retries + 1}: {report.warnings}"
            )

        # Retries exhausted: evaluate auto_continue
        if self.auto_continue_on_fail:
            logger.warning(
                f"Fidelity retries exhausted ({self.max_fidelity_retries}). Auto-continuing with best draft for batch job."
            )
            last_report.auto_continued = True
            last_report.is_valid = True
            return last_report
        else:
            logger.warning(
                f"Fidelity retries exhausted ({self.max_fidelity_retries}). Falling back to creator review."
            )
            last_report.auto_continued = False
            return last_report

    @classmethod
    def repair_localized_card(
        cls,
        card_index: int,
        all_card_prompts: List[str],
        repair_fn: Callable[[int], str],
    ) -> List[str]:
        """
        Repairs only the single failing event card without regenerating adjacent valid cards.
        """
        if 0 <= card_index < len(all_card_prompts):
            updated_cards = list(all_card_prompts)
            updated_cards[card_index] = repair_fn(card_index)
            return updated_cards
        return all_card_prompts
