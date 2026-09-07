"""
Canonical modality taxonomy for Milimo Music (production source of truth).

Single owner for repo -> category (audio | image | video) decisions.
Precedence (highest wins):
  1. Explicit user override (handled by callers, not here).
  2. Hugging Face pipeline_tag (exact map, includes H3's `image-text-to-video`).
  3. HF tags / card data keywords (e.g. `text-to-video`, `hailuo`).
  4. Repo-ID keyword rules with word-boundary-ish matching + org disambiguation.
  5. Filename signals (diffusion transformer layouts vs audio codec layouts).
  6. Safe fallback: audio + needs_review=True (never silent "custom").

Why this exists: MiniMax owns BOTH music (audio) and H3/Hailuo (video), so a
bare `minimax` substring must never decide alone. Video disambiguators
(h3, hailuo, image-text-to-video) must outrank the org name.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

VALID_CATEGORIES = ("audio", "image", "video")

# Exact pipeline_tag -> category. Keep exhaustive for HF's video family.
PIPELINE_TAG_MAP: Dict[str, str] = {
    # video (note: H3 uses image-text-to-video, previously missing everywhere)
    "text-to-video": "video",
    "image-to-video": "video",
    "image-text-to-video": "video",
    "video-to-video": "video",
    "text-to-video-synthesis": "video",
    "zero-shot-video-classification": "video",
    # audio
    "text-to-audio": "audio",
    "audio-to-audio": "audio",
    "automatic-speech-recognition": "audio",
    "voice-conversion": "audio",
    "text-to-speech": "audio",
    "audio-classification": "audio",
    # image
    "text-to-image": "image",
    "image-to-image": "image",
    "image-to-video": "video",  # kept here for readability; video wins
    "unconditional-image-generation": "image",
    "image-classification": "image",
}

# Keyword rules evaluated in order: (category, [keywords]).
# Matching is substring on lowercased repo id, but video is checked before
# audio so MiniMax-H3/Hailuo can outrank the bare `minimax` audio signal.
# `minimax` itself is deliberately NOT an audio keyword anymore; music
# variants are matched via `music`, `music3`, `mxfp4`, etc.
VIDEO_KEYWORDS = [
    "minimax-h3", "hailuo", "image-text-to-video",
    "h3-mlx", "/h3", "h3-", "-h3",
    "text-to-video", "image-to-video", "video-to-video",
    "cogvideo", "hunyuanvideo", "hunyuan-video", "wan2", "wan-2", "wanvideo",
    "ltx-video", "mochi", "sora", "pika", "runway",
    "stable-video", "animatediff",
    "video", "t2v", "i2v",
]
IMAGE_KEYWORDS = [
    "flux", "sdxl", "sd-xl", "stable-diffusion-image", "lora-art",
    "text-to-image", "image-to-image",
    "controlnet", "midjourney",
    "image", "paint", "illustration",
]
AUDIO_KEYWORDS = [
    "minimax-music", "music3", "music-3", "mxfp4",
    "musicgen", "audiocraft", "musiclm",
    "text-to-audio", "audio-to-audio",
    "voice-conversion", "text-to-speech",
    "whisper", "wav2vec", "hubert", "bark", "vall-e", "xtts",
    "music", "audio", "sound", "voice", "speech", "tts", "vocoder",
]


def _contains_any(haystack: str, needles: List[str]) -> Optional[str]:
    for n in needles:
        if n and n in haystack:
            return n
    return None


def infer_modality(
    repo_id: str = "",
    filenames: Optional[List[str]] = None,
    pipeline_tag: Optional[str] = None,
    hf_tags: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Return (category, reason). Never returns 'custom'; unknown -> audio+review."""
    low_repo = (repo_id or "").lower()
    pipe = (pipeline_tag or "").strip().lower()

    # 1. Exact pipeline tag.
    if pipe and pipe in PIPELINE_TAG_MAP:
        return PIPELINE_TAG_MAP[pipe], f"pipeline_tag:{pipe}"

    # 2. HF tags / card keywords.
    tag_blob = " ".join(t.lower() for t in (hf_tags or []) if t)
    if tag_blob:
        if any(k in tag_blob for k in ("video", "hailuo", "h3", "t2v", "i2v", "diffusion-video")):
            # disambiguate: music tags win back to audio only if explicitly musical
            if "music" not in tag_blob and "audio" not in tag_blob:
                return "video", "hf_tags:video-signal"
        if any(k in tag_blob for k in ("text-to-image", "image-generation", "flux", "sdxl")):
            return "image", "hf_tags:image-signal"
        if any(k in tag_blob for k in ("audio", "music", "speech", "tts")):
            return "audio", "hf_tags:audio-signal"

    # 3. Repo-id keyword rules: video > image > audio (org-disambiguated).
    hit = _contains_any(low_repo, VIDEO_KEYWORDS)
    if hit:
        return "video", f"repo_keyword:{hit}"
    hit = _contains_any(low_repo, IMAGE_KEYWORDS)
    if hit:
        # Guard: generic words like `image`/`paint` should not beat an
        # explicit music signal in the same id (rare, but deterministic).
        if "music" not in low_repo and "audio" not in low_repo:
            return "image", f"repo_keyword:{hit}"
    hit = _contains_any(low_repo, AUDIO_KEYWORDS)
    if hit:
        return "audio", f"repo_keyword:{hit}"

    # 4. Filename signals.
    if filenames:
        blob = " ".join(filenames).lower()
        # video DiT layouts (H3/Wan/Hunyuan/CogVideoX style)
        if any(k in blob for k in (
            "transformer/blocks", "transformer_blocks", "dit_block",
            "video_vae", "video-vae", "causal_video", "patchify",
            "minimax_h3", "hailuo",
            "wan2", "hunyuan", "cogvideo",
        )):
            return "video", "filenames:video-dit-signal"
        if any(k in blob for k in ("vae/diffusion", "unet", "transformer_blocks", "text_encoder_2", "text_encoder", ".vae.")):
            # unet/vae/text-encoder layouts without video or audio-codec markers
            # => image diffusion (keeps bare `*.vae.safetensors` compat).
            if "audio" not in blob and "codec" not in blob and "rvq" not in blob:
                return "image", "filenames:image-diffusion-signal"
        if any(k in blob for k in ("codec", "mel", "vocoder", "rvq", "encodec", "audioseal")):
            return "audio", "filenames:audio-codec-signal"

    return "audio", "fallback:needs_review"


def infer_category(
    repo_id: str = "",
    filenames: Optional[List[str]] = None,
    pipeline_tag: Optional[str] = None,
    hf_tags: Optional[List[str]] = None,
) -> str:
    """Back-compat wrapper returning just the category string."""
    cat, _ = infer_modality(repo_id, filenames, pipeline_tag, hf_tags)
    return cat
