"""Hooked MiniMax Music 3 local inference: true cancellation + live progress.

The upstream library decodes autoregressively (duration x 25 frames) with no
cancellation surface and yields only at completion — multi-hour black boxes.
This module re-implements ONLY the orchestration loops (~80 lines), calling the
library's own kernels (ar_one_frame, denoise_chunk, vocoder, ...), injecting:

  * cancel_event checks every CHECK_EVERY frames  -> preemption in seconds
  * progress_cb(fraction_done, stage)             -> honest % for SSE/UI

Measured RTF on M3 Max ≈ 130x realtime: a 60s track ≈ 2h GPU. ETAs derive from
this, not hope.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import mlx.core as mx

    from mlx_audio.music.models.minimax_music3.ar import (
        ar_one_frame,
        qwen3_hidden,
    )
    from mlx_audio.music.models.minimax_music3.minimax_music3 import (
        CHUNK_FRAMES,
        CHUNK_HOP,
        DIT_CFG_SCALE,
        OVERLAP_LATENT_LENGTH,
        _chunk_starts,
        _crop_waveform,
        denoise_chunk,
    )
    from mlx_audio.audio_io import write as audio_write
    HAS_MLX_AUDIO = True
except ImportError:
    mx = None
    ar_one_frame = None
    qwen3_hidden = None
    CHUNK_FRAMES = 0
    CHUNK_HOP = 0
    DIT_CFG_SCALE = 0.0
    OVERLAP_LATENT_LENGTH = 0
    _chunk_starts = None
    _crop_waveform = None
    denoise_chunk = None
    audio_write = None
    HAS_MLX_AUDIO = False

logger = logging.getLogger("milimo.minimax_local")

ProgressCB = Callable[[float, str], None]
CHECK_EVERY = 25          # frames between cancel checks (~every few seconds)
PROGRESS_EVERY = 100      # frames between progress emissions


class GenerationCancelled(Exception):
    pass


def _check(cancel_event: Optional[threading.Event]):
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Cancelled by user during local inference")


def _report(cb: Optional[ProgressCB], frac: float, stage: str):
    if cb is not None:
        try:
            cb(min(0.99, frac), stage)
        except Exception:  # progress must never kill generation
            logger.exception("progress callback failed")


try:
    from mlx_audio.music.models.minimax_music3.ar import (
        lm_logits,
        sample_top_k,
        generate_depth_codes,
        fuse_frame_hiddens,
        _embed_audio_frame,
        ARFrame,
    )
except ImportError:
    lm_logits = None
    sample_top_k = None
    generate_depth_codes = None
    fuse_frame_hiddens = None
    _embed_audio_frame = None
    ARFrame = None


def ar_one_frame_tuned(
    language_model,
    depth,
    config,
    last_hidden,
    cache,
    rng_key,
    emit_frame: bool = True,
    cfg_scale: float = 1.5,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """Autoregressive step respecting user-specified temperature, CFG scale, and top_k."""
    if lm_logits is None:
        return ar_one_frame(language_model, depth, config, last_hidden, cache, rng_key, emit_frame=emit_frame)

    logits = lm_logits(language_model, last_hidden).astype(mx.float32)
    token_ids = mx.arange(logits.shape[-1])
    allowed = mx.logical_or(
        mx.logical_and(
            token_ids >= config.audio_code_offset,
            token_ids < config.audio_code_offset + config.semantic_vocab_size,
        ),
        token_ids == config.audio_end_token_id,
    )
    logits = mx.where(allowed[None, :], logits, -1e9)

    # 1. Temperature scaling on logits
    temp = max(0.05, float(temperature))
    scaled_logits = logits / temp

    # 2. CFG Guidance in AR space
    conditional, unconditional = scaled_logits[:1], scaled_logits[1:2]
    ar_cfg = max(0.1, min(5.0, float(cfg_scale)))
    guided = unconditional + (conditional - unconditional) * ar_cfg

    # 3. Top-K cutoff
    effective_top_k = max(1, min(int(top_k), conditional.shape[-1]))
    threshold = mx.min(mx.topk(conditional, effective_top_k, axis=-1), axis=-1, keepdims=True)
    guided = mx.where(conditional < threshold, -1e9, guided)
    guided = mx.where(allowed[None, :], guided, -1e9)
    sampled, key = sample_top_k(guided, rng_key, effective_top_k)

    if int(sampled.item()) == config.audio_end_token_id:
        return ARFrame(
            semantic_code=sampled,
            residual_codes=mx.zeros((2, config.residual_codebooks), dtype=mx.int32),
            frame_hidden=mx.zeros((1, config.num_codebooks * config.hidden_size)),
            last_hidden=last_hidden,
            cache=cache,
            ended=True,
        )

    semantic_code = (
        mx.concatenate([sampled, sampled], axis=0) - config.audio_code_offset
    )
    frame_codes, depth_hidden, key = generate_depth_codes(
        language_model,
        depth,
        config,
        last_hidden,
        semantic_code,
        key,
    )
    frame_hidden = (
        fuse_frame_hiddens(last_hidden[:1], depth_hidden)
        if emit_frame
        else mx.zeros((1, config.num_codebooks * config.hidden_size))
    )
    feedback = _embed_audio_frame(language_model, depth, config, frame_codes)
    hidden, cache = qwen3_hidden(language_model, feedback, cache)
    return ARFrame(
        semantic_code=semantic_code,
        residual_codes=frame_codes[:, 1:],
        frame_hidden=frame_hidden,
        last_hidden=hidden[:, -1],
        cache=cache,
        ended=False,
    )


def generate_frame_hiddens_hooked(
    language_model,
    depth_decoder,
    config,
    text_ids,
    max_frames: int,
    seed: int,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    top_k: int = 50,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[ProgressCB] = None,
):
    """Mirror of library ar.generate_frame_hiddens with hooks and sampling controls injected."""
    if not HAS_MLX_AUDIO:
        raise RuntimeError("mlx_audio is not available on this platform.")
    mx.random.seed(seed)
    key = mx.random.key(seed)
    embeddings = language_model.model.embed_tokens(text_ids)
    hidden, cache = qwen3_hidden(language_model, embeddings)
    last_hidden = hidden[:, -1]
    frames = []
    started = time.monotonic()
    for frame_index in range(max_frames + 1):
        if frame_index % CHECK_EVERY == 0:
            _check(cancel_event)
            if progress_cb and frame_index % PROGRESS_EVERY == 0 and frame_index:
                rate = frame_index / max(1e-9, time.monotonic() - started)
                eta_s = (max_frames - frame_index) / max(rate, 1e-9)
                _report(progress_cb, frame_index / max_frames,
                        f"Composing ({int(rate)} frames/s, ETA {int(eta_s / 60)}m)")
        key, subkey = mx.random.split(key)
        result = ar_one_frame_tuned(
            language_model, depth_decoder, config,
            last_hidden, cache, subkey,
            emit_frame=frame_index > 0,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
        )
        last_hidden, cache = result.last_hidden, result.cache
        if result.ended:
            break
        if frame_index > 0:
            frames.append(result.frame_hidden)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError("MiniMax Music 3 generated zero audio frames")
    return mx.stack(frames, axis=1)


def run_flow_hooked(model, frame_hiddens, num_inference_steps: int, seed: int,
                    cfg_scale: float = 1.5,
                    cancel_event=None, progress_cb=None):
    """Mirror of model._run_flow with per-chunk cancel checks and dynamic CFG guidance scale."""
    starts = _chunk_starts(frame_hiddens.shape[1])
    waves = []
    previous_latent = None
    previous_condition = None
    mx.random.seed(seed + 7)
    total = len(starts)
    effective_cfg = float(cfg_scale) if cfg_scale is not None and cfg_scale > 0 else DIT_CFG_SCALE
    for index, start in enumerate(starts):
        _check(cancel_event)
        _report(progress_cb, index / max(total, 1),
                f"Rendering audio chunk {index + 1}/{total}")
        end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
        condition = model.condition_encoder(frame_hiddens[:, start:end])
        noise = mx.random.normal(
            (1, model.config.dit_in_channels, condition.shape[1])
        ).astype(condition.dtype)
        latents, condition = denoise_chunk(
            model.transformer, noise, condition,
            num_inference_steps=num_inference_steps,
            guidance_scale=effective_cfg,
            previous_latent=previous_latent,
            previous_condition=previous_condition,
        )
        carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
        carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[:, carry_start:carry_end]
        waves.append(model.vocoder(latents))
    return mx.concatenate(
        [_crop_waveform(wave, i, len(waves)) for i, wave in enumerate(waves)],
        axis=-1,
    )


def generate_music_hooked(
    model, caption: str, lyrics: str, duration_sec: float,
    steps: int, seed: int, output_path: str,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    top_k: int = 50,
    cancel_event=None, progress_cb=None
) -> str:
    """Full hooked pipeline with genuine temperature, CFG, and top_k sampling controls."""
    max_frames = max(1, int(duration_sec * model.config.frame_rate))
    text_ids = model._text_ids(caption, lyrics)

    def ar_progress(frac, msg):
        # AR phase is ~90% of wall time; map to first 85% of overall bar.
        _report(progress_cb, frac * 0.85, msg)

    frame_hiddens = generate_frame_hiddens_hooked(
        model.language_model, model.rvq_depth_decoder, model.config,
        text_ids, max_frames, seed,
        temperature=temperature,
        cfg_scale=cfg_scale,
        top_k=top_k,
        cancel_event=cancel_event,
        progress_cb=ar_progress,
    )
    mx.eval(frame_hiddens)
    _check(cancel_event)
    audio = run_flow_hooked(
        model, frame_hiddens, steps, seed,
        cfg_scale=cfg_scale,
        cancel_event=cancel_event,
        progress_cb=progress_cb
    )
    mx.eval(audio)
    _check(cancel_event)
    waveform = mx.clip(audio[0].transpose(1, 0).astype(mx.float32), -1.0, 1.0)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio_write(destination, waveform, model.sample_rate)
    return str(destination)
