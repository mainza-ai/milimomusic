"""
Real Neural Source Separation (Demucs / HTDemucs).

Separates the ACTUAL generated master audio into real per-source stems
(vocals, drums, bass, other) using Meta's open-source HTDemucs model. This is
genuine source separation on the real audio waveform — never synthesized
oscillators pretending to be instruments.

The model loads lazily and is cached as a process singleton so it is loaded
only once per server lifetime. It runs on MPS (Apple Silicon), CUDA, or CPU.
On Apple Silicon the MPS-enable-fallback flag is set because HTDemucs uses one
conv1d op the MPS backend does not support natively (falls back to CPU for it).
"""

import os
import time
import logging
import threading

# HTDemucs uses a conv1d whose output channels exceed the MPS limit; this flag
# lets those ops fall back to CPU. Must be set before torch/conv sees it.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
STEM_DIR = "generated_audio/stems"

# htdemucs fixed output source order (names and channel order are both derived
# from the model itself; we keep the canonical 4).
_HTDEMUCS_SOURCES = ("drums", "bass", "other", "vocals")

_model = None
_model_lock = threading.Lock()
_model_load_time = 0.0


def _load_model():
    """Load (and cache) the htdemucs model on the best available device."""
    global _model, _model_load_time
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        t0 = time.time()
        import torch
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        # Run on CUDA when available, otherwise CPU. We deliberately do NOT use
        # MPS: htdemucs uses a conv1d op with >65536 output channels that the
        # MPS backend cannot run (even with PYTORCH_ENABLE_MPS_FALLBACK the op
        # is unsupported in this torch release), so CPU/CUDA is the reliable,
        # cross-platform path. Inference runs in a worker thread.
        if torch.cuda.is_available():
            logger.info("real_separator: htdemucs on CUDA")
            model = model.to("cuda")
        else:
            logger.info("real_separator: htdemucs on CPU")
        model.eval()
        _model = model
        _model_load_time = time.time() - t0
        logger.info(f"real_separator: htdemucs loaded in {_model_load_time:.1f}s")
        return model


def _torch_device():
    # Must match model placement in _load_model: CUDA if available else CPU.
    # Never MPS — htdemucs' conv1d with >65536 output channels is unsupported
    # on the MPS backend in this torch release.
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unload_model():
    """Release the cached htdemucs model from memory.

    Keeps both heavy models (MiniMax MLX ~28GB + HTDemucs) from staying resident
    at once after separation is done. Safe to call anytime — the model is lazily
    reloaded on the next ``separate_sources`` call (~seconds).
    """
    global _model
    with _model_lock:
        if _model is not None:
            import gc
            _model = None
            gc.collect()
            logger.info("real_separator: htdemucs model released from memory.")


def separate_sources(
    master_wav_path: str,
    out_dir: str = STEM_DIR,
    job_id: str = "",
    shifts: int = 1,
) -> dict[str, str]:
    """Separate a mixed WAV into real per-source stems.

    Args:
        master_wav_path: path to the generated master audio (WAV).
        out_dir: directory to write stems into.
        job_id: used to name the output files.
        shifts: Demucs shift count (1 = a little faster, still real).

    Returns:
        Mapping of source name -> "/audio/stems/<job>_<source>.wav".
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model

    model = _load_model()
    os.makedirs(out_dir, exist_ok=True)

    wav, sr = torchaudio.load(master_wav_path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    device = _torch_device()
    sources = apply_model(model, wav[None], device=device, shifts=shifts)
    # sources: [batch=1, n_sources, channels, time]
    stems = sources[0]
    names = list(model.sources)  # canonical order (drums, bass, other, vocals)

    result: dict[str, str] = {}
    for i, name in enumerate(names):
        stem_path = f"{out_dir}/{job_id}_{name}.wav"
        torchaudio.save(stem_path, stems[i], model.samplerate)
        result[name] = f"/audio/stems/{os.path.basename(out_dir).replace('stems/','') if False else job_id}_{name}.wav"
    return result
