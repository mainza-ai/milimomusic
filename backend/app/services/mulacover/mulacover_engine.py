"""
Production-Hardened MuLaCover Generation Engine.
Features:
- Cross-platform hardware auto-negotiation (CUDA / Apple Silicon MPS / CPU).
- Fine-grained cancellation check on every 80ms autoregressive step.
- Live SSE progress reporting to client subscribers.
- Thread-safe memory lifecycle management and clean KV cache resetting.
"""

import os
import gc
import math
import logging
import asyncio
from contextlib import nullcontext
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, Tuple, TYPE_CHECKING

_repo_root = Path(__file__).resolve().parents[4]
_mulacover_src = _repo_root / "mulacover" / "src"
if _mulacover_src.exists() and str(_mulacover_src) not in sys.path:
    sys.path.insert(0, str(_mulacover_src))

import torch
import soundfile as sf
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from mulacover.configuration import MuLaCoverConfig
    from mulacover.modeling import MuLaCover
    from mulacover.symbolic import SymbolicCondition
    from mulacover._codec.modeling import HeartCodec
    from mulacover.pipeline import MuLaCoverGenConfig

from app.services.mulacover.formatters import format_style_tags, sanitize_lyrics_for_mulacover
from app.core.paths import get_models_dir, get_generated_audio_dir

logger = logging.getLogger(__name__)


def resolve_compute_environment():
    """Detect available compute accelerator and optimal precision."""
    if torch.cuda.is_available():
        return torch.device("cuda"), {
            "mulacover": torch.bfloat16,
            "codec": torch.float32,
            "qwen": torch.float32,
            "transcriptor": torch.float32,
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon unified memory
        return torch.device("mps"), {
            "mulacover": torch.float16,
            "codec": torch.float32,
            "qwen": torch.float32,
            "transcriptor": torch.float32,
        }
    else:
        return torch.device("cpu"), {
            "mulacover": torch.float32,
            "codec": torch.float32,
            "qwen": torch.float32,
            "transcriptor": torch.float32,
        }


class MuLaCoverEngine:
    """Production wrapper for MuLaCover model execution."""

    def __init__(
        self,
        model_root: Optional[Union[str, Path]] = None,
        lazy_load: bool = True,
    ):
        if model_root is None:
            model_root = get_models_dir("audio") / "HeartMuLa__MuLaCover"

        self.model_root = Path(model_root)
        self.device, self.dtypes = resolve_compute_environment()
        self.lazy_load = lazy_load

        self._paths = None
        self._mulacover = None
        self._codec = None
        self._qwen = None
        self._qwen_tokenizer = None
        self._text_tokenizer = None
        self._gen_config = None
        self._model_config = None
        self._cache_batch_size = None
        self._lock = asyncio.Lock()

    def _init_metadata(self):
        if self._paths is None:
            from mulacover.pipeline import MuLaCoverGenConfig, _resolve_paths
            from mulacover.configuration import MuLaCoverConfig
            self._paths = _resolve_paths(str(self.model_root))
            self._text_tokenizer = Tokenizer.from_file(str(self._paths["tokenizer"]))
            self._text_tokenizer.no_truncation()
            self._gen_config = MuLaCoverGenConfig.from_file(self._paths["gen_config"])
            self._model_config = MuLaCoverConfig.from_pretrained(
                self._paths["mulacover"], local_files_only=True
            )

    @property
    def mulacover(self) -> Any:
        self._init_metadata()
        if self._mulacover is None:
            from mulacover.modeling import MuLaCover
            logger.info(f"Loading MuLaCover backbone onto {self.device} ({self.dtypes['mulacover']})...")
            model, loading_info = MuLaCover.from_pretrained(
                self._paths["mulacover"],
                device_map=self.device,
                dtype=self.dtypes["mulacover"],
                local_files_only=True,
                output_loading_info=True,
            )
            self._mulacover = model.eval()
        return self._mulacover

    @property
    def codec(self) -> Any:
        self._init_metadata()
        if self._codec is None:
            from mulacover._codec.modeling import HeartCodec
            logger.info(f"Loading HeartCodec onto {self.device}...")
            self._codec = HeartCodec.from_pretrained(
                self._paths["codec"],
                device_map=self.device,
                dtype=self.dtypes["codec"],
                local_files_only=True,
            ).eval()
        return self._codec

    def _load_qwen(self):
        self._init_metadata()
        if self._qwen is None:
            from transformers import AutoModel, AutoTokenizer
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                self._paths["qwen"], padding_side="left", local_files_only=True
            )
            self._qwen = (
                AutoModel.from_pretrained(
                    self._paths["qwen"],
                    dtype=self.dtypes["qwen"],
                    local_files_only=True,
                )
                .to(self.device)
                .eval()
            )

    def _release_component(self, name: str):
        if not self.lazy_load:
            return
        setattr(self, f"_{name}", None)
        if name == "qwen":
            self._qwen_tokenizer = None
        if name == "mulacover":
            self._cache_batch_size = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    @torch.inference_mode()
    def encode_style(self, tags: str) -> torch.Tensor:
        self._init_metadata()
        if not tags:
            return torch.zeros(self._model_config.qwen_dim)
        try:
            self._load_qwen()
            batch = self._qwen_tokenizer(
                [tags], padding=True, truncation=False, return_tensors="pt"
            )
            if batch["input_ids"].shape[1] > 8192:
                raise ValueError("Style exceeds maximum Qwen token context of 8192")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            hidden = self._qwen(**batch).last_hidden_state
            embedding = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
            return embedding[0].cpu()
        finally:
            self._release_component("qwen")

    def _build_prompt_tokens(self, lyrics: str, tags: str):
        c = self._gen_config
        text_ids = lambda text: self._text_tokenizer.encode(text, add_special_tokens=False).ids
        lyric_ids = [c.text_bos_id] + text_ids(lyrics) + [c.text_eos_id]
        tag_ids = [c.tag_start_id] + text_ids(tags) + [c.tag_end_id]
        qwen_index, muq_index = len(tag_ids) + 1, len(tag_ids) + 4
        ids = (
            tag_ids
            + [c.qwen_start_id, 0, c.qwen_end_id, c.muq_start_id, 0, c.muq_end_id]
            + lyric_ids
        )
        tokens = torch.full(
            (len(ids), self._model_config.audio_num_codebooks + 1),
            c.empty_id,
            dtype=torch.long,
        )
        tokens[:, -1] = torch.tensor(ids, dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[:, -1] = True
        return tokens, mask, qwen_index, muq_index

    async def generate_cover(
        self,
        condition: Any,
        lyrics: str,
        tags: str,
        output_path: Union[str, Path],
        duration_ms: int = 120_000,
        temperature: float = 1.0,
        cfg_scale: float = 1.5,
        topk: int = 250,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute full production cover synthesis with cancellation and progress reporting."""
        async with self._lock:
            self._init_metadata()
            clean_lyrics = sanitize_lyrics_for_mulacover(lyrics).lower()
            clean_tags = format_style_tags(tags=tags).lower()

            tokens, mask, qwen_idx, muq_idx = self._build_prompt_tokens(clean_lyrics, clean_tags)
            style_emb = self.encode_style(clean_tags)
            symbolic_tensors = condition.to_tensors()

            batch_size = 2 if cfg_scale > 1.0 else 1
            def batch(tensor):
                return tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim))

            model_inputs = {
                "tokens": batch(tokens),
                "tokens_mask": batch(mask),
                "input_pos": batch(torch.arange(len(tokens), dtype=torch.long)),
                "qwen_embedding": batch(style_emb),
                "muq_embedding": batch(torch.zeros(self._model_config.muq_dim)),
                "qwen_indices": torch.full((batch_size,), qwen_idx, dtype=torch.long),
                "muq_indices": torch.full((batch_size,), muq_idx, dtype=torch.long),
                **{k: batch(v) for k, v in symbolic_tensors.items()},
            }

            # Forward generation loop running in executor
            loop = asyncio.get_running_loop()
            frames = await loop.run_in_executor(
                None,
                lambda: self._forward_cancellable(
                    model_inputs=model_inputs,
                    duration_ms=duration_ms,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            )

            # Codec detokenization
            if progress_callback:
                progress_callback(95, 100, "Detokenizing neural audio waveform with HeartCodec...")

            waveform, sample_rate = await loop.run_in_executor(
                None,
                lambda: self._detokenize(frames)
            )

            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_file), waveform.numpy().T, sample_rate)

            actual_duration = float(waveform.shape[-1]) / float(sample_rate)
            return {
                "audio_path": str(out_file),
                "duration_sec": round(actual_duration, 2),
                "sample_rate": sample_rate,
                "metadata": {
                    "provider": "mulacover",
                    "effective_tags": clean_tags,
                    "effective_lyrics": clean_lyrics,
                    "bpm": condition.bpm,
                }
            }

    def _forward_cancellable(
        self,
        model_inputs: Dict[str, torch.Tensor],
        duration_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        progress_callback: Optional[Callable],
        cancel_event: Optional[Any],
    ) -> torch.Tensor:
        dtype = self.dtypes["mulacover"]
        inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        for k in ("pianoroll", "drum_pianoroll", "chord"):
            inputs[k] = inputs[k].to(dtype)

        tokens = inputs.pop("tokens")
        mask = inputs.pop("tokens_mask")
        position = inputs.pop("input_pos")
        batch_size = tokens.shape[0]
        max_steps = duration_ms // 80

        frames = []
        try:
            model = self.mulacover
            if self._cache_batch_size != batch_size:
                self._cache_batch_size = None
                model.setup_caches(batch_size)
                self._cache_batch_size = batch_size
            model.reset_caches()

            autocast = (
                torch.autocast(self.device.type, dtype=dtype)
                if self.device.type == "cuda"
                else nullcontext()
            )

            with autocast:
                for step in range(max_steps):
                    if cancel_event is not None and cancel_event.is_set():
                        logger.warning(f"MuLaCover cancelled at step {step}/{max_steps}")
                        raise asyncio.CancelledError("MuLaCover generation cancelled by user.")

                    if progress_callback and step % 25 == 0:
                        pct = min(90, int(15 + (step / max(1, max_steps)) * 75))
                        progress_callback(step, max_steps, f"Generating audio tokens ({step}/{max_steps})...")

                    sample = model.generate_frame(
                        tokens=tokens,
                        tokens_mask=mask,
                        input_pos=position,
                        temperature=temperature,
                        topk=topk,
                        cfg_scale=cfg_scale,
                        first_step=step == 0,
                        **inputs,
                    )

                    if torch.any(sample[0] >= self._gen_config.audio_eos_id):
                        logger.info(f"MuLaCover reached EOS at step {step}")
                        break

                    frames.append(sample[0].cpu())
                    tokens = torch.full(
                        (batch_size, 1, self._model_config.audio_num_codebooks + 1),
                        self._gen_config.empty_id,
                        dtype=torch.long,
                        device=self.device,
                    )
                    tokens[:, 0, :-1] = sample
                    mask = torch.ones_like(tokens, dtype=torch.bool)
                    mask[..., -1] = False
                    position = position[:, -1:] + 1
                    inputs["qwen_indices"] = None
                    inputs["muq_indices"] = None
                    inputs["muq_embedding"] = None

            if not frames:
                raise RuntimeError("MuLaCover emitted EOS before generating any audio frames")

            return torch.stack(frames, dim=-1)
        finally:
            if self._mulacover is not None and self._cache_batch_size is not None:
                self._mulacover.reset_caches()
            self._release_component("mulacover")

    def _detokenize(self, frames: torch.Tensor) -> Tuple[torch.Tensor, int]:
        try:
            codec = self.codec
            waveform = codec.detokenize(
                frames.to(self.device),
                duration=29.76,
                num_steps=10,
                guidance_scale=1.25,
                disable_progress=True,
            )
            sr = codec.sample_rate
            return waveform.detach().float().cpu(), sr
        finally:
            self._release_component("codec")


mulacover_engine = MuLaCoverEngine()
