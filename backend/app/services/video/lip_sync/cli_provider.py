"""CLI-backed avatar providers (plan 8.4).

The tiers above the jaw-warp are external repos with real argparse entry points.
Rather than porting 22B models into process, the studio drives them as subprocesses
with a declared contract, and reports exactly what is installed. That is what makes
the tier ladder *functional*: download the weights, point an env var at them, and
the engine becomes selectable — no code change.

Each spec carries the metadata routing needs: licence, territory blocks, VRAM and
frame rate. A user in a blocked territory is never offered the engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.video.lip_sync.base import BaseLipSyncProvider

logger = logging.getLogger("milimo.video.lipsync.cli")

TEMP_ROOT_ENV = "MILIMO_AVATAR_TMP"


@dataclass(frozen=True)
class AvatarCliSpec:
    """Everything needed to run and to *describe* one external engine."""

    id: str
    name: str
    env_var: str                 # where the user points at the repo/weights
    script: str                  # entry point inside that directory
    licence: str
    licence_class: str           # "permissive" | "conditional" | "non_commercial"
    min_vram_gb: float
    fps: int
    territory_blocked: tuple = ()
    singing: bool = False
    args_template: tuple = ()    # (flag, placeholder) pairs, resolved below
    #: Engines like InfiniteTalk take a JSON request file instead of flags. When
    #: set, this names the env var holding a user-supplied template; we substitute
    #: `{{image}}`, `{{audio}}`, `{{prompt}}` and `{{frames}}` and keep every other
    #: key verbatim, so the engine's own schema stays the source of truth.
    json_template_env: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": "local",
            "licence": self.licence,
            "licence_class": self.licence_class,
            "min_vram_gb": self.min_vram_gb,
            "fps": self.fps,
            "territory_blocked": list(self.territory_blocked),
            "singing": self.singing,
            "env_var": self.env_var,
            "script": self.script,
            "needs_json_template": self.json_template_env,
            "neural": True,
            "notes": self.notes,
        }


#: Engines with a clean image+audio entry point.
#: Deliberately only what can be driven non-interactively — InfiniteTalk and
#: LongCat take JSON/torchrun launchers and are left to the plan rather than
#: half-wired.
AVATAR_CLI_SPECS: Dict[str, AvatarCliSpec] = {
    "echomimic_flash": AvatarCliSpec(
        id="echomimic_flash",
        name="EchoMimicV3 Flash (Apache-2.0, 12 GB, 25 fps)",
        env_var="MILIMO_ECHOMIMIC_DIR",
        script="infer_flash.py",
        licence="Apache-2.0 (code and weights)",
        licence_class="permissive",
        min_vram_gb=12.0,
        fps=25,
        singing=False,
        args_template=(
            ("--image_path", "image"),
            ("--audio_path", "audio"),
            ("--video_length", "frames"),
            ("--fps", "fps"),
            ("--sample_size", "sample_size"),
            ("--save_path", "output"),
        ),
        notes=(
            "Speech-trained but the friendliest local tier: 8 steps, no face mask, "
            "expects 16 kHz audio normalised to about -23 LUFS."
        ),
    ),
    "wan_s2v": AvatarCliSpec(
        id="wan_s2v",
        name="Wan 2.2 S2V-14B (Apache-2.0, 80 GB, 16 fps, singing-trained)",
        env_var="MILIMO_WAN_S2V_DIR",
        script="generate.py",
        licence="Apache-2.0",
        licence_class="permissive",
        min_vram_gb=80.0,
        fps=16,
        singing=True,
        args_template=(
            ("--task", "constant:s2v-14B"),
            ("--size", "size"),
            ("--ckpt_dir", "ckpt_dir"),
            ("--prompt", "prompt"),
            ("--image", "image"),
            ("--audio", "audio"),
            ("--offload_model", "constant:True"),
            ("--convert_model_dtype", "flag"),
        ),
        notes=(
            "The only tier with explicit singing training. Needs the official repo "
            "plus its checkpoints; 80 GB is the documented floor."
        ),
    ),
    "infinitetalk": AvatarCliSpec(
        id="infinitetalk",
        name="InfiniteTalk (Apache-2.0, streaming, image or video)",
        env_var="MILIMO_INFINITETALK_DIR",
        script="generate_infinitetalk.py",
        licence="Apache-2.0",
        licence_class="permissive",
        min_vram_gb=24.0,
        fps=25,
        singing=True,
        json_template_env="MILIMO_INFINITETALK_TEMPLATE",
        args_template=(
            ("--input_json", "json"),
            ("--size", "constant:infinitetalk-480"),
            ("--mode", "constant:streaming"),
            ("--num_persistent_param_in_dit", "constant:0"),
            ("--sample_steps", "constant:40"),
        ),
        notes=(
            "Streaming, unlimited length, ships a singing example. Its CLI is driven "
            "by a request JSON, so point MILIMO_INFINITETALK_TEMPLATE at a copy of the "
            "repo's example with the paths left as placeholders."
        ),
    ),
}


def _resolve_root(spec: AvatarCliSpec) -> Optional[str]:
    """The directory the user pointed the env var at, when it holds the script."""
    root = os.environ.get(spec.env_var)
    if not root:
        return None
    expanded = os.path.expanduser(root)
    if not os.path.isdir(expanded):
        return None
    return expanded


def _script_path(spec: AvatarCliSpec) -> Optional[str]:
    root = _resolve_root(spec)
    if not root:
        return None
    candidate = os.path.join(root, spec.script)
    return candidate if os.path.isfile(candidate) else None


def spec_available(spec: AvatarCliSpec) -> bool:
    """Available only when *everything* the engine needs is present.

    Delegates to `spec_reason` so availability and the explanation can never
    disagree — the class of bug that made LivePortrait claim a neural engine while
    rendering a jaw-warp.
    """
    return spec_reason(spec) is None


def _template_path(spec: AvatarCliSpec) -> Optional[str]:
    if not spec.json_template_env:
        return None
    raw = os.environ.get(spec.json_template_env)
    if not raw:
        return None
    path = os.path.expanduser(raw)
    return path if os.path.isfile(path) else None


def spec_reason(spec: AvatarCliSpec) -> Optional[str]:
    """Why an engine cannot run here, phrased as the fix."""
    root = _resolve_root(spec)
    if root is None:
        return f"Set {spec.env_var} to the cloned {spec.name.split(' (')[0]} directory."
    if _script_path(spec) is None:
        return f"{spec.env_var} is set but {spec.script} is not inside it."
    if spec.json_template_env and _template_path(spec) is None:
        return (
            f"Set {spec.json_template_env} to a request-JSON template for "
            f"{spec.name.split(' (')[0]}."
        )
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is not on PATH; audio slicing and clip assembly need it."
    return None


def render_json_request(
    template_path: str,
    *,
    image_path: str,
    audio_path: str,
    prompt: str,
    duration: float,
    fps: int,
    out_path: str,
) -> Optional[str]:
    """Materialise an engine's request JSON from a template.

    Unknown keys are preserved verbatim: the engine's schema wins, we only fill the
    three placeholders. Returns the JSON text, or None when the template is invalid.
    """
    import json as _json

    try:
        with open(template_path, encoding="utf-8") as fh:
            template = _json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read avatar request template %s: %s", template_path, exc)
        return None

    frames = max(1, int(round(duration * fps)))
    replacements = {
        "{{image}}": image_path,
        "{{audio}}": audio_path,
        "{{prompt}}": prompt or "a person is singing",
        "{{frames}}": frames,
        "{{output}}": out_path,
    }

    def _substitute(value: Any) -> Any:
        if isinstance(value, str):
            for token, replacement in replacements.items():
                value = value.replace(token, str(replacement))
            # A whole-value frame placeholder should stay numeric.
            if value.strip().isdigit():
                return int(value)
            return value
        if isinstance(value, list):
            return [_substitute(v) for v in value]
        if isinstance(value, dict):
            return {k: _substitute(v) for k, v in value.items()}
        return value

    try:
        return _json.dumps(_substitute(template))
    except (TypeError, ValueError) as exc:
        logger.warning("Could not serialise avatar request template: %s", exc)
        return None


def build_argv(
    spec: AvatarCliSpec,
    *,
    script_path: str,
    image_path: str,
    audio_path: str,
    out_path: str,
    prompt: str,
    duration: float,
    width: int,
    height: int,
    python: Optional[str] = None,
) -> List[str]:
    """Resolve the spec's argument template into a concrete command.

    Kept pure so the contract is testable without any weights.
    """
    frames = max(1, int(round(duration * spec.fps)))
    values = {
        "json": os.environ.get("MILIMO_AVATAR_REQUEST_JSON", ""),
        "image": image_path,
        "audio": audio_path,
        "output": out_path,
        "prompt": prompt or "a person is singing",
        "frames": str(frames),
        "fps": str(spec.fps),
        "sample_size": f"{int(width)} {int(height)}",
        "size": f"{int(width)}*{int(height)}",
        "ckpt_dir": os.path.dirname(script_path),
    }
    argv: List[str] = [python or "python", script_path]
    for flag, token in spec.args_template:
        if token.startswith("constant:"):
            value = token.split(":", 1)[1]
            if value == "True":
                argv += [flag]
                # Wan's CLI takes `--offload_model True` as a value flag.
                argv[-1] = flag
                argv.append("True")
            else:
                argv += [flag, value]
            continue
        if token == "flag":
            argv.append(flag)
            continue
        resolved = values.get(token)
        if resolved is None:
            continue
        argv += [flag, resolved]
    return argv


class CliAvatarProvider(BaseLipSyncProvider):
    """Runs an external avatar engine as a subprocess."""

    def __init__(self, spec: AvatarCliSpec, *, timeout_s: int = 7200):
        self.spec = spec
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self.spec.id

    @property
    def is_available(self) -> bool:
        return spec_reason(self.spec) is None

    async def render_lip_sync(
        self,
        face_image_path: str,
        vocal_audio_path: str,
        start_time: float,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        **kwargs
    ) -> bool:
        script = _script_path(self.spec)
        if not script:
            self._set_outcome("jaw_warp", True, spec_reason(self.spec))
            return False

        cancel_event = kwargs.get("cancel_event")
        cancel_check = kwargs.get("cancel_check")
        if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
            raise asyncio.CancelledError("Lip sync cancelled by user.")

        tmp_dir = os.environ.get(TEMP_ROOT_ENV) or os.path.dirname(out_path) or "/tmp"
        slice_audio = os.path.join(tmp_dir, f"slice_{uuid.uuid4().hex[:8]}.wav")
        cut = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{float(start_time):.3f}", "-t", f"{float(duration):.3f}",
            "-i", vocal_audio_path, "-ar", "16000", "-ac", "1", slice_audio,
        ]
        proc_cut = await asyncio.create_subprocess_exec(
            *cut, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc_cut.communicate()
        if not os.path.isfile(slice_audio) or os.path.getsize(slice_audio) == 0:
            self._set_outcome("jaw_warp", True, f"{self.spec.id}: could not slice source audio")
            return False

        request_json = None
        if self.spec.json_template_env:
            template = _template_path(self.spec)
            payload = render_json_request(
                template, image_path=face_image_path, audio_path=slice_audio,
                prompt=str(kwargs.get("prompt") or ""), duration=duration,
                fps=self.spec.fps, out_path=out_path,
            ) if template else None
            if not payload:
                self._set_outcome("jaw_warp", True, spec_reason(self.spec) or "request template invalid")
                return False
            request_json = os.path.join(tmp_dir, f"request_{uuid.uuid4().hex[:8]}.json")
            with open(request_json, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.environ["MILIMO_AVATAR_REQUEST_JSON"] = request_json

        argv = build_argv(
            self.spec, script_path=script, image_path=face_image_path,
            audio_path=slice_audio, out_path=out_path,
            prompt=str(kwargs.get("prompt") or ""),
            duration=duration, width=width, height=height,
        )
        logger.info("%s: running %s", self.spec.id, " ".join(argv[:4]) + " …")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=os.path.dirname(script),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                self._set_outcome("jaw_warp", True, f"{self.spec.id} timed out after {self.timeout_s}s")
                return False
        except asyncio.CancelledError:
            proc.kill()
            raise
        except Exception as exc:
            self._set_outcome("jaw_warp", True, f"{self.spec.id} could not start: {exc}")
            return False
        finally:
            if request_json and os.path.isfile(request_json):
                try:
                    os.remove(request_json)
                except OSError:
                    pass
            os.environ.pop("MILIMO_AVATAR_REQUEST_JSON", None)
            if os.path.isfile(slice_audio):
                try:
                    os.remove(slice_audio)
                except OSError:
                    pass

        produced = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        if proc.returncode != 0 or not produced:
            tail = (stderr or b"")[-300:].decode("utf-8", errors="ignore")
            self._set_outcome(
                "jaw_warp", True,
                f"{self.spec.id} exited {proc.returncode}"
                + (f": {tail}" if tail else ""),
            )
            return False

        self._set_outcome(self.spec.id, False, None)
        return True


def cli_providers() -> List[CliAvatarProvider]:
    return [CliAvatarProvider(spec) for spec in AVATAR_CLI_SPECS.values()]


def provider_for(engine_id: str) -> Optional[CliAvatarProvider]:
    spec = AVATAR_CLI_SPECS.get((engine_id or "").strip().lower())
    return CliAvatarProvider(spec) if spec else None
