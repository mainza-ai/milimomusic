"""
Centralized Path Resolution for Milimo Music.
Ensures consistent absolute path resolution whether running:
- Locally from repo root
- Locally from backend/ directory (via uvicorn app.main:app)
- Inside Docker container (/app)
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import json
from uuid import UUID


def get_repo_root() -> Path:
    """Return the canonical repository root path.
    Detected by checking parents until we find marker files/directories
    (e.g., .git, heartlib, docker-compose.yml, backend).
    """
    if os.environ.get("MILIMO_ROOT"):
        return Path(os.environ["MILIMO_ROOT"]).resolve()

    current = Path(__file__).resolve().parent  # backend/app/core
    # Walk parents upwards to find the top-level repository root
    for parent in list(current.parents):
        if (parent / ".git").exists() or (parent / "docker-compose.yml").exists() or ((parent / "heartlib").exists() and (parent / "backend").exists()):
            return parent

    # Check for Docker container standard root /app
    for parent in [current] + list(current.parents):
        if parent == Path("/app"):
            return parent

    # Fallback: backend/app/core -> app -> backend -> repo_root
    return current.parent.parent.parent


_REPO_ROOT = get_repo_root()


def get_models_dir(category: Optional[str] = None) -> Path:
    """Return the root models directory (default: <REPO_ROOT>/models)
    or a modality-specific subfolder (models/audio, models/image, models/video, models/audio_separator).
    Honours MODELS_DIRECTORY or MODEL_DIRECTORY env vars if set (and not pointing to legacy heartlib/ckpt).
    """
    env_dir = os.environ.get("MODELS_DIRECTORY") or os.environ.get("MODEL_DIRECTORY")
    if env_dir and not env_dir.strip().endswith("heartlib/ckpt") and not env_dir.strip().endswith("heartlib/ckpt/"):
        base = Path(os.path.expanduser(env_dir)).resolve()
    else:
        base = _REPO_ROOT / "models"

    if category:
        sub = base / category
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_data_dir() -> Path:
    """Return runtime data directory (<REPO_ROOT>/data)."""
    env_dir = os.environ.get("DATA_DIRECTORY")
    base = Path(os.path.expanduser(env_dir)).resolve() if env_dir else _REPO_ROOT / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_checkpoints_dir() -> Path:
    """Return fine-tuning checkpoints directory (<REPO_ROOT>/data/checkpoints)."""
    env_dir = os.environ.get("CHECKPOINTS_DIRECTORY")
    base = Path(os.path.expanduser(env_dir)).resolve() if env_dir else get_data_dir() / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_datasets_dir() -> Path:
    """Return datasets directory (<REPO_ROOT>/data/datasets)."""
    env_dir = os.environ.get("DATASETS_DIRECTORY")
    base = Path(os.path.expanduser(env_dir)).resolve() if env_dir else get_data_dir() / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_heartmula_ckpt_dir() -> Path:
    """Return legacy HeartMuLa checkpoint directory.
    Checks models/heartmula first, then heartlib/ckpt.
    """
    env_dir = os.environ.get("HEARTMULA_MODEL_PATH")
    if env_dir:
        return Path(os.path.expanduser(env_dir)).resolve()

    models_heartmula = get_models_dir() / "heartmula"
    if models_heartmula.exists() and (models_heartmula / "HeartMuLa-oss-3B").exists():
        return models_heartmula

    return (_REPO_ROOT / "heartlib" / "ckpt").resolve()


def get_generated_audio_dir() -> Path:
    """Return generated audio output directory (<REPO_ROOT>/generated_audio)."""
    env_dir = os.environ.get("GENERATED_AUDIO_DIRECTORY")
    base = Path(os.path.expanduser(env_dir)).resolve() if env_dir else _REPO_ROOT / "generated_audio"
    base.mkdir(parents=True, exist_ok=True)
    return base


def resolve_audio_file(path: Optional[str]) -> Optional[str]:
    """Resolve any audio file candidate path or URL to an existing local file on disk.
    
    Robust against:
    - Web static URLs (/audio/..., http://localhost:8000/audio/...)
    - Relative paths evaluated when CWD is <REPO_ROOT>/backend or <REPO_ROOT>
    - Missing or alternate common audio extensions (.wav, .mp3, .flac, .ogg, .m4a)
    """
    if not path or not isinstance(path, str):
        return None

    cleaned_path = path.strip()
    if not cleaned_path:
        return None

    if "://" in cleaned_path:
        from urllib.parse import urlparse
        cleaned_path = urlparse(cleaned_path).path

    # If it's already an absolute path that exists and is non-empty
    if os.path.isabs(cleaned_path) and os.path.isfile(cleaned_path) and os.path.getsize(cleaned_path) > 0:
        return os.path.abspath(cleaned_path)

    basename = os.path.basename(cleaned_path)
    relative_no_slash = cleaned_path.lstrip("/")
    after_audio = cleaned_path.split("/audio/")[-1].lstrip("/") if "/audio/" in cleaned_path else ""
    after_stems = cleaned_path.split("/stems/")[-1].lstrip("/") if "/stems/" in cleaned_path else ""
    after_covers = cleaned_path.split("/covers/")[-1].lstrip("/") if "/covers/" in cleaned_path else ""

    repo_root = get_repo_root()
    gen_dir = get_generated_audio_dir()
    data_dir = get_data_dir()

    search_dirs = [
        gen_dir,
        gen_dir / "stems",
        gen_dir / "mastered",
        gen_dir / "converted_vocals",
        gen_dir / "videos",
        gen_dir / "videos" / "keyframes",
        repo_root / "generated_audio",
        repo_root / "generated_audio" / "stems",
        repo_root / "generated_audio" / "mastered",
        repo_root / "generated_audio" / "converted_vocals",
        repo_root / "generated_audio" / "videos",
        repo_root / "backend" / "generated_audio",
        repo_root / "backend" / "generated_audio" / "stems",
        repo_root / "backend" / "generated_audio" / "mastered",
        repo_root / "backend" / "generated_audio" / "converted_vocals",
        data_dir,
        data_dir / "audio",
        data_dir / "uploads",
        data_dir / "covers",
        data_dir / "video_cache",
        data_dir / "video_cache" / "keyframes",
        repo_root / "data" / "covers",
        repo_root / "data" / "uploads",
        repo_root / "backend" / "data" / "covers",
        Path.cwd(),
        Path.cwd() / "generated_audio",
        Path.cwd() / "generated_audio" / "stems",
        Path.cwd() / "backend" / "generated_audio",
        Path.cwd().parent / "generated_audio",
    ]

    candidates: List[str] = [
        cleaned_path,
        os.path.abspath(cleaned_path),
    ]

    for d in search_dirs:
        candidates.append(str(d / basename))
        if relative_no_slash:
            candidates.append(str(d / relative_no_slash))
        if after_audio:
            candidates.append(str(d / after_audio))
            candidates.append(str(d / "videos" / after_audio))
        if after_stems:
            candidates.append(str(d / after_stems))
            candidates.append(str(d / "stems" / after_stems))
        if after_covers:
            candidates.append(str(d / after_covers))
            candidates.append(str(d / "covers" / after_covers))

    base_name_no_ext, ext = os.path.splitext(basename)
    alt_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    for cand in candidates:
        cand_p = Path(cand)
        if cand_p.is_file() and cand_p.stat().st_size > 0:
            return str(cand_p.resolve())

        if ext.lower() in alt_exts:
            for alt in alt_exts:
                if alt.lower() != ext.lower():
                    alt_cand = cand_p.with_suffix(alt)
                    if alt_cand.is_file() and alt_cand.stat().st_size > 0:
                        return str(alt_cand.resolve())

    return None


def resolve_stem_file(
    job_id: Union[str, UUID],
    stem_name: str,
    stems_json: Optional[Union[str, Dict[str, Any]]] = None
) -> Optional[str]:
    """Find a specific stem file (vocals, drums, bass, other, instrumental) on disk.
    
    1. First checks `stems_json` payload or dictionary.
    2. Then searches on disk across canonical stems directories using both
       UUID string with hyphens and hex format without hyphens.
    """
    if stems_json:
        try:
            data = json.loads(stems_json) if isinstance(stems_json, str) else stems_json
            if isinstance(data, dict):
                stem_val = data.get(stem_name) or data.get(f"{stem_name}_path")
                if stem_val and isinstance(stem_val, str):
                    resolved = resolve_audio_file(stem_val)
                    if resolved:
                        return resolved

                parts = data.get("instrumental_parts")
                if isinstance(parts, dict) and parts.get(stem_name):
                    resolved = resolve_audio_file(parts[stem_name])
                    if resolved:
                        return resolved
        except Exception:
            pass

    id_str = str(job_id)
    id_clean = id_str.replace("-", "")
    gen_dir = get_generated_audio_dir()
    repo_root = get_repo_root()

    search_dirs = [
        gen_dir / "stems",
        gen_dir,
        repo_root / "generated_audio" / "stems",
        repo_root / "generated_audio",
        repo_root / "backend" / "generated_audio" / "stems",
        Path.cwd() / "generated_audio" / "stems",
        Path.cwd() / "stems",
    ]

    exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    name_variations = [stem_name]
    if stem_name == "vocals":
        name_variations.extend(["voice", "vocal"])
    elif stem_name == "instrumental":
        name_variations.extend(["backing", "instruments"])

    for d in search_dirs:
        for var in name_variations:
            for prefix in [id_str, id_clean]:
                for ext in exts:
                    cand = d / f"{prefix}_{var}{ext}"
                    if cand.is_file() and cand.stat().st_size > 0:
                        return str(cand.resolve())

            for ext in exts:
                cand = d / id_str / f"{var}{ext}"
                if cand.is_file() and cand.stat().st_size > 0:
                    return str(cand.resolve())

    return None


def resolve_image_file(path: Optional[str]) -> Optional[str]:
    """Resolve an image file (cover, character avatar, keyframe) on disk."""
    if not path or not isinstance(path, str):
        return None
    cleaned = path.strip()
    if not cleaned:
        return None
    if "://" in cleaned:
        from urllib.parse import urlparse
        cleaned = urlparse(cleaned).path

    if os.path.isabs(cleaned) and os.path.isfile(cleaned) and os.path.getsize(cleaned) > 0:
        return os.path.abspath(cleaned)

    base = os.path.basename(cleaned)
    rel = cleaned.lstrip("/")
    after_covers = cleaned.split("/covers/")[-1].lstrip("/") if "/covers/" in cleaned else ""

    data_dir = get_data_dir()
    gen_dir = get_generated_audio_dir()
    repo_root = get_repo_root()

    search_dirs = [
        data_dir / "covers",
        data_dir / "video_cache" / "keyframes",
        gen_dir / "videos" / "keyframes",
        gen_dir / "videos",
        gen_dir,
        data_dir,
        repo_root / "data" / "covers",
        repo_root / "backend" / "data" / "covers",
        Path.cwd() / "data" / "covers",
        Path.cwd(),
    ]

    candidates = [
        cleaned,
        os.path.abspath(cleaned),
    ]

    for d in search_dirs:
        candidates.append(str(d / base))
        if rel:
            candidates.append(str(d / rel))
        if after_covers:
            candidates.append(str(d / after_covers))
            candidates.append(str(d / "covers" / after_covers))

    alt_exts = [".png", ".jpg", ".jpeg", ".webp"]
    _, ext = os.path.splitext(base)

    for c in candidates:
        cp = Path(c)
        if cp.is_file() and cp.stat().st_size > 0:
            return str(cp.resolve())
        if ext.lower() in alt_exts:
            for alt in alt_exts:
                if alt.lower() != ext.lower():
                    alt_cand = cp.with_suffix(alt)
                    if alt_cand.is_file() and alt_cand.stat().st_size > 0:
                        return str(alt_cand.resolve())

    return None
