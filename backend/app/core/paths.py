"""
Centralized Path Resolution for Milimo Music.
Ensures consistent absolute path resolution whether running:
- Locally from repo root
- Locally from backend/ directory (via uvicorn app.main:app)
- Inside Docker container (/app)
"""

import os
from pathlib import Path
from typing import Optional


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
