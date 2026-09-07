#!/usr/bin/env python3
"""
Migration script to reorganize model directories:
- Standardizes models into repo-root `models/{audio,image,video,audio_separator}`.
- Moves non-HeartMuLa models (FLUX.2, test models) out of `heartlib/ckpt/` into `models/`.
- Updates `custom_models.json` with updated local paths.
- Consolidates `audio_separator` models into `models/audio_separator`.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add backend to path to import paths helper
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.paths import (
    get_repo_root,
    get_models_dir,
    get_data_dir,
    get_heartmula_ckpt_dir,
)


def migrate():
    root = get_repo_root()
    print(f"==> Milimo Music Model Directory Migration")
    print(f"    Repo root: {root}")

    models_dir = get_models_dir()
    audio_dir = get_models_dir("audio")
    image_dir = get_models_dir("image")
    video_dir = get_models_dir("video")
    sep_dir = get_models_dir("audio_separator")

    print(f"    Target models dir: {models_dir}")

    # 1. Relocate models from heartlib/ckpt
    heartlib_ckpt = root / "heartlib" / "ckpt"
    path_replacements = {}

    if heartlib_ckpt.exists():
        for item in heartlib_ckpt.iterdir():
            # Skip heartlib specific legacy files
            if item.name in ("HeartMuLa-oss-3B", "HeartCodec-oss", "tokenizer.json", "gen_config.json", ".gitattributes", ".DS_Store", "README.md", ".cache"):
                continue

            # Check if FLUX or image model
            if "FLUX" in item.name or "flux" in item.name.lower():
                target = image_dir / item.name
            else:
                target = models_dir / item.name

            print(f"    Moving {item} -> {target}...")
            if target.exists():
                print(f"    Target {target} already exists, skipping move.")
            else:
                shutil.move(str(item), str(target))

            path_replacements[str(item.resolve())] = str(target.resolve())
            path_replacements[str(item)] = str(target)

    # 2. Consolidate backend/models/audio_separator if it exists
    backend_sep = root / "backend" / "models" / "audio_separator"
    if backend_sep.exists() and backend_sep != sep_dir:
        for f in backend_sep.iterdir():
            dest = sep_dir / f.name
            if not dest.exists():
                print(f"    Consolidating {f.name} into {sep_dir}...")
                shutil.copy2(str(f), str(dest))
        # Remove redundant backend/models directory
        try:
            shutil.rmtree(str(root / "backend" / "models"))
            print(f"    Cleaned up redundant {root / 'backend' / 'models'}")
        except Exception as e:
            print(f"    Warning removing backend/models: {e}")

    # 3. Update custom_models.json
    custom_model_files = [
        root / "backend" / "data" / "models" / "custom_models.json",
        root / "data" / "models" / "custom_models.json",
    ]

    for cm_path in custom_model_files:
        if cm_path.exists():
            try:
                with open(cm_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                updated = False
                for entry in data:
                    old_path = entry.get("local_path")
                    if old_path:
                        for old, new in path_replacements.items():
                            if old in old_path:
                                entry["local_path"] = old_path.replace(old, new)
                                updated = True
                                print(f"    Updated {entry.get('id')} path in {cm_path.name} -> {entry['local_path']}")
                if updated:
                    with open(cm_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"    Saved updated {cm_path}")
            except Exception as e:
                print(f"    Error updating {cm_path}: {e}")

    print("==> Migration complete!")


if __name__ == "__main__":
    migrate()
