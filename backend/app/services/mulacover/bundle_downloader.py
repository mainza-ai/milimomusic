"""
Automated Composite Bundle Downloader for MuLaCover.
Handles sequential, resumable acquisition of the 4 required checkpoints:
1. MuLaCover-3B Generator (safetensors shards, config, tokenizer) from HeartMuLa/MuLaCover.
2. Qwen3-Embedding-0.6B text prompt encoder from Qwen/Qwen3-Embedding-0.6B.
3. SymbolicTranscriptor (YourMT3 from mimbres/YourMT3 Space + 5 ChordNet models from GitHub).
4. Audio Codec (HeartCodec-oss via symlink to heartlib or download from HeartMuLa/HeartCodec-oss-20260123).
"""

import os
import glob
import logging
import threading
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

MULACOVER_REPO = "HeartMuLa/MuLaCover"
QWEN_EMBED_REPO = "Qwen/Qwen3-Embedding-0.6B"
CODEC_REPO = "HeartMuLa/HeartCodec-oss-20260123"
YOURMT3_SPACE = "mimbres/YourMT3"
YOURMT3_SPACE_FILE = "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"

CHORD_BASE_URL = "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data"
CHORD_FILES = [
    f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict"
    for fold in range(5)
]


def resolve_mulacover_dir() -> Path:
    """Return canonical models/audio/HeartMuLa__MuLaCover directory."""
    from app.core.paths import get_models_dir
    return get_models_dir("audio") / "HeartMuLa__MuLaCover"


def is_mulacover_installed(base_dir: Optional[Path] = None) -> bool:
    """Verify that all core components of the MuLaCover bundle are present on disk."""
    root = base_dir or resolve_mulacover_dir()
    if not root.is_dir():
        return False

    # Check MuLaCover generator shards & config
    gen_dir = root / "MuLaCover"
    if not (gen_dir.is_dir() and (gen_dir / "config.json").is_file()):
        return False
    safetensors = list(gen_dir.glob("*.safetensors"))
    if len(safetensors) < 5:
        return False

    # Check Qwen text embedding model
    qwen_dir = root / "Qwen3-Embedding-0.6B"
    if not (qwen_dir.is_dir() and (qwen_dir / "config.json").is_file()):
        return False

    # Check HeartCodec
    codec_dir = root / "HeartCodec-oss"
    if not (codec_dir.is_dir() or codec_dir.is_symlink()):
        return False

    # Check SymbolicTranscriptor
    st_dir = root / "SymbolicTranscriptor"
    if not (st_dir.is_dir() and (st_dir / "yourmt3" / "last.ckpt").is_file()):
        return False
    ch_files = list((st_dir / "chord").glob("*.best.sdict")) if (st_dir / "chord").is_dir() else []
    if len(ch_files) < 5:
        return False

    return True


def get_bundle_manifest(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect and resolve all files needed across the composite bundle."""
    root = base_dir or resolve_mulacover_dir()
    api = HfApi()

    manifest: List[Dict[str, Any]] = []

    # 1. MuLaCover files
    try:
        mc_info = api.model_info(MULACOVER_REPO, files_metadata=True)
        for s in (mc_info.siblings or []):
            if s.size:
                manifest.append({
                    "kind": "hf_model",
                    "repo_id": MULACOVER_REPO,
                    "repo_type": "model",
                    "rfilename": s.rfilename,
                    "target_dir": str(root / "MuLaCover"),
                    "target_file": str(root / "MuLaCover" / s.rfilename),
                    "size": int(s.size),
                    "label": f"MuLaCover: {s.rfilename}",
                })
    except Exception as e:
        logger.error(f"Failed to fetch {MULACOVER_REPO} manifest: {e}")

    # 2. Qwen3-Embedding-0.6B files
    try:
        qw_info = api.model_info(QWEN_EMBED_REPO, files_metadata=True)
        for s in (qw_info.siblings or []):
            if s.size:
                manifest.append({
                    "kind": "hf_model",
                    "repo_id": QWEN_EMBED_REPO,
                    "repo_type": "model",
                    "rfilename": s.rfilename,
                    "target_dir": str(root / "Qwen3-Embedding-0.6B"),
                    "target_file": str(root / "Qwen3-Embedding-0.6B" / s.rfilename),
                    "size": int(s.size),
                    "label": f"Qwen3-Embed: {s.rfilename}",
                })
    except Exception as e:
        logger.error(f"Failed to fetch {QWEN_EMBED_REPO} manifest: {e}")

    # 3. YourMT3 checkpoint (from Space)
    manifest.append({
        "kind": "hf_space",
        "repo_id": YOURMT3_SPACE,
        "repo_type": "space",
        "rfilename": YOURMT3_SPACE_FILE,
        "target_dir": str(root / "SymbolicTranscriptor" / "yourmt3"),
        "target_file": str(root / "SymbolicTranscriptor" / "yourmt3" / "last.ckpt"),
        "size": 536_000_000,
        "label": "YourMT3: last.ckpt",
    })

    # 4. 5 ChordNet models (from GitHub)
    for c_file in CHORD_FILES:
        manifest.append({
            "kind": "http",
            "url": f"{CHORD_BASE_URL}/{c_file}",
            "target_dir": str(root / "SymbolicTranscriptor" / "chord"),
            "target_file": str(root / "SymbolicTranscriptor" / "chord" / c_file),
            "size": 500_000,
            "label": f"ChordNet: {c_file.split('_s')[-1]}",
        })

    # 5. HeartCodec-oss (check local first)
    from app.core.paths import get_heartmula_ckpt_dir, get_repo_root
    local_codec_candidates = [
        get_heartmula_ckpt_dir() / "HeartCodec-oss",
        get_repo_root() / "heartlib" / "ckpt" / "HeartCodec-oss",
    ]
    found_local_codec = None
    for cand in local_codec_candidates:
        if cand.is_dir():
            found_local_codec = cand
            break

    if not found_local_codec:
        try:
            codec_info = api.model_info(CODEC_REPO, files_metadata=True)
            for s in (codec_info.siblings or []):
                if s.size:
                    manifest.append({
                        "kind": "hf_model",
                        "repo_id": CODEC_REPO,
                        "repo_type": "model",
                        "rfilename": s.rfilename,
                        "target_dir": str(root / "HeartCodec-oss"),
                        "target_file": str(root / "HeartCodec-oss" / s.rfilename),
                        "size": int(s.size),
                        "label": f"HeartCodec: {s.rfilename}",
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch {CODEC_REPO} manifest: {e}")

    total_bytes = sum(m["size"] for m in manifest)
    return {
        "manifest": manifest,
        "total_files": len(manifest),
        "total_bytes": total_bytes,
        "found_local_codec": str(found_local_codec) if found_local_codec else None
    }


def download_mulacover_bundle(
    download_id: str,
    rec: Dict[str, Any],
    cancel_event: threading.Event,
    base_dir: Optional[Path] = None
):
    """Worker function to execute the full composite download sequence."""
    root = base_dir or resolve_mulacover_dir()
    root.mkdir(parents=True, exist_ok=True)

    bundle_info = get_bundle_manifest(root)
    manifest: List[Dict[str, Any]] = bundle_info["manifest"]
    total_bytes = bundle_info["total_bytes"]
    found_local_codec = bundle_info["found_local_codec"]

    # Symlink HeartCodec if available locally
    target_codec = root / "HeartCodec-oss"
    if found_local_codec and not target_codec.exists():
        try:
            target_codec.symlink_to(Path(found_local_codec).resolve())
            logger.info(f"Symlinked existing HeartCodec from {found_local_codec}")
        except Exception as e:
            logger.warning(f"Symlink creation failed: {e}")

    rec["total_files"] = len(manifest)
    rec["total_bytes"] = total_bytes
    rec["status"] = "downloading"

    # Pre-count already completed files for instant resume
    for item in manifest:
        t_file = item["target_file"]
        if os.path.isfile(t_file) and os.path.getsize(t_file) > 0:
            rec["files_done"] += 1
            rec["received_bytes"] += item["size"]

    try:
        for item in manifest:
            if cancel_event.is_set():
                rec["status"] = "cancelled"
                return

            t_file = item["target_file"]
            t_dir = item["target_dir"]
            os.makedirs(t_dir, exist_ok=True)

            # Skip if already exists and non-empty
            if os.path.isfile(t_file) and os.path.getsize(t_file) > 0:
                continue

            rec["current_file"] = item["label"]

            # Download according to source kind
            kind = item["kind"]
            if kind in ("hf_model", "hf_space"):
                last_err = None
                for attempt in range(3):
                    try:
                        hf_hub_download(
                            repo_id=item["repo_id"],
                            filename=item["rfilename"],
                            repo_type=item.get("repo_type", "model"),
                            local_dir=t_dir
                        )
                        # If file downloaded with full subpath, move to expected target if different
                        subpath_file = os.path.join(t_dir, item["rfilename"])
                        if subpath_file != t_file and os.path.isfile(subpath_file):
                            os.makedirs(os.path.dirname(t_file), exist_ok=True)
                            os.replace(subpath_file, t_file)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning(f"Download {item['label']} attempt {attempt+1}/3 failed: {e}")
                        import time
                        time.sleep(2 ** attempt)
                if last_err is not None:
                    raise last_err

            elif kind == "http":
                # Streamed direct HTTP download
                resp = requests.get(item["url"], stream=True, timeout=30)
                resp.raise_for_status()
                with open(t_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if cancel_event.is_set():
                            rec["status"] = "cancelled"
                            return
                        if chunk:
                            f.write(chunk)

            rec["files_done"] += 1
            rec["received_bytes"] += item["size"]

        # Clean stray lock files
        for lock in glob.glob(str(root / "**" / "*.lock"), recursive=True):
            try:
                os.remove(lock)
            except OSError:
                pass

        rec["status"] = "completed"
        rec["current_file"] = ""

        # Auto-register in model manager
        try:
            from app.services.model_manager import model_manager
            model_manager.register_custom_model(
                MULACOVER_REPO,
                metadata={"local_path": str(root), "category": "audio", "name": "MuLaCover-3B"}
            )
        except Exception as e:
            logger.warning(f"Could not auto-register MuLaCover: {e}")

    except Exception as e:
        rec["status"] = "error"
        rec["error"] = str(e)[:500]
        logger.error(f"MuLaCover bundle download {download_id} failed: {e}")
