import asyncio
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Anchor the working directory to the backend root so all relative artifact paths
# (generated_audio/, data/, llm_config.json, database.db) resolve consistently and the
# /audio static mount + exports work regardless of where uvicorn was launched from.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
import re
from datetime import datetime, timezone
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session, create_engine, select, or_, text
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.services.music_service import music_service, event_manager
from app.services.llm_service import LLMService
from app.services.model_manager import model_manager
from app.services.voice_service import voice_service
from app.transcription.muscriptor_provider import muscriptor_provider
from app.transcription.real_separator import separate_sources
from app.transcription.instrument_stems import render_instrument_parts
from app.transcription.mastering import mastering_engine
from app.transcription.karaoke import lyric_sync_engine
from app.providers.registry import provider_registry
from app.models import (
    Job,
    JobStatus,
    GenerationRequest,
    LyricsRequest,
    LyricsChatRequest,
    EnhancePromptRequest,
    InspirationRequest,
    LLMConfigUpdate,
    ProviderConfig,
    VoiceProfileCreate,
    MasteringRequest,
    Project,
    ProjectCreate,
    ProjectUpdate
)

# Database
sqlite_file_name = "jobs.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.exec(text("PRAGMA journal_mode=WAL;"))
        
        # Automatic Migration: check and add missing columns to job table
        existing_cols = {row[1] for row in session.exec(text("PRAGMA table_info(job);")).all()}
        new_columns = {
            "model_provider": "VARCHAR",
            "llm_model": "VARCHAR",
            "parent_job_id": "VARCHAR",
            "temperature": "FLOAT",
            "cfg_scale": "FLOAT",
            "topk": "INTEGER",
            "midi_path": "VARCHAR",
            "musicxml_path": "VARCHAR",
            "notes_json": "TEXT",
            "stems_json": "TEXT",
            "beat_grid_json": "TEXT",
            "timed_lyrics_json": "TEXT",
            "structured_caption_json": "TEXT",
            "voice_profile_id": "VARCHAR",
            "project_id": "VARCHAR"
        }
        for col, col_type in new_columns.items():
            if col not in existing_cols:
                try:
                    session.exec(text(f"ALTER TABLE job ADD COLUMN {col} {col_type};"))
                except Exception as e:
                    print(f"Migration notice for {col}: {e}")
        session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    await music_service.initialize()
    yield
    event_manager.shutdown()
    music_service.shutdown_all()


app = FastAPI(lifespan=lifespan, title="Milimo Music v2 API — AI Music Production DAW")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from starlette.requests import Request


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc.errors()}")
    return await request_validation_exception_handler(request, exc)


# Static Files (Audio Serving)
os.makedirs("generated_audio", exist_ok=True)
os.makedirs("generated_audio/stems", exist_ok=True)
os.makedirs("generated_audio/mastered", exist_ok=True)
os.makedirs("generated_audio/converted_vocals", exist_ok=True)
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")


# --- Core & Health ---

@app.get("/health")
def health_check():
    active_caps = provider_registry.get_active_capabilities()
    return {
        "status": "ok",
        "active_provider": active_caps.provider_id,
        "display_name": active_caps.display_name,
        "version": active_caps.version
    }


# --- Model Management & Tree Endpoints ---

@app.get("/models/tree")
def get_model_tree():
    """Return model tree with MiniMax Music 3, HeartMuLa, adapters, sizes and install states."""
    return {"models": model_manager.get_model_tree()}


@app.get("/models/capabilities")
def get_model_capabilities():
    """Return capability manifests for all registered generation providers."""
    return {"capabilities": [c.model_dump() for c in provider_registry.list_capabilities()]}


@app.get("/models/hardware")
def get_hardware_profile():
    """Detect hardware configuration and capability recommendations."""
    return {"hardware": model_manager.detect_hardware()}


@app.get("/models/check/{model_id}")
def check_model_dependencies(model_id: str):
    """Check if a selected model variant is downloaded or missing."""
    return model_manager.check_missing_dependencies(model_id)


@app.post("/models/active/{provider_id}")
def set_active_generation_provider(provider_id: str):
    """Switch active default generation model."""
    success = provider_registry.set_active_provider(provider_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")
    return {"active_provider": provider_id, "capabilities": provider_registry.get_active_capabilities().model_dump()}


# --- Voice Training Studio & Profiles API ---

@app.get("/voice/profiles")
def list_voice_profiles():
    """List all custom and pre-trained vocal identities."""
    return {"profiles": voice_service.list_profiles()}


@app.post("/voice/profiles")
def create_voice_profile(req: VoiceProfileCreate):
    """Create a new Voice Identity profile with mandatory consent confirmation."""
    try:
        profile = voice_service.create_profile(
            name=req.name,
            description=req.description,
            consent_confirmed=req.consent_confirmed,
            f0_method=req.f0_method
        )
        return {"profile": profile}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/voice/profiles/{profile_id}")
def delete_voice_profile(profile_id: str):
    """Delete a voice profile."""
    if voice_service.delete_profile(profile_id):
        return {"success": True, "profile_id": profile_id}
    raise HTTPException(status_code=404, detail="Profile not found")


# --- Production DAW Workspace & Export Routes ---

@app.post("/transcribe/upload")
async def transcribe_uploaded_audio(file: UploadFile = File(...)):
    """
    Import user audio file into the DAW, separating stems and generating MIDI/MusicXML.
    """
    import uuid
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    upload_path = os.path.join("generated_audio", filename)

    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    # 1. Real neural source separation (HTDemucs) on the uploaded audio
    loop = asyncio.get_running_loop()
    stems = await loop.run_in_executor(
        None, separate_sources, upload_path, "generated_audio/stems", job_id, 1
    )

    # 2. MuScriptor transcription
    transcription = await muscriptor_provider.transcribe(upload_path, job_id)

    # 2b. MuScriptor-derived per-instrument stems (dynamic instrument parts + GM programs)
    instrument_parts: dict[str, str] = {}
    instrument_programs: dict[str, int] = {}
    try:
        instrument_parts, instrument_programs = render_instrument_parts(
            transcription.notes, job_id, duration_sec=None
        )
    except Exception as e:
        logger.warning(f"Per-instrument stem rendering skipped for import {job_id}: {e}")

    # 3. Create Job session in DB
    job = Job(
        id=UUID(job_id),
        title=f"Imported: {file.filename}",
        prompt="User imported audio",
        audio_path=f"/audio/{filename}",
        status=JobStatus.COMPLETED,
        midi_path=transcription.midi_path,
        musicxml_path=transcription.musicxml_path,
        notes_json=json.dumps(transcription.notes),
        beat_grid_json=json.dumps(transcription.beat_grid),
        stems_json=json.dumps({
            "vocals": stems.get("vocals", ""),
            "drums": stems.get("drums", ""),
            "bass": stems.get("bass", ""),
            "other": stems.get("other", ""),
            "stems_source": "htdemucs",
            "instrumental_parts": instrument_parts,
            "instrument_programs": instrument_programs,
            "sources_available": ["muscriptor", "htdemucs"],
            "default_source": "muscriptor",
        })
    )

    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)

    return {"job_id": job.id, "job": job}


def get_job_by_id(session: Session, job_id_input: Any) -> Optional[Job]:
    """Universal robust job lookup handling UUID objects, hyphenated strings, and raw 32-hex strings."""
    if not job_id_input:
        return None
    
    clean_str = str(job_id_input).strip()
    hex_str = clean_str.replace("-", "")

    # 1. Try UUID object
    try:
        u = UUID(hex_str)
        job = session.get(Job, u)
        if job:
            return job
    except Exception:
        pass

    # 2. Try session.get with string forms
    job = session.get(Job, hex_str)
    if job:
        return job
    job = session.get(Job, clean_str)
    if job:
        return job

    # 3. Try select with hex_str and clean_str
    try:
        job = session.exec(select(Job).where(Job.id == hex_str)).one_or_none()
        if job:
            return job
    except Exception:
        pass

    try:
        job = session.exec(select(Job).where(Job.id == clean_str)).one_or_none()
        if job:
            return job
    except Exception:
        pass

    return None


@app.get("/transcribe/export/{job_id}/{export_format}")
def export_track_asset(job_id: str, export_format: str):
    """
    Export DAW assets: midi, musicxml, ableton, lrc, srt, stems.
    """
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job_id_str = str(job.id)

        if export_format == "midi":
            if not job.midi_path:
                raise HTTPException(status_code=404, detail="MIDI not available for this track")
            file_path = job.midi_path.replace("/audio/", "generated_audio/")
            return FileResponse(file_path, media_type="audio/midi", filename=f"{job.title or 'milimo_track'}.mid")

        elif export_format == "musicxml":
            if not job.musicxml_path:
                raise HTTPException(status_code=404, detail="MusicXML not available")
            file_path = job.musicxml_path.replace("/audio/", "generated_audio/")
            return FileResponse(file_path, media_type="application/vnd.recordare.musicxml+xml", filename=f"{job.title or 'milimo_track'}.musicxml")

        elif export_format == "lrc":
            timed_lines = json.loads(job.timed_lyrics_json) if job.timed_lyrics_json else []
            lrc_content = lyric_sync_engine.generate_lrc(timed_lines)
            return Response(content=lrc_content, media_type="text/plain", headers={
                "Content-Disposition": f"attachment; filename={job.title or 'milimo_lyrics'}.lrc"
            })

        elif export_format == "srt":
            timed_lines = json.loads(job.timed_lyrics_json) if job.timed_lyrics_json else []
            srt_content = lyric_sync_engine.generate_srt(timed_lines)
            return Response(content=srt_content, media_type="text/plain", headers={
                "Content-Disposition": f"attachment; filename={job.title or 'milimo_lyrics'}.srt"
            })

        elif export_format == "ableton":
            # Export DAW Ableton session descriptor
            notes = json.loads(job.notes_json) if job.notes_json else []
            beat_grid = json.loads(job.beat_grid_json) if job.beat_grid_json else {}
            stems = json.loads(job.stems_json) if job.stems_json else {}
            parts = stems.get("instrumental_parts", {})
            instrument_tracks = [
                {"name": inst, "audio": url, "source": "muscriptor",
                 "program": stems.get("instrument_programs", {}).get(inst, 0)}
                for inst, url in parts.items()
            ]
            ableton_desc = {
                "format": "ableton-midi-multitrack",
                "bpm": beat_grid.get("bpm", 120.0),
                "source": stems.get("stems_source", "htdemucs"),
                "tracks": [
                    {"name": "Vocals", "audio": stems.get("vocals"), "source": "htdemucs"},
                    {"name": "Drums", "audio": stems.get("drums"), "source": "htdemucs"},
                    {"name": "Bass", "audio": stems.get("bass"), "source": "htdemucs"},
                    {"name": "Instruments", "audio": stems.get("other"), "source": "htdemucs"}
                ],
                "instrumental_parts": instrument_tracks,
                "note_events": notes
            }
            return Response(content=json.dumps(ableton_desc, indent=2), media_type="application/json", headers={
                "Content-Disposition": f"attachment; filename={job.title or 'milimo_ableton'}.json"
            })

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_format}")


@app.post("/mastering/match/{job_id}")
async def apply_reference_mastering(job_id: UUID, req: MasteringRequest = MasteringRequest()):
    """Apply Matchering reference mastering to track."""
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job or not job.audio_path:
            raise HTTPException(status_code=404, detail="Track not found")

        result = await mastering_engine.match_master(
            target_audio_path=job.audio_path,
            reference_audio_path=None,
            job_id=str(job_id),
            target_lufs=req.target_lufs
        )

        job.audio_path = result.mastered_audio_path
        session.add(job)
        session.commit()
        session.refresh(job)

        return {"status": "mastered", "audio_path": job.audio_path, "lufs": result.target_lufs}


@app.post("/workspace/{job_id}/notes")
def save_workspace_notes(job_id: UUID, notes: List[Dict[str, Any]] = Body(...)):
    """Save edited note events from the Piano Roll editor."""
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job.notes_json = json.dumps(notes)
        session.add(job)
        session.commit()
        return {"status": "saved", "note_count": len(notes)}


# --- LLM Config & Co-Writer Endpoints ---

@app.get("/models/lyrics")
def get_lyrics_models():
    return {"models": LLMService.get_models()}


@app.get("/config/llm")
def get_llm_config():
    return LLMService.get_config()


@app.post("/config/llm")
def update_llm_config(config: LLMConfigUpdate):
    try:
        if config.provider:
            LLMService.set_active_provider(config.provider)
        if config.openai:
            LLMService.update_config("openai", config.openai.model_dump(exclude_unset=True))
        if config.gemini:
            LLMService.update_config("gemini", config.gemini.model_dump(exclude_unset=True))
        if config.openrouter:
            LLMService.update_config("openrouter", config.openrouter.model_dump(exclude_unset=True))
        if config.lmstudio:
            LLMService.update_config("lmstudio", config.lmstudio.model_dump(exclude_unset=True))
        if config.ollama:
            LLMService.update_config("ollama", config.ollama.model_dump(exclude_unset=True))
        if config.deepseek:
            LLMService.update_config("deepseek", config.deepseek.model_dump(exclude_unset=True))
        if config.opencode:
            LLMService.update_config("opencode", config.opencode.model_dump(exclude_unset=True))
        if config.omlx:
            LLMService.update_config("omlx", config.omlx.model_dump(exclude_unset=True))
        return LLMService.get_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/fetch-models")
def fetch_models(request: LLMConfigUpdate):
    try:
        provider = request.provider
        if not provider:
            raise HTTPException(status_code=400, detail="Provider required")
        api_key = None
        base_url = None
        if provider == "openai" and request.openai:
            api_key = request.openai.api_key
        elif provider == "deepseek" and request.deepseek:
            api_key = request.deepseek.api_key
        elif provider == "gemini" and request.gemini:
            api_key = request.gemini.api_key
        elif provider == "openrouter" and request.openrouter:
            api_key = request.openrouter.api_key
        elif provider == "lmstudio" and request.lmstudio:
            base_url = request.lmstudio.base_url
        elif provider == "ollama" and request.ollama:
            base_url = request.ollama.base_url
        elif provider == "opencode" and request.opencode:
            api_key = request.opencode.api_key
            base_url = request.opencode.base_url
        elif provider == "omlx" and request.omlx:
            api_key = request.omlx.api_key
            base_url = request.omlx.base_url

        models = LLMService.fetch_available_models(provider, api_key, base_url)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


@app.post("/generate/enhance_prompt")
def enhance_prompt(req: EnhancePromptRequest):
    try:
        result = LLMService.enhance_prompt(req.concept, req.model_name)
        return result
    except Exception:
        return {"topic": req.concept, "tags": "Pop, Electronic, Modern DAW Master"}


@app.post("/generate/evaluate_inspiration")
def generate_inspiration(req: InspirationRequest):
    try:
        result = LLMService.generate_inspiration(req.model_name)
        return result
    except Exception:
        return {"topic": "A cinematic journey through neon skies", "tags": "Synthwave, Electronic, 128 BPM, Punchy Drums"}


@app.post("/generate/styles")
def generate_styles(req: InspirationRequest):
    try:
        styles = LLMService.generate_styles_list(req.model_name)
        return {"styles": styles}
    except Exception:
        return {"styles": ["Pop", "Rock", "Synthwave", "R&B", "Acoustic", "Cinematic"]}


# --- Style Registry ---
from app.services.style_registry import StyleRegistry, Style
from pydantic import BaseModel


class StyleCreate(BaseModel):
    name: str
    description: Optional[str] = None


class PathsConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_directory: Optional[str] = None
    checkpoints_directory: Optional[str] = None
    datasets_directory: Optional[str] = None


@app.get("/styles")
def get_styles():
    registry = StyleRegistry()
    styles = registry.get_all_styles()
    return {"styles": [s.to_dict() for s in styles]}


@app.post("/styles/custom")
def add_custom_style(style: StyleCreate):
    try:
        registry = StyleRegistry()
        created = registry.add_custom_style(style.name, style.description)
        return {"style": created.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/styles/custom/{name}")
def remove_custom_style(name: str):
    registry = StyleRegistry()
    if registry.remove_custom_style(name):
        return {"status": "deleted", "name": name}
    raise HTTPException(status_code=404, detail=f"Custom style '{name}' not found")


# --- Paths Configuration ---
from app.services.config_manager import ConfigManager


@app.get("/config/paths")
def get_paths_config():
    config = ConfigManager().get_config()
    return config.get("paths", {})


@app.post("/config/paths")
def update_paths_config(paths: PathsConfig):
    updates = paths.model_dump(exclude_unset=True)
    if updates:
        ConfigManager().update_config({"paths": updates})
    return ConfigManager().get_config().get("paths", {})


# --- Fine-Tuning & Training Studio API ---
from app.services.fine_tuning_service import (
    fine_tuning_service, 
    TrainingConfig, 
    Dataset as FTDataset,
    TrainingJob as FTJob
)


class DatasetCreate(BaseModel):
    name: str
    styles: List[str]


class TrainingConfigRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    dataset_id: str
    method: str = "lora"
    epochs: int = 3
    learning_rate: float = 0.0001
    batch_size: int = 2
    lora_rank: int = 8


@app.post("/training/datasets")
def create_dataset(data: DatasetCreate):
    dataset = fine_tuning_service.create_dataset(data.name, data.styles)
    return {"dataset": dataset.to_dict()}


@app.get("/training/datasets")
def list_datasets():
    datasets = fine_tuning_service.list_datasets()
    return {"datasets": [d.to_dict() for d in datasets]}


@app.get("/training/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    dataset = fine_tuning_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"dataset": dataset.to_dict()}


@app.post("/training/datasets/{dataset_id}/audio")
async def upload_audio(dataset_id: str, file: UploadFile = File(...), caption: str = Form(...)):
    try:
        content = await file.read()
        audio_file = fine_tuning_service.add_audio_file(
            dataset_id, file.filename, caption, content
        )
        return {"audio_file": {"filename": audio_file.filename, "caption": audio_file.caption}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/training/datasets/{dataset_id}/audio/{filename}")
def get_dataset_audio(dataset_id: str, filename: str):
    audio_path = fine_tuning_service.datasets_dir / dataset_id / "audio" / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=filename
    )


@app.delete("/training/datasets/{dataset_id}/audio/{filename}")
def delete_audio(dataset_id: str, filename: str):
    success = fine_tuning_service.remove_audio_file(dataset_id, filename)
    if not success:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return {"success": True}


@app.get("/training/datasets/{dataset_id}/validate")
def validate_dataset(dataset_id: str):
    return fine_tuning_service.validate_dataset(dataset_id)


@app.delete("/training/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    success = fine_tuning_service.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True}


@app.post("/training/jobs")
def create_training_job(config: TrainingConfigRequest):
    try:
        training_config = TrainingConfig(
            dataset_id=config.dataset_id,
            method=config.method,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            lora_rank=config.lora_rank
        )
        job = fine_tuning_service.create_training_job(training_config)
        return {"job": job.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/training/jobs")
def list_training_jobs():
    jobs = fine_tuning_service.list_jobs()
    return {"jobs": [j.to_dict() for j in jobs]}


@app.get("/training/jobs/{job_id}")
def get_training_job(job_id: str):
    job = fine_tuning_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job.to_dict()}


@app.post("/training/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str):
    if fine_tuning_service.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    raise HTTPException(status_code=404, detail="Job not found or not running")


@app.get("/training/checkpoints")
def list_checkpoints():
    checkpoints = fine_tuning_service.list_checkpoints()
    return {"checkpoints": [c.to_dict() for c in checkpoints]}


@app.post("/training/checkpoints/{checkpoint_id}/activate")
async def activate_checkpoint(checkpoint_id: str):
    if fine_tuning_service.activate_checkpoint(checkpoint_id):
        music_service.unload_lora()
        await music_service.initialize()
        return {"status": "activated", "checkpoint_id": checkpoint_id}
    raise HTTPException(status_code=404, detail="Checkpoint not found")


@app.post("/training/checkpoints/deactivate")
async def deactivate_checkpoint():
    fine_tuning_service.deactivate_all_checkpoints()
    music_service.unload_lora()
    await music_service.initialize()
    return {"status": "deactivated", "message": "Reverted to base model"}


# --- Lyrics & Co-Writer Endpoints ---

@app.post("/generate/lyrics")
async def generate_lyrics(req: LyricsRequest):
    try:
        from app.services.lyrics_graph import sanitize_lyrics
        lyrics = await LLMService.generate_lyrics_async(req.topic, req.model_name, req.seed_lyrics, req.tags)
        return {"lyrics": sanitize_lyrics(lyrics)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/lyrics-chat")
async def chat_with_lyrics(req: LyricsChatRequest):
    try:
        from app.services.lyrics_graph import sanitize_lyrics
        result = await LLMService.chat_with_lyrics_async(
            req.current_lyrics, 
            req.user_message, 
            req.model_name, 
            req.chat_history, 
            req.topic, 
            req.get_tags_string()
        )
        if result and "lyrics" in result:
            result["lyrics"] = sanitize_lyrics(result["lyrics"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProducerComposeRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = None


@app.post("/producer/compose")
async def producer_compose(req: ProducerComposeRequest):
    """The 'Ask Producer' flow.

    Given a free-text prompt, the producer (LLM) actually writes the full lyrics,
    derives section structure, a title, and style tags — and returns them so the
    frontend can populate the composer panel and then generate the final track.
    """
    try:
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        # None lets LLMService use the active configured model; a real model string may be passed.
        model_name = req.model_name

        # 1) Write the actual lyrics through the AI Co-Writer (real lyrics + section tags).
        try:
            lyrics = await LLMService.generate_lyrics_async(prompt, model_name)
        except Exception as e:
            lyrics = f"[Verse 1]\n{prompt}\n[Chorus]\n{prompt}"
            logger = logging.getLogger(__name__)
            logger.warning(f"Producer lyric generation failed, using prompt fallback: {e}")

        # 2) Ask the LLM to derive title + genre/style tags (JSON).
        title = None
        tags = None
        structured_caption_text = None
        try:
            derived = LLMService.enhance_prompt(
                prompt + "\n\nReturn ONLY a strict JSON object with keys: title, genre(s), mood(s), instruments.",
                model_name
            ) if hasattr(LLMService, "enhance_prompt") else None
        except Exception:
            derived = None

        # Fallback derivation without depending on enhance_prompt's return shape.
        try:
            if derived and isinstance(derived, dict):
                title = derived.get("title")
                _genre = derived.get("genre") or derived.get("genres") or ""
                _mood = derived.get("mood") or ""
                _instruments = derived.get("instruments") or ""
                tags = ", ".join(x for x in [_genre, _mood] if x)
                structured_caption_text = _instruments
        except Exception:
            pass

        if not title:
            words = [w for w in re.sub(r"[^A-Za-z0-9 ]", " ", prompt).split() if len(w) > 3]
            title = " ".join(words[:5])[:60] or "Untitled Session"
        if not tags:
            tags = "Pop, Electronic"
        if not structured_caption_text:
            structured_caption_text = "Drums, Bass, Synths, Vocals"

        return {
            "title": title,
            "lyrics": lyrics,
            "tags": tags,
            "structured_caption": {
                "global_metadata": f"Genre: {tags.split(',')[0].strip()}\nMood: Energetic & Upbeat",
                "vocal_details": "Lead Vocals: Clear, Expressive, Dynamic",
                "arrangement": f"Instrumentation: {structured_caption_text}\nProduction: Studio Master",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Music Generation & Track Extensions ---

@app.post("/generate/music")
async def generate_music(req: GenerationRequest, background_tasks: BackgroundTasks):
    import random
    seed_val = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    job = Job(
        prompt=req.prompt, 
        lyrics=req.lyrics, 
        duration_ms=req.duration_ms, 
        tags=req.tags, 
        seed=seed_val,
        model_provider=req.model_provider or "minimax_music3",
        llm_model=req.llm_model,
        parent_job_id=req.parent_job_id,
        project_id=req.project_id,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
        topk=req.topk,
        voice_profile_id=req.voice_profile_id
    )

    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)

    # Publish initial queued event to SSE subscribers
    event_manager.publish("job_update", {
        "job_id": str(job.id),
        "status": "queued",
        "prompt": job.prompt,
        "model_provider": job.model_provider
    })

    # Enqueue pipeline in background
    background_tasks.add_task(music_service.generate_task, job.id, req, engine)
    
    return {"job_id": job.id, "status": job.status, "model_provider": job.model_provider}


@app.post("/jobs/{job_id}/inpaint")
async def inpaint_track(job_id: UUID, request: dict = Body(...)):
    start_time = request.get("start_time")
    end_time = request.get("end_time")
    
    if start_time is None or end_time is None:
        raise HTTPException(status_code=400, detail="start_time and end_time required")
        
    from app.services.inpainting_service import inpainting_service
    asyncio.create_task(inpainting_service.regenerate_segment(str(job_id), float(start_time), float(end_time), engine))
    
    return {"status": "queued", "message": "In-painting started"}


@app.get("/jobs/{job_id}", response_model=Job)
def get_job_status(job_id: str):
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@app.get("/history", response_model=List[Job])
def get_history(limit: int = 50, offset: int = 0, status: Optional[str] = None, search: Optional[str] = None):
    with Session(engine) as session:
        query = select(Job).order_by(Job.created_at.desc())
        
        if status and status != 'all':
            if status == 'favorites':
                query = query.where(Job.is_favorite == True)
            else:
                query = query.where(Job.status == status)
            
        if search:
            query = query.where(or_(
                Job.title.contains(search), 
                Job.prompt.contains(search), 
                Job.tags.contains(search)
            ))
            
        jobs = session.exec(query.offset(offset).limit(limit)).all()
        return jobs


@app.post("/jobs/{job_id}/favorite", response_model=Job)
def toggle_favorite(job_id: str):
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.is_favorite = not job.is_favorite
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


@app.patch("/jobs/{job_id}", response_model=Job)
def rename_job(job_id: str, upgrade: dict):
    new_title = upgrade.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")
        
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.title = new_title
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


@app.get("/download_track/{job_id}")
def download_track(job_id: str):
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if not job or not job.audio_path:
            raise HTTPException(status_code=404, detail="Track not found")
            
        filename = job.audio_path.replace("/audio/", "")
        file_path = f"generated_audio/{filename}"
        
        safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', job.title or "untitled").strip().replace(" ", "_")
        download_name = f"{safe_title}.mp3"
        
        return FileResponse(file_path, media_type="audio/mpeg", filename=download_name)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    music_service.cancel_job(str(job_id))
    
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_id_str = str(job.id)

        # Delete physical audio file
        if job.audio_path:
            filename = job.audio_path.replace("/audio/", "")
            file_path = f"generated_audio/{filename}"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Error deleting file {file_path}: {e}")

        # Delete MIDI, MusicXML, and Stems
        for ext in [".mid", ".musicxml"]:
            aux_path = f"generated_audio/{job_id_str}{ext}"
            if os.path.exists(aux_path):
                try:
                    os.remove(aux_path)
                except Exception:
                    pass

        # Delete separated stem WAVs
        for stem_name in ["vocals", "drums", "bass", "other", "instrumental"]:
            stem_path = f"generated_audio/stems/{job_id_str}_{stem_name}.wav"
            if os.path.exists(stem_path):
                try:
                    os.remove(stem_path)
                except Exception:
                    pass

        # Remove from database and commit
        session.delete(job)
        session.commit()
            
    return {"status": "deleted", "id": job_id_str}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if music_service.cancel_job(str(job_id)):
        return {"status": "cancelling", "id": job_id}
    
    with Session(engine) as session:
        job = get_job_by_id(session, job_id)
        if job and job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            job.status = JobStatus.FAILED
            job.error_msg = "Cancelled by user"
            session.add(job)
            session.commit()
            return {"status": "cancelled", "id": job_id}
            
    raise HTTPException(status_code=400, detail="Job not active or already completed")


# --- Projects System API ---

@app.post("/projects", response_model=Project)
def create_project(data: ProjectCreate):
    with Session(engine) as session:
        project = Project(
            name=data.name,
            description=data.description,
            tags=data.tags,
            bpm=data.bpm or 120,
            key_signature=data.key_signature or "C Major",
            color=data.color or "teal",
            icon=data.icon or "folder"
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        return project


@app.get("/projects")
def list_projects():
    with Session(engine) as session:
        projects = session.exec(select(Project).order_by(Project.updated_at.desc())).all()
        result = []
        for p in projects:
            p_id_str = str(p.id)
            # Count child sessions
            jobs = session.exec(select(Job).where(Job.project_id == p_id_str)).all()
            total_duration_s = sum((j.duration_ms or 0) / 1000 for j in jobs if j.status == JobStatus.COMPLETED)
            stems_count = sum(1 for j in jobs if j.stems_json is not None)
            midi_count = sum(1 for j in jobs if j.midi_path is not None)
            
            p_dict = p.model_dump()
            p_dict["track_count"] = len(jobs)
            p_dict["total_duration_s"] = total_duration_s
            p_dict["stems_count"] = stems_count
            p_dict["midi_count"] = midi_count
            result.append(p_dict)
        return result


@app.get("/projects/{project_id}")
def get_project(project_id: UUID):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        p_id_str = str(project_id)
        jobs = session.exec(select(Job).where(Job.project_id == p_id_str).order_by(Job.created_at.desc())).all()
        
        p_dict = project.model_dump()
        p_dict["jobs"] = [j.model_dump() for j in jobs]
        p_dict["track_count"] = len(jobs)
        p_dict["total_duration_s"] = sum((j.duration_ms or 0) / 1000 for j in jobs if j.status == JobStatus.COMPLETED)
        return p_dict


@app.put("/projects/{project_id}", response_model=Project)
def update_project(project_id: UUID, data: ProjectUpdate):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if data.name is not None:
            project.name = data.name
        if data.description is not None:
            project.description = data.description
        if data.tags is not None:
            project.tags = data.tags
        if data.bpm is not None:
            project.bpm = data.bpm
        if data.key_signature is not None:
            project.key_signature = data.key_signature
        if data.color is not None:
            project.color = data.color
        if data.icon is not None:
            project.icon = data.icon
            
        project.updated_at = datetime.now(timezone.utc)
        session.add(project)
        session.commit()
        session.refresh(project)
        return project


@app.delete("/projects/{project_id}")
def delete_project(project_id: UUID):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Dissociate child jobs rather than deleting audio files
        p_id_str = str(project_id)
        jobs = session.exec(select(Job).where(Job.project_id == p_id_str)).all()
        for j in jobs:
            j.project_id = None
            session.add(j)
            
        session.delete(project)
        session.commit()
        return {"status": "deleted", "id": project_id}


@app.post("/projects/{project_id}/tracks")
def add_track_to_project(project_id: UUID, body: dict = Body(...)):
    job_id_str = body.get("job_id")
    if not job_id_str:
        raise HTTPException(status_code=400, detail="job_id is required")
        
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
            
        job = session.get(Job, UUID(job_id_str))
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job.project_id = str(project_id)
        project.updated_at = datetime.now(timezone.utc)
        session.add(job)
        session.add(project)
        session.commit()
        return {"status": "added", "project_id": project_id, "job_id": job_id_str}


@app.delete("/projects/{project_id}/tracks/{job_id}")
def remove_track_from_project(project_id: UUID, job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.project_id = None
        session.add(job)
        session.commit()
        return {"status": "removed", "job_id": job_id}


@app.get("/events")
async def events():
    async def event_generator():
        q = event_manager.subscribe()
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=1.0)
                    if "event: shutdown" in data:
                        break
                    yield data
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            event_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=1)
