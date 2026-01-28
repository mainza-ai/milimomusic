import asyncio
import os
# Enable MPS Fallback for operations not supported on MPS (e.g. large channel conv1d)
# Must be set before torch is imported!
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
from contextlib import asynccontextmanager

# Configure logging to suppress verbose debug output from dependencies
# optimization: Configure BEFORE imports to catch everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Silence specific noisy loggers
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # Reduce access log spam if needed, or keep INFO
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session, create_engine, select, or_, text
from typing import List, Optional
from uuid import UUID

from app.services.music_service import music_service
from app.services.llm_service import LLMService
from app.models import Job, JobStatus, GenerationRequest, LyricsRequest, LyricsChatRequest, EnhancePromptRequest, InspirationRequest, LLMConfigUpdate, ProviderConfig

# Database
sqlite_file_name = "jobs.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # Enable WAL mode for better concurrency (Readers don't block Writers)
    # This prevents "database is locked" errors when deleting tracks while generating.
    with Session(engine) as session:
        session.exec(text("PRAGMA journal_mode=WAL;"))
        session.commit()

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    # Initialize Heartlib (Non-blocking usually, but loading big models might take a sec)
    # We trigger it but don't await strictly if we want faster startup, 
    # but for safety we await to ensure model is ready before traffic.
    await music_service.initialize() 
    yield
    # Shutdown Event Manager (Closes SSE connections)
    event_manager.shutdown()
    music_service.shutdown_all()

app = FastAPI(lifespan=lifespan, title="Milimo Music API")

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
    # For debugging, also print the raw body if accessible
    try:
        body = await request.json()
        print(f"Validation Body: {body}")
    except:
        pass
    return await request_validation_exception_handler(request, exc)

# Static Files (Audio Serving)
import os
os.makedirs("generated_audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": music_service.pipeline is not None}

@app.get("/models/lyrics")
def get_lyrics_models():
    return {"models": LLMService.get_models()}

@app.get("/config/llm")
def get_llm_config():
    return LLMService.get_config()

@app.post("/config/llm")
def  update_llm_config(config: LLMConfigUpdate):
    try:
        # Update provider if specified
        if config.provider:
            LLMService.set_active_provider(config.provider)
        
        # Update specific provider settings
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
            
        return LLMService.get_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/fetch-models")
def fetch_models(request: LLMConfigUpdate):
    """
    Fetch models for a specific provider using passed credentials/url.
    Does NOT save the config.
    """
    try:
        provider = request.provider
        if not provider:
            raise HTTPException(status_code=400, detail="Provider required")
        
        # Extract relevant credentials from the request body
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

        models = LLMService.fetch_available_models(provider, api_key, base_url)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


@app.post("/generate/enhance_prompt")
def enhance_prompt(req: EnhancePromptRequest):
    try:
        result = LLMService.enhance_prompt(req.concept, req.model_name)
        return result
    except Exception as e:
        # Fallback
        return {"topic": req.concept, "tags": "Pop"}

@app.post("/generate/evaluate_inspiration")
def generate_inspiration(req: InspirationRequest):
    try:
        result = LLMService.generate_inspiration(req.model_name)
        return result
    except Exception as e:
        return {"topic": "A futuristic city in the clouds", "tags": "Electronic, ambient, sci-fi"}

@app.post("/generate/styles")
def generate_styles(req: InspirationRequest):
    # Reusing InspirationRequest since we just need the model_name
    try:
        styles = LLMService.generate_styles_list(req.model_name)
        return {"styles": styles}
    except Exception:
        return {"styles": ["Pop", "Rock", "Jazz"]} # Fallback

# --- Style Management API ---
from app.services.style_registry import StyleRegistry, Style
from pydantic import BaseModel

class StyleCreate(BaseModel):
    name: str
    description: Optional[str] = None

class PathsConfig(BaseModel):
    model_config = {"protected_namespaces": ()}  # Allow model_ prefix
    model_directory: Optional[str] = None
    checkpoints_directory: Optional[str] = None
    datasets_directory: Optional[str] = None

@app.get("/styles")
def get_styles():
    """Get all available styles (official + custom + trained)."""
    registry = StyleRegistry()
    styles = registry.get_all_styles()
    return {"styles": [s.to_dict() for s in styles]}

@app.post("/styles/custom")
def add_custom_style(style: StyleCreate):
    """Add a new custom style."""
    try:
        registry = StyleRegistry()
        created = registry.add_custom_style(style.name, style.description)
        return {"style": created.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/styles/custom/{name}")
def remove_custom_style(name: str):
    """Remove a custom style."""
    registry = StyleRegistry()
    if registry.remove_custom_style(name):
        return {"status": "deleted", "name": name}
    raise HTTPException(status_code=404, detail=f"Custom style '{name}' not found")

# --- Path Configuration API ---
from app.services.config_manager import ConfigManager

@app.get("/config/paths")
def get_paths_config():
    """Get current path configuration."""
    config = ConfigManager().get_config()
    return config.get("paths", {})

@app.post("/config/paths")
def update_paths_config(paths: PathsConfig):
    """Update path configuration."""
    updates = paths.model_dump(exclude_unset=True)
    if updates:
        ConfigManager().update_config({"paths": updates})
    return ConfigManager().get_config().get("paths", {})

@app.post("/config/paths/validate")
def validate_paths(paths: PathsConfig):
    """Validate if paths contain valid model/checkpoint files."""
    import os
    results = {}
    
    if paths.model_directory:
        model_dir = os.path.expanduser(paths.model_directory)
        # Check for HeartMuLa model structure
        heartmula_path = os.path.join(model_dir, "HeartMuLa-oss-3B")
        heartcodec_path = os.path.join(model_dir, "HeartCodec-oss")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        results["model_directory"] = {
            "path": model_dir,
            "exists": os.path.isdir(model_dir),
            "has_heartmula": os.path.isdir(heartmula_path),
            "has_heartcodec": os.path.isdir(heartcodec_path),
            "has_tokenizer": os.path.isfile(tokenizer_path),
            "valid": all([
                os.path.isdir(model_dir),
                os.path.isdir(heartmula_path),
                os.path.isdir(heartcodec_path),
                os.path.isfile(tokenizer_path)
            ])
        }
    
    if paths.checkpoints_directory:
        ckpt_dir = os.path.expanduser(paths.checkpoints_directory)
        results["checkpoints_directory"] = {
            "path": ckpt_dir,
            "exists": os.path.isdir(ckpt_dir) if os.path.exists(ckpt_dir) else False,
            "valid": True  # Will be created if doesn't exist
        }
    
    if paths.datasets_directory:
        data_dir = os.path.expanduser(paths.datasets_directory)
        results["datasets_directory"] = {
            "path": data_dir,
            "exists": os.path.isdir(data_dir) if os.path.exists(data_dir) else False,
            "valid": True  # Will be created if doesn't exist
        }
    
    return results

# --- Fine-Tuning API ---
from app.services.fine_tuning_service import (
    fine_tuning_service, 
    TrainingConfig, 
    Dataset as FTDataset,
    TrainingJob as FTJob
)
from fastapi import UploadFile, File, Form

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

# Dataset Endpoints
@app.post("/training/datasets")
def create_dataset(data: DatasetCreate):
    """Create a new training dataset."""
    dataset = fine_tuning_service.create_dataset(data.name, data.styles)
    return {"dataset": dataset.to_dict()}

@app.get("/training/datasets")
def list_datasets():
    """List all training datasets."""
    datasets = fine_tuning_service.list_datasets()
    return {"datasets": [d.to_dict() for d in datasets]}

@app.get("/training/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """Get a specific dataset."""
    dataset = fine_tuning_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"dataset": dataset.to_dict()}

@app.post("/training/datasets/{dataset_id}/audio")
async def upload_audio(dataset_id: str, file: UploadFile = File(...), caption: str = Form(...)):
    """Upload an audio file to a dataset."""
    try:
        content = await file.read()
        audio_file = fine_tuning_service.add_audio_file(
            dataset_id, file.filename, caption, content
        )
        return {"audio_file": {"filename": audio_file.filename, "caption": audio_file.caption}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi.responses import FileResponse

@app.get("/training/datasets/{dataset_id}/audio/{filename}")
def get_dataset_audio(dataset_id: str, filename: str):
    """Serve an audio file from a dataset for preview."""
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
    """Delete an audio file from a dataset."""
    success = fine_tuning_service.remove_audio_file(dataset_id, filename)
    if not success:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return {"success": True}

class CaptionUpdate(BaseModel):
    caption: str

@app.put("/training/datasets/{dataset_id}/audio/{filename}")
def update_audio_caption(dataset_id: str, filename: str, data: CaptionUpdate):
    """Update the caption/lyrics for an audio file."""
    success = fine_tuning_service.update_audio_caption(dataset_id, filename, data.caption)
    if not success:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return {"success": True}

@app.get("/training/datasets/{dataset_id}/validate")
def validate_dataset(dataset_id: str):
    """Check if dataset meets minimum requirements (10 files)."""
    return fine_tuning_service.validate_dataset(dataset_id)

@app.put("/training/datasets/{dataset_id}")
def update_dataset(dataset_id: str, data: DatasetCreate):
    """Update a dataset's name or styles."""
    dataset = fine_tuning_service.update_dataset(dataset_id, data.name, data.styles)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"dataset": dataset.to_dict()}

@app.delete("/training/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Delete a dataset and all its files."""
    success = fine_tuning_service.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True}

class PreprocessRequest(BaseModel):
    force: bool = True

@app.post("/training/datasets/{dataset_id}/preprocess")
async def preprocess_dataset(dataset_id: str, request: PreprocessRequest = PreprocessRequest()):
    """Preprocess a dataset (tokenize audio files with correct tag format)."""
    import asyncio
    loop = asyncio.get_event_loop()
    # Run in executor since this is blocking
    result = await loop.run_in_executor(
        None, 
        lambda: fine_tuning_service.preprocess_dataset(dataset_id, force=request.force)
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Preprocessing failed"))
    return result

# Training Job Endpoints

@app.post("/training/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    if fine_tuning_service.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    raise HTTPException(status_code=404, detail="Job not found or not running")

@app.post("/training/jobs")
def create_training_job(config: TrainingConfigRequest):
    """Start a new training job."""
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
    """List all training jobs."""
    jobs = fine_tuning_service.list_jobs()
    return {"jobs": [j.to_dict() for j in jobs]}

@app.get("/training/jobs/{job_id}")
def get_training_job(job_id: str):
    """Get training job status."""
    job = fine_tuning_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job.to_dict()}

@app.get("/training/jobs/{job_id}/logs")
def get_training_logs(job_id: str, offset: int = 0):
    """Get training logs."""
    logs = fine_tuning_service.get_job_logs(job_id, offset)
    return {"logs": logs, "offset": offset + len(logs)}

@app.delete("/training/jobs/{job_id}")
def delete_training_job(job_id: str):
    """Delete a training job."""
    if fine_tuning_service.delete_job(job_id):
        return {"success": True, "job_id": job_id}
    raise HTTPException(status_code=404, detail="Job not found")

# Checkpoint Endpoints
@app.get("/training/checkpoints")
def list_checkpoints():
    """List all model checkpoints."""
    checkpoints = fine_tuning_service.list_checkpoints()
    return {"checkpoints": [c.to_dict() for c in checkpoints]}

@app.post("/training/checkpoints/{checkpoint_id}/activate")
async def activate_checkpoint(checkpoint_id: str):
    """Set a checkpoint as active."""
    if fine_tuning_service.activate_checkpoint(checkpoint_id):
        # 1. Unload current (Sets pipeline to None)
        music_service.unload_lora()
        
        # 2. Re-initialize (Loads base model + Active LoRA automatically)
        await music_service.initialize()
        
        # 3. Refresh style registry so new trained styles appear
        from app.services.style_registry import StyleRegistry
        StyleRegistry().refresh()
        
        return {"status": "activated", "checkpoint_id": checkpoint_id, "message": "Model reloaded with new weights"}
    raise HTTPException(status_code=404, detail="Checkpoint not found")

@app.post("/training/checkpoints/deactivate")
async def deactivate_checkpoint():
    """Deactivate any active checkpoint (revert to base model)."""
    fine_tuning_service.deactivate_all_checkpoints()
    
    # Reload model stack
    music_service.unload_lora()
    await music_service.initialize()
    
    # Refresh style registry
    from app.services.style_registry import StyleRegistry
    StyleRegistry().refresh()
    
    return {"status": "deactivated", "message": "Reverted to base model"}

@app.delete("/training/checkpoints/{checkpoint_id}")
def delete_checkpoint(checkpoint_id: str):
    """Delete a checkpoint."""
    if fine_tuning_service.delete_checkpoint(checkpoint_id):
        return {"status": "deleted", "checkpoint_id": checkpoint_id}
    raise HTTPException(status_code=404, detail="Checkpoint not found")

@app.post("/generate/lyrics")
async def generate_lyrics(req: LyricsRequest):
    try:
        lyrics = await LLMService.generate_lyrics_async(req.topic, req.model_name, req.seed_lyrics, req.tags)
        return {"lyrics": lyrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/lyrics-chat")
async def chat_with_lyrics(req: LyricsChatRequest):
    try:
        result = await LLMService.chat_with_lyrics_async(
            req.current_lyrics, 
            req.user_message, 
            req.model_name, 
            req.chat_history, 
            req.topic, 
            req.get_tags_string()  # Normalize tags array to string
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/music")
async def generate_music(req: GenerationRequest, background_tasks: BackgroundTasks):
    # Create Job Record
    seed_val = req.seed
    if seed_val is None:
         import random
         seed_val = random.randint(0, 2**32 - 1)
         
    job = Job(
        prompt=req.prompt, 
        lyrics=req.lyrics, 
        duration_ms=req.duration_ms, 
        tags=req.tags, 
        seed=seed_val,
        llm_model=req.llm_model,
        parent_job_id=req.parent_job_id,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
        topk=req.topk
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    
    # Enqueue Background Task
    background_tasks.add_task(music_service.generate_task, job.id, req, engine)
    
    return {"job_id": job.id, "status": job.status}

from fastapi import Body
@app.post("/jobs/{job_id}/inpaint")
async def inpaint_track(job_id: UUID, request: dict = Body(...)):
    """
    Repair a segment of audio.
    Body: { "start_time": 10.0, "end_time": 15.0 }
    """
    start_time = request.get("start_time")
    end_time = request.get("end_time")
    
    if start_time is None or end_time is None:
        raise HTTPException(status_code=400, detail="start_time and end_time required")
        
    from app.services.inpainting_service import inpainting_service
    # Run in background
    # Note: inpainting_service uses same DB engine reference
    asyncio.create_task(inpainting_service.regenerate_segment(str(job_id), float(start_time), float(end_time), engine))
    
    return {"status": "queued", "message": "In-painting started"}

@app.get("/jobs/{job_id}", response_model=Job)
def get_job_status(job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
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
            # Case insensitive search usually requires ilike in Postgres, but SQLite uses LIKE which is case-insensitive by default for ASCII.
            query = query.where(or_(
                Job.title.contains(search), 
                Job.prompt.contains(search), 
                Job.tags.contains(search)
            ))
            
        jobs = session.exec(query.offset(offset).limit(limit)).all()
        return jobs

@app.post("/jobs/{job_id}/favorite", response_model=Job)
def toggle_favorite(job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Toggle
        job.is_favorite = not job.is_favorite
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

@app.patch("/jobs/{job_id}", response_model=Job)
def rename_job(job_id: UUID, upgrade: dict):
    # Minimal schema for update, expecting {"title": "new name"}
    new_title = upgrade.get("title")
    if not new_title:
        raise HTTPException(status_code=400, detail="Title is required")
        
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job.title = new_title
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

@app.get("/download_track/{job_id}")
def download_track(job_id: UUID):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job or not job.audio_path:
            raise HTTPException(status_code=404, detail="Track not found")
            
        # audio_path is "/audio/filename.mp3" -> "backend/generated_audio/filename.mp3"
        filename = job.audio_path.replace("/audio/", "")
        file_path = f"generated_audio/{filename}"
        
        # Sanitize Title for Filename
        import re
        safe_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', job.title or "untitled")
        safe_title = safe_title.strip().replace(" ", "_")
        download_name = f"{safe_title}.mp3"
        
        return FileResponse(file_path, media_type="audio/mpeg", filename=download_name)

@app.delete("/jobs/{job_id}")
def delete_job(job_id: UUID):
    # 1. Cancel active task (Release GPU)
    music_service.cancel_job(str(job_id))
    
    file_path = None
    
    # 2. Read Phase (Short Lock)
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Prepare file path
        if job.audio_path:
            filename = job.audio_path.replace("/audio/", "")
            file_path = f"generated_audio/{filename}"

    # 3. I/O Phase (No Lock)
    if file_path:
        import os
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                
    # 4. Write Phase (Short Lock)
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).one_or_none()
        if job:
            session.delete(job)
            session.commit()
            
    return {"status": "deleted", "id": job_id}

@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: UUID):
    # Try to cancel running task via service
    if music_service.cancel_job(str(job_id)):
        return {"status": "cancelling", "id": job_id}
    
    # If not running, maybe update status in DB directly?
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job and job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            job.status = JobStatus.FAILED
            job.error_msg = "Cancelled by user"
            session.add(job)
            session.commit()
            return {"status": "cancelled", "id": job_id}
            
    raise HTTPException(status_code=400, detail="Job not active or already completed")

from fastapi.responses import StreamingResponse
from app.services.music_service import event_manager

@app.get("/events")
async def events():
    async def event_generator():
        q = event_manager.subscribe()
        try:
            while True:
                # Wait for new event using asyncio.wait_for to allow checking client disconnected
                # actually Queue.get is async so it yields control
                try:
                    data = await asyncio.wait_for(q.get(), timeout=1.0)
                    if "event: shutdown" in data:
                        break
                    yield data
                except asyncio.TimeoutError:
                    # Wake up loop to check for cancellation or keep-alive
                    # yield ": keep-alive\n\n" # Optional: send comment to keep client connection alive
                    continue
        except asyncio.CancelledError:
             # Server shutting down
             pass
        except Exception:
            pass
        finally:
            event_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=1)
