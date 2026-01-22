import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session, create_engine, select, or_
from typing import List, Optional
from uuid import UUID

from app.services.music_service import music_service
from app.services.llm_service import LLMService
from app.models import Job, JobStatus, GenerationRequest, LyricsRequest, EnhancePromptRequest, InspirationRequest, LLMConfigUpdate, ProviderConfig



# Database
sqlite_file_name = "jobs.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

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

@app.post("/generate/lyrics")
def generate_lyrics(req: LyricsRequest):
    try:
        lyrics = LLMService.generate_lyrics(req.topic, req.model_name, req.seed_lyrics)
        return {"lyrics": lyrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/music")
async def generate_music(req: GenerationRequest, background_tasks: BackgroundTasks):
    # Create Job Record
    seed_val = req.seed
    if seed_val is None:
         import random
         seed_val = random.randint(0, 2**32 - 1)
         
    job = Job(prompt=req.prompt, lyrics=req.lyrics, duration_ms=req.duration_ms, tags=req.tags, seed=seed_val)
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    
    # Enqueue Background Task
    background_tasks.add_task(music_service.generate_task, job.id, req, engine)
    
    return {"job_id": job.id, "status": job.status}

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
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Delete audio file if exists
        if job.audio_path:
            # audio_path is like "/audio/filename.mp3"
            # We need to map it back to "backend/generated_audio/filename.mp3"
            filename = job.audio_path.replace("/audio/", "")
            file_path = f"generated_audio/{filename}"
            import os
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
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
