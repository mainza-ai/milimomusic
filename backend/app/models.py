from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    status: JobStatus = Field(default=JobStatus.QUEUED)
    title: Optional[str] = None
    prompt: str
    lyrics: Optional[str] = None
    tags: Optional[str] = None  # Added field for Style/Tags
    seed: Optional[int] = None # Added for Seed Consistency
    audio_path: Optional[str] = None
    duration_ms: int = 240000
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_msg: Optional[str] = None

class GenerationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    prompt: str
    lyrics: Optional[str] = None
    duration_ms: int = 30000
    temperature: float = 1.0
    cfg_scale: float = 1.5
    topk: int = 50
    tags: Optional[str] = None
    seed: Optional[int] = None # Added for Seed Consistency
    llm_model: Optional[str] = None # specific LLM usage for title/lyrics
    parent_job_id: Optional[str] = None # For Track Extension (Phase 9)

class LyricsRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    topic: str
    model_name: str = "llama3"
    seed_lyrics: Optional[str] = None

class EnhancePromptRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    concept: str
    model_name: str = "llama3"

class InspirationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "llama3"
