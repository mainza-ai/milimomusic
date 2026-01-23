from datetime import datetime, timezone
from typing import Optional, Any
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel
from pydantic import field_validator
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
    is_favorite: bool = Field(default=False)

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
    model_name: Optional[str] = None
    seed_lyrics: Optional[str] = None
    tags: Optional[Any] = None

    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, list):
            return ", ".join(str(t) for t in v)
        return str(v)

class LyricsChatRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    current_lyrics: str
    user_message: str
    model_name: Optional[str] = None
    topic: Optional[str] = None
    tags: Optional[Any] = None  # Accept string or list, will be normalized
    chat_history: Optional[list[dict[str, Any]]] = None 
    
    def get_tags_string(self) -> Optional[str]:
        """Normalize tags to string format."""
        if self.tags is None:
            return None
        if isinstance(self.tags, list):
            return ", ".join(str(t) for t in self.tags)
        return str(self.tags)

class EnhancePromptRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    concept: str
    model_name: Optional[str] = None

class InspirationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    model_name: Optional[str] = None

class ProviderConfig(SQLModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None

class LLMConfigUpdate(SQLModel):
    provider: Optional[str] = None
    openai: Optional[ProviderConfig] = None
    gemini: Optional[ProviderConfig] = None
    openrouter: Optional[ProviderConfig] = None
    lmstudio: Optional[ProviderConfig] = None
    ollama: Optional[ProviderConfig] = None
    deepseek: Optional[ProviderConfig] = None

