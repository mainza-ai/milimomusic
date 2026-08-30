from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
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
    tags: Optional[str] = None  # Style / Tags
    seed: Optional[int] = None
    audio_path: Optional[str] = None
    duration_ms: int = 240000

    # Generation Model & Provider
    model_provider: Optional[str] = Field(default="minimax_music3")
    llm_model: Optional[str] = Field(default=None) # Track which LLM was used for lyrics
    parent_job_id: Optional[str] = Field(default=None) # For extensions
    temperature: Optional[float] = Field(default=None)
    cfg_scale: Optional[float] = Field(default=None)
    topk: Optional[int] = Field(default=None)

    # Visual Artwork Assets
    cover_image_path: Optional[str] = Field(default=None)
    image_prompt: Optional[str] = Field(default=None)

    # v2 Multitrack & MuScriptor Transcription Assets
    midi_path: Optional[str] = Field(default=None)
    musicxml_path: Optional[str] = Field(default=None)
    notes_json: Optional[str] = Field(default=None)         # JSON serialized NoteEvent list
    stems_json: Optional[str] = Field(default=None)         # JSON serialized {vocals, drums, bass, other, instrumental}
    beat_grid_json: Optional[str] = Field(default=None)     # JSON serialized BeatGrid
    timed_lyrics_json: Optional[str] = Field(default=None)  # JSON serialized word-level timestamps
    structured_caption_json: Optional[str] = Field(default=None)
    
    # Generation provenance: whether the track came from genuine model inference
    # or the procedural fallback synth (surfaced to the UI, never silent).
    used_fallback_synth: bool = Field(default=False)
    fallback_reason: Optional[str] = Field(default=None)

    # Project & Session Association
    project_id: Optional[str] = Field(default=None, index=True)
    session_id: Optional[str] = Field(default=None, index=True)
    mastered_path: Optional[str] = None   # B8: mastering result stored separately — original master never clobbered
    release_id: Optional[str] = Field(default=None, index=True)
    artist_profile_id: Optional[str] = Field(default=None, index=True)  # album provenance (declared: migration adds the column; bridge writes it)
    voice_profile_id: Optional[str] = Field(default=None)  # persisted per track

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_msg: Optional[str] = None
    is_favorite: bool = Field(default=False)


class Project(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    name: str
    description: Optional[str] = None
    cover_image_path: Optional[str] = Field(default=None)
    image_prompt: Optional[str] = Field(default=None)
    tags: Optional[str] = None # Genre / Target Style
    bpm: Optional[int] = Field(default=120)
    key_signature: Optional[str] = Field(default="C Major")
    color: Optional[str] = Field(default="teal") # 'teal' | 'cyan' | 'amber' | 'emerald' | 'sky'
    icon: Optional[str] = Field(default="folder")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProjectCreate(SQLModel):
    name: str
    description: Optional[str] = None
    cover_image_path: Optional[str] = None
    image_prompt: Optional[str] = None
    tags: Optional[str] = None
    bpm: Optional[int] = 120
    key_signature: Optional[str] = "C Major"
    color: Optional[str] = "teal"
    icon: Optional[str] = "folder"


class ProjectUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cover_image_path: Optional[str] = None
    image_prompt: Optional[str] = None
    tags: Optional[str] = None
    bpm: Optional[int] = None
    key_signature: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None


class Session(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    title: str = Field(default="New session")
    project_id: Optional[str] = Field(default=None, index=True)
    active_job_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionCreate(SQLModel):
    title: Optional[str] = "New session"
    project_id: Optional[str] = None
    active_job_id: Optional[str] = None


class SessionUpdate(SQLModel):
    title: Optional[str] = None
    project_id: Optional[str] = None
    active_job_id: Optional[str] = None


class SessionMessage(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id", index=True)
    role: str = Field(default="user") # 'user' | 'producer' | 'system'
    content: str
    audio_attachment_path: Optional[str] = None
    generated_job_id: Optional[str] = None
    preset_data_json: Optional[str] = None # JSON serialized preset parameters
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMessageCreate(SQLModel):
    role: str = "user"
    content: str
    audio_attachment_path: Optional[str] = None
    generated_job_id: Optional[str] = None
    preset_data_json: Optional[str] = None


class CoverPromptRequest(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[str] = None
    genre: Optional[str] = None


class CoverImageRequest(SQLModel):
    prompt: str
    aspect_ratio: Optional[str] = "1:1"
    style: Optional[str] = "cinematic album cover"


class GenerationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    prompt: str
    lyrics: Optional[str] = None
    title: Optional[str] = None
    duration_ms: int = 30000
    temperature: float = 1.0
    cfg_scale: float = 1.5
    topk: int = 50
    tags: Optional[Any] = None # Allow list or string, validator will fix
    seed: Optional[int] = None
    model_provider: Optional[str] = "minimax_music3" # MiniMax Music 3 default
    llm_model: Optional[str] = None
    parent_job_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    cover_image_path: Optional[str] = None
    image_prompt: Optional[str] = None
    is_instrumental: Optional[bool] = False
    structured_caption: Optional[Dict[str, str]] = None
    voice_profile_id: Optional[str] = None

    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, list):
            return ", ".join(str(t) for t in v)
        return str(v)


class LyricsRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    topic: str
    model_name: Optional[str] = None
    seed_lyrics: Optional[str] = None
    tags: Optional[Any] = None
    model_provider: Optional[str] = "minimax_music3"

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
    tags: Optional[Any] = None
    chat_history: Optional[list[dict[str, Any]]] = None 
    model_provider: Optional[str] = "minimax_music3"
    
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
    model_provider: Optional[str] = "minimax_music3"


class RewriteCaptionRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    concept: str
    lyrics: Optional[str] = None
    tags: Optional[str] = None
    model_name: Optional[str] = None
    model_provider: Optional[str] = "minimax_music3"


class InspirationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    model_name: Optional[str] = None
    model_provider: Optional[str] = "minimax_music3"


class ProviderConfig(SQLModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class LLMConfigUpdate(SQLModel):
    provider: Optional[str] = None
    nvidia: Optional[ProviderConfig] = None
    openai: Optional[ProviderConfig] = None
    gemini: Optional[ProviderConfig] = None
    openrouter: Optional[ProviderConfig] = None
    lmstudio: Optional[ProviderConfig] = None
    ollama: Optional[ProviderConfig] = None
    deepseek: Optional[ProviderConfig] = None
    opencode: Optional[ProviderConfig] = None
    omlx: Optional[ProviderConfig] = None


class VoiceProfileCreate(SQLModel):
    name: str
    description: str
    consent_confirmed: bool
    f0_method: str = "rmvpe"


class MasteringRequest(SQLModel):
    target_lufs: float = -14.0
    reference_job_id: Optional[str] = None


# ---------------------------------------------------------------------------
# AI Agent Foundation — identity, assignment, releases, run ledger
# (wiki/concepts/artist-profiles-vision.md · wiki/concepts/agent-foundation.md)
# ---------------------------------------------------------------------------
class ArtistProfile(SQLModel, table=True):
    """The unit of creative identity: unlimited per project, agents assigned per artist."""

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    project_id: Optional[str] = Field(default=None, index=True)
    name: str
    bio: str = ""
    lore_json: str = "{}"                 # structured world/identity document (World Builder owns)
    tags: str = ""                        # default style DNA for this artist
    cover_image_path: Optional[str] = None
    default_provider: Optional[str] = None   # optional per-artist LLM override
    default_model: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentAssignment(SQLModel, table=True):
    """Which agent serves an artist — data, not code. Per-artist model overrides ride here."""

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    profile_id: str = Field(..., index=True)
    role: str                             # e.g. world_builder | experiencer | songwriter | producer
    agent_name: str                       # registry key (app.agents.registry.AGENTS)
    model_provider: Optional[str] = None  # override chain head for this assignment
    model: Optional[str] = None
    config_json: str = "{}"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Release(SQLModel, table=True):
    """An album/EP container. Tracks (Jobs) attach via release_id."""

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    profile_id: str = Field(..., index=True)
    title: str
    description: str = ""
    status: str = "planned"               # planned | in_progress | completed
    vision_json: str = "{}"               # ExperiencerVision / orchestrator artifacts
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRun(SQLModel, table=True):
    """Ledger of every agent execution: status, I/O, typed errors, usage."""

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    agent_name: str = Field(..., index=True)
    status: str = Field(default="running", index=True)  # running|succeeded|failed|cancelled
    input_json: str = "{}"
    output_json: str = ""
    error_type: str = ""
    error_message: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    attempts_json: str = "[]"             # per-attempt records from the resilience policy
    # Orchestration columns (album runner): resume cursor, hierarchy, budgets.
    parent_run_id: Optional[str] = Field(default=None, index=True)
    state_json: str = "{}"
    progress: int = 0
    budget_json: str = "{}"
    session_id: Optional[str] = Field(default=None, index=True)
    project_id: Optional[str] = Field(default=None, index=True)
    profile_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None


class ArtistProfileCreate(SQLModel):
    project_id: Optional[str] = None
    name: str
    bio: str = ""
    tags: str = ""
    cover_image_path: Optional[str] = None
    default_provider: Optional[str] = None
    default_model: Optional[str] = None


class ArtistProfileUpdate(SQLModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    tags: Optional[str] = None
    cover_image_path: Optional[str] = None
    lore_json: Optional[str] = None   # structured world/identity document (World Builder domain)
    default_provider: Optional[str] = None
    default_model: Optional[str] = None


class AgentAssignmentCreate(SQLModel):
    role: str
    agent_name: str
    model_provider: Optional[str] = None
    model: Optional[str] = None
    config_json: str = "{}"


class AgentAssignmentSet(SQLModel):
    """Replace-all semantics for an artist's crew (idempotent, no drift)."""

    assignments: List[AgentAssignmentCreate]


class AgentRunRequest(SQLModel):
    """Generic agent invocation envelope; `input` is validated against the agent's schema."""

    input: Dict[str, Any]
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    profile_id: Optional[str] = None


class ReleaseCreate(SQLModel):
    profile_id: str
    title: str
    description: str = ""


class ReleaseUpdate(SQLModel):
    """Partial release edit; `status` is validated against the transition table."""

    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
