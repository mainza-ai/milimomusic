from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from uuid import UUID, uuid4
import json
from sqlmodel import Field, SQLModel
from pydantic import field_serializer, field_validator
from enum import Enum


def _canonical_uuidish(value):
    """Normalise any UUID-shaped string to the 32-hex form the DB stores.

    SQLAlchemy writes `UUID` columns as 32 hex characters, but association columns
    are plain strings and callers send the hyphenated form. Foreign keys compare
    raw text, so a validator here makes one encoding the only possibility —
    including for test fixtures and any future code path (plan 2.8).
    """
    if value is None:
        return value
    if isinstance(value, UUID):
        return value.hex
    text = str(value).strip()
    if not text or "/" in text or "?" in text:
        return value
    try:
        return UUID(text).hex
    except (ValueError, AttributeError, TypeError):
        return value


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


# Canonical defaults / limits for user-facing names. Single source of truth:
# the chat auto-rename guard and the frontend creation default must agree.
DEFAULT_SESSION_TITLE = "New session"
MAX_NAME_LENGTH = 120


def validate_display_name(value: Optional[str], *, field: str = "name", allow_none: bool = True) -> Optional[str]:
    """Strip + validate a user-facing name. Returns the stripped value.

    Raises ValueError on blank or over-long input so routes can map to 422.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field} must not be empty")
    stripped = str(value).strip()
    if not stripped:
        raise ValueError(f"{field} must not be blank")
    if len(stripped) > MAX_NAME_LENGTH:
        raise ValueError(f"{field} must be at most {MAX_NAME_LENGTH} characters")
    return stripped


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
    parent_job_id: Optional[str] = Field(default=None) # For extensions and repairs
    is_extension: bool = Field(default=False)
    is_repair: bool = Field(default=False)
    extend_from_sec: Optional[float] = Field(default=None)
    extension_mode: Optional[str] = Field(default=None) # kv_rollforward | state_restored | fresh_segment_fallback | no_parent_audio
    tail_lyrics_ok: Optional[bool] = Field(default=None) # post-hoc: tail sings the expected remainder (None = not checked)
    replay_verified: Optional[bool] = Field(default=None) # post-hoc: full-timeline render replayed the parent groove (None = not checked)
    model_prompt: Optional[str] = Field(default=None) # EXACT caption string fed to the model (verbatim replay conditioning)
    model_lyrics: Optional[str] = Field(default=None) # EXACT lyrics string fed to the model (verbatim replay conditioning)
    temperature: Optional[float] = Field(default=None)
    cfg_scale: Optional[float] = Field(default=None)
    topk: Optional[int] = Field(default=None)

    # Cover & Remix Conditioning (MuLaCover)
    is_cover: bool = Field(default=False)
    cover_mode: Optional[str] = Field(default=None) # "audio_reference" | "symbolic_midi"
    ref_audio_path: Optional[str] = Field(default=None)
    melody_midi_path: Optional[str] = Field(default=None)
    chord_midi_path: Optional[str] = Field(default=None)
    drum_midi_path: Optional[str] = Field(default=None)
    bpm: Optional[float] = Field(default=None)

    # Visual Artwork Assets
    cover_image_path: Optional[str] = Field(default=None)
    image_prompt: Optional[str] = Field(default=None)
    video_path: Optional[str] = Field(default=None)
    video_config_json: Optional[str] = Field(default=None)  # render config snapshot for faithful re-render

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
    # Per-stage provenance for every artifact produced for this job (plan 1.1/1.2):
    # {"generation": {...}, "separation": {...}, "transcription": {...},
    #  "voice": {...}, "image": {...}}. Absent stage = not attempted.
    provenance_json: Optional[str] = Field(default=None)
    # Derived one-word verdict the UI can filter on:
    #   neural    — every produced artifact came from real model inference
    #   degraded  — at least one stage fell back to DSP/heuristic output
    #   placeholder — at least one stage produced synthetic stand-in content
    quality: Optional[str] = Field(default=None)

    # Project & Session Association (real FKs — see Alembic 0006; a dangling
    # pointer used to be possible and the boot cascade code existed to paper over it)
    project_id: Optional[str] = Field(
        default=None, index=True, foreign_key="project.id", ondelete="SET NULL")
    session_id: Optional[str] = Field(
        default=None, index=True, foreign_key="session.id", ondelete="SET NULL")

    @field_validator("project_id", "session_id", "release_id", "artist_profile_id", mode="before")
    @classmethod
    def _canonical_assoc_ids(cls, v):
        return _canonical_uuidish(v)

    @field_serializer("project_id", "session_id", "release_id", "artist_profile_id",
                      "parent_job_id", when_used="json")
    def _present_assoc_ids(self, v):
        """Emit the hyphenated form so association ids look like `id` on the wire.

        The database keeps the canonical 32-hex form (foreign keys compare raw
        strings); the API stays uniform so clients never see two spellings.
        """
        if not v:
            return v
        try:
            return str(UUID(str(v)))
        except (ValueError, AttributeError, TypeError):
            return v
    mastered_path: Optional[str] = None   # B8: mastering result stored separately — original master never clobbered
    release_id: Optional[str] = Field(default=None, index=True)
    artist_profile_id: Optional[str] = Field(default=None, index=True)  # album provenance (declared: migration adds the column; bridge writes it)
    voice_profile_id: Optional[str] = Field(default=None)  # persisted per track

    # Ownership (decision D2): always "local" today, but present on every
    # user-owned row so a hosted mode is a migration rather than a rewrite.
    owner_id: str = Field(default="local", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Liveness: bumped by every pipeline progress tick. `updated_at` cannot serve
    # as a heartbeat because the pipeline only writes it on status transitions.
    last_heartbeat_at: Optional[datetime] = Field(default=None)
    # DAW session state (plan 3.8): mix, loop, tempo map, automation and clip edits
    # live here so they survive a cache clear or a different browser. Versioned JSON.
    session_state_json: Optional[str] = Field(default=None)
    # Bounded retries (plan 2.2): a transient OOM/503 must not permanently fail a job.
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=2)
    error_msg: Optional[str] = None
    is_favorite: bool = Field(default=False)
    # Soft delete (plan 2.11): a mistaken DELETE used to destroy the audio file
    # permanently. `deleted_at` hides the row; `?purge=true` is the explicit,
    # irreversible removal.
    deleted_at: Optional[datetime] = Field(default=None)


class JobUpdate(SQLModel):
    """Typed update schema for PATCH /jobs/{job_id} (replaces the untyped dict)."""
    title: Optional[str] = None
    tags: Optional[str] = None
    prompt: Optional[str] = None
    is_favorite: Optional[bool] = None
    project_id: Optional[str] = None
    cover_image_path: Optional[str] = None
    lyrics: Optional[str] = None


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
    title: str = Field(default=DEFAULT_SESSION_TITLE)
    project_id: Optional[str] = Field(default=None, index=True)
    active_job_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionCreate(SQLModel):
    title: Optional[str] = DEFAULT_SESSION_TITLE
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


class Character(SQLModel, table=True):
    """A reusable performer identity (plan 8.3).

    Consistency used to live in per-render config, so the same performer had to be
    re-described for every song. A character is the description plus a reference
    plate, referenced by id — and the same plate seeds keyframes and conditions the
    verifier across sessions.
    """

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)   # the "character bible"
    plate_path: Optional[str] = Field(default=None)    # canonical file on disk
    plate_sha256: Optional[str] = Field(default=None)  # dedup / provenance
    lora_path: Optional[str] = Field(default=None)     # reserved for per-character LoRA
    owner_id: str = Field(default="local", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id) if self.id else None,
            "name": self.name,
            "description": self.description,
            "plate_path": self.plate_path,
            "plate_sha256": self.plate_sha256,
            "has_plate": bool(self.plate_path),
            "lora_path": self.lora_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class VideoRender(SQLModel, table=True):
    """One video render attempt (take) for a job — plan 7.4.

    `Job.video_path` used to be a single string, so every re-render silently
    overwrote the previous master and A/B comparison was impossible. A take keeps
    its config, artifacts, QC verdict and payload so the UI can switch between them.
    """

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    job_id: UUID = Field(foreign_key="job.id", ondelete="CASCADE", index=True)
    task_id: Optional[str] = Field(default=None, index=True)
    kind: str = Field(default="advanced", index=True)  # advanced|lyric|reactive|variant|export
    status: str = Field(default="rendering", index=True)  # rendering|completed|failed|cancelled
    config_json: Optional[str] = Field(default=None)
    video_path: Optional[str] = Field(default=None)   # public URL, as served
    file_path: Optional[str] = Field(default=None)    # absolute path on disk
    poster_path: Optional[str] = Field(default=None)
    proxy_path: Optional[str] = Field(default=None)
    qc_json: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    duration_ms: Optional[int] = Field(default=None)
    width: Optional[int] = Field(default=None)
    height: Optional[int] = Field(default=None)
    fps: Optional[float] = Field(default=None)
    is_selected: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        qc = None
        if self.qc_json:
            try:
                qc = json.loads(self.qc_json)
            except (TypeError, json.JSONDecodeError):
                qc = None
        cfg = None
        if self.config_json:
            try:
                cfg = json.loads(self.config_json)
            except (TypeError, json.JSONDecodeError):
                cfg = None
        return {
            "id": str(self.id) if self.id else None,
            "job_id": str(self.job_id) if self.job_id else None,
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "config": cfg,
            "video_path": self.video_path,
            "file_path": self.file_path,
            "poster_path": self.poster_path,
            "proxy_path": self.proxy_path,
            "qc": qc,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_selected": self.is_selected,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


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
    lyrics: Optional[str] = None


class CoverImageRequest(SQLModel):
    prompt: str
    aspect_ratio: Optional[str] = "1:1"
    style: Optional[str] = "cinematic album cover"
    model_id: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None


class JobCoverGenerateRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    prompt: Optional[str] = None
    aspect_ratio: Optional[str] = "1:1"
    style: Optional[str] = None
    model_id: Optional[str] = None
    artist: Optional[str] = None


class VideoPlanRequest(SQLModel):
    max_clip_duration: Optional[float] = None
    model_name: Optional[str] = "wan_14b"
    visual_style: Optional[str] = "neon-cyberpunk"
    custom_style_prompt: Optional[str] = None
    bpm: Optional[float] = None
    provider: Optional[str] = "local"
    pacing_bias: Optional[int] = 0
    visible_cast: Optional[List[str]] = None
    character_desc: Optional[str] = None
    # Plan 8.17: planning must see the same performer the render will use, or the
    # storyboard the user approves describes a different person.
    character_id: Optional[str] = None
    consistency_mode: Optional[str] = "character"
    chain_scope: Optional[str] = "section"
    scenes: Optional[List[Dict[str, Any]]] = None
    use_llm: Optional[bool] = True
    force_refresh: Optional[bool] = False
    async_mode: Optional[bool] = False


class VideoRenderRequest(SQLModel):
    model_name: Optional[str] = "wan_14b"
    visual_style: Optional[str] = "neon-cyberpunk"
    custom_style_prompt: Optional[str] = None
    resolution: Optional[str] = "720p"
    aspect_ratio: Optional[str] = "16:9"
    provider: Optional[str] = "local"
    lip_sync_engine: Optional[str] = "live_portrait"
    enable_lip_sync: Optional[bool] = True
    burn_lyrics: Optional[bool] = True
    subtitle_style: Optional[str] = "neon"
    transition_style: Optional[str] = "beat_cut"
    max_clip_duration: Optional[float] = None
    mode: Optional[str] = "production_multiclip"
    face_image_path: Optional[str] = None
    character_image_path: Optional[str] = None
    pacing_bias: Optional[int] = 0
    visible_cast: Optional[List[str]] = None
    character_desc: Optional[str] = None
    characterPromptNote: Optional[str] = None
    # Subject continuity across scenes (audit E6): "off" | "character" | "chain".
    consistency_mode: Optional[str] = "character"
    # Reusable performer identity (plan 8.3): description + reference plate.
    character_id: Optional[str] = None
    # How far chaining reaches: "section" (default) resets at a section change,
    # "all" chains across the whole song.
    chain_scope: Optional[str] = "section"
    confirm_cost: Optional[bool] = False
    scenes: Optional[List[Dict[str, Any]]] = None
    clips: Optional[List[Dict[str, Any]]] = None


class LyricVideoRequest(SQLModel):
    style_preset: Optional[str] = "neon"
    aspect_ratio: Optional[str] = "16:9"
    resolution: Optional[str] = "720p"
    background_mode: Optional[str] = "cover_art"
    font_family: Optional[str] = None
    font_size_override: Optional[int] = None
    include_spectrum: Optional[bool] = False
    burn_lyrics: Optional[bool] = True
    cover_image_path: Optional[str] = None


class DirectorTreatmentRequest(SQLModel):
    model_name: Optional[str] = "wan_14b"
    max_clip_duration: Optional[float] = None
    visual_style: Optional[str] = "neon-cyberpunk"
    custom_style_prompt: Optional[str] = None
    pacing_bias: Optional[int] = 0
    visible_cast: Optional[List[str]] = None
    character_desc: Optional[str] = None


class ReimagineSceneRequest(SQLModel):
    user_instruction: Optional[str] = None
    visual_style: Optional[str] = "neon-cyberpunk"
    current_scene: Optional[Dict[str, Any]] = None
    character_desc: Optional[str] = None


class KeyframesRequest(SQLModel):
    visual_style: Optional[str] = "neon-cyberpunk"
    custom_style_prompt: Optional[str] = None
    resolution: Optional[str] = "720p"
    aspect_ratio: Optional[str] = "16:9"
    force_regenerate: bool = False
    scenes: Optional[List[Dict[str, Any]]] = None
    background: bool = False


class SceneRegenerateRequest(SQLModel):
    prompt: Optional[str] = None
    camera: Optional[str] = None
    lighting: Optional[str] = None
    visual_style: Optional[str] = "neon-cyberpunk"
    custom_style_prompt: Optional[str] = None


class GenerationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    prompt: str
    # Bounded retries for transient failures (plan 2.2).
    max_attempts: int = Field(default=2, ge=1, le=5)
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
    auto_generate_cover: Optional[bool] = True
    cover_image_model_id: Optional[str] = None

    # Cover conditioning options
    is_cover: Optional[bool] = False
    ref_audio_path: Optional[str] = None
    melody_midi_path: Optional[str] = None
    chord_midi_path: Optional[str] = None
    drum_midi_path: Optional[str] = None
    bpm: Optional[float] = None
    transcription_engine: Optional[str] = "milimo_neural"

    # Track Extension & Inpainting options
    is_extension: Optional[bool] = False
    is_repair: Optional[bool] = False
    extend_from_sec: Optional[float] = None
    target_duration_sec: Optional[float] = None
    crossfade_sec: Optional[float] = 1.5
    parent_audio_path: Optional[str] = None
    parent_prompt: Optional[str] = None
    parent_lyrics: Optional[str] = None
    parent_structured_caption: Optional[Dict[str, str]] = None
    beat_grid: Optional[Dict[str, Any]] = None
    tail_lyrics: Optional[str] = None # unsung remainder slice: fallback-only conditioning + UI preview (never steers the replay)

    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, list):
            return ", ".join(str(t) for t in v)
        return str(v)


class TrackExtendRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    target_duration_sec: float = Field(..., ge=15.0, le=600.0)
    extend_from_sec: Optional[float] = None
    additional_lyrics: Optional[str] = None
    auto_generate_lyrics: bool = Field(default=False, description="Whether to automatically generate continuation lyrics with AI. Default: False.")
    prompt: Optional[str] = None
    crossfade_sec: float = Field(default=1.5, ge=0.1, le=5.0)
    seed: Optional[int] = Field(default=None, description="Override seed for the new section (default: inherit parent seed for max continuity).")


class TrackInpaintRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    start_time: float = Field(..., ge=0.0, description="Start time of region to repair in seconds.")
    end_time: float = Field(..., ge=0.1, description="End time of region to repair in seconds.")
    prompt: Optional[str] = None
    crossfade_sec: float = Field(default=1.0, ge=0.05, le=5.0)


class CoverGenerationRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    title: Optional[str] = None
    ref_audio_path: Optional[str] = None
    melody_midi_path: Optional[str] = None
    chord_midi_path: Optional[str] = None
    drum_midi_path: Optional[str] = None
    bpm: Optional[float] = None
    lyrics: Optional[str] = None
    prompt: Optional[str] = None
    tags: Optional[Any] = None
    duration_ms: int = 120000
    temperature: float = 1.0
    cfg_scale: float = 1.5
    topk: int = 250
    model_provider: Optional[str] = "mulacover"
    transcription_engine: str = "milimo_neural"
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    voice_profile_id: Optional[str] = None

    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, list):
            return ", ".join(str(t) for t in v)
        return str(v)


class LeadSheetExtractRequest(SQLModel):
    model_config = {"protected_namespaces": ()}
    audio_path: str
    bpm: Optional[float] = None
    transcription_engine: str = "milimo_neural"


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
    anthropic: Optional[ProviderConfig] = None


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
    voice_profile_id: Optional[str] = None   # linked singing voice (voice_service profile) — A1
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

    @field_validator("profile_id", mode="before")
    @classmethod
    def _canonical_profile_id(cls, v):
        return _canonical_uuidish(v)

    @field_serializer("profile_id", when_used="json")
    def _present_profile_id(self, v):
        if not v:
            return v
        try:
            return str(UUID(str(v)))
        except (ValueError, AttributeError, TypeError):
            return v

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    profile_id: str = Field(..., index=True, foreign_key="artistprofile.id", ondelete="CASCADE")
    title: str
    description: str = ""
    status: str = "planned"               # planned | in_progress | completed
    vision_json: str = "{}"               # ExperiencerVision / orchestrator artifacts
    track_order_json: str = "[]"          # B2: explicit cur order — array of Job ids
    cover_image_path: Optional[str] = None  # A4: release artwork
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
    release_id: Optional[str] = Field(default=None, index=True)  # album/retry runs (was only in input_json — see C2 backfill)
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
    voice_profile_id: Optional[str] = None


class ArtistProfileUpdate(SQLModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    tags: Optional[str] = None
    cover_image_path: Optional[str] = None
    lore_json: Optional[str] = None   # structured world/identity document (World Builder domain)
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    voice_profile_id: Optional[str] = None


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
    vision: Optional[dict] = None  # persist the ExperiencerVision the release was saved from (honors track_target on produce)


class ReleaseUpdate(SQLModel):
    """Partial release edit; `status` is validated against the transition table."""

    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    cover_image_path: Optional[str] = None


class Playlist(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    description: Optional[str] = ""
    cover_color: Optional[str] = "from-teal-500 to-cyan-500"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlaylistTrack(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    playlist_id: str = Field(index=True)
    job_id: str = Field(index=True)
    position: int = Field(default=0)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlaylistCreate(SQLModel):
    name: str
    description: Optional[str] = ""
    cover_color: Optional[str] = None
    song_ids: Optional[List[str]] = None


class PlaylistUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cover_color: Optional[str] = None


class StudioUserProfile(SQLModel, table=True):
    id: str = Field(default="default", primary_key=True)
    artist_name: str = "Mainza Kangombe"
    bio: str = "Founder & Audio Architect. Exploring generative AI soundscapes, neural synthesis, and offline DAW workflows."
    email: Optional[str] = ""
    avatar_url: Optional[str] = None
    social_links_json: str = "{}"
    preferences_json: str = "{}"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StudioUserProfileUpdate(SQLModel):
    artist_name: Optional[str] = None
    bio: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    social_links: Optional[Dict[str, str]] = None
    preferences: Optional[Dict[str, Any]] = None


class VisualAssetType(str, Enum):
    ALBUM_COVER = "album_cover"
    CONCEPT_ART = "concept_art"
    SCENE_KEYFRAME = "scene_keyframe"
    CHARACTER_PLATE = "character_plate"
    CUSTOM = "custom"


class VisualAsset(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    asset_type: str = Field(default="concept_art", index=True)
    title: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    style: Optional[str] = None
    aspect_ratio: str = Field(default="1:1")
    image_url: str
    image_path: str
    model_id: Optional[str] = None
    # What actually rendered this image ("mflux_flux2_mlx",
    # "diffusers_neural_diffusion", "studio_procedural_raster", "user_upload").
    # Without it a Pillow gradient was indistinguishable from FLUX output.
    engine: Optional[str] = None
    seed: Optional[int] = None
    width: int = Field(default=1024)
    height: int = Field(default=1024)
    linked_job_id: Optional[str] = Field(default=None, index=True)
    linked_project_id: Optional[str] = Field(default=None, index=True)
    is_favorite: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VisualAssetCreate(SQLModel):
    prompt: str
    title: Optional[str] = None
    negative_prompt: Optional[str] = None
    style: Optional[str] = "cinematic album cover"
    aspect_ratio: Optional[str] = "1:1"
    asset_type: Optional[str] = "concept_art"
    model_id: Optional[str] = None
    linked_job_id: Optional[str] = None
    linked_project_id: Optional[str] = None


class VisualAssetUpdate(SQLModel):
    title: Optional[str] = None
    is_favorite: Optional[bool] = None
    asset_type: Optional[str] = None




# ── Typed request bodies (plan 5.2) ─────────────────────────────────────────
# These routes accepted a bare `dict`, so they had no OpenAPI contract, no 422
# validation, and `payload.get("x")` reached the database unchecked.

class ShareRequest(SQLModel):
    ttl_hours: Optional[float] = Field(default=72, ge=0.1, le=24 * 365)


class StoryboardRequest(SQLModel):
    clips: List[Dict[str, Any]] = Field(default_factory=list)
    scenes: Optional[List[Dict[str, Any]]] = None


class WorkspaceSessionRequest(SQLModel):
    state: Dict[str, Any]


class GalleryRouteRequest(SQLModel):
    media_path: str
    target_slot: str
    target_job_or_session_id: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None


class PlaylistTrackRequest(SQLModel):
    job_id: str


class ProjectTrackRequest(SQLModel):
    job_id: str


class ReleaseTrackAttachRequest(SQLModel):
    job_id: str


class ReleaseTrackOrderRequest(SQLModel):
    """Accepts the documented `job_ids` or the legacy `track_order`."""
    job_ids: Optional[List[str]] = None
    track_order: Optional[List[str]] = None


class ProduceReleaseRequest(SQLModel):
    autopilot: bool = False
    crew: Optional[Dict[str, bool]] = None
    song_plan: Optional[List[Dict[str, Any]]] = None
    budget: Optional[Dict[str, Any]] = None


class CommitVocalRequest(SQLModel):
    vocal_path: str
    master_path: Optional[str] = None


class VoiceConvertRequest(SQLModel):
    voice_profile_id: str
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    formant_preserve: bool = True
    # The API accepts 0-1 or 0-100 (percent); the handler normalises.
    dry_wet: float = Field(default=1.0, ge=0.0, le=100.0)
    f0_method: Optional[str] = None


class RetakeClipRequest(SQLModel):
    prompt: Optional[str] = None
    visual_style: Optional[str] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    reference_image_path: Optional[str] = None
    # The retake modal has always sent these; the API used to drop them silently
    # (audit U3 / plan 7.3). They are composed into the still prompt.
    camera: Optional[str] = None
    lighting: Optional[str] = None
    # Persist the edit on the server so a reload does not lose it (plan 7.3).
    persist: bool = True


class EditorRetakeRequest(SQLModel):
    asset_path: str


class ActiveModelRequest(SQLModel):
    model_id: str


class ProfileCoverRequest(SQLModel):
    cover_image_path: str


class AgentResumeRequest(SQLModel):
    approve: bool = True
    note: Optional[str] = None


class VideoStoryboardRequest(SQLModel):
    visual_style: Optional[str] = None
    custom_style_prompt: Optional[str] = None


class VideoRenderRequestModel(SQLModel):
    visual_style: Optional[str] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    burn_lyrics: bool = True
    audio_reactive: bool = True


class ActiveVideoEngineRequest(SQLModel):
    engine: Optional[str] = None
    model_id: Optional[str] = None


class EditorProjectRequest(SQLModel):
    """An editor project is an opaque, versioned document; only the id is required."""
    model_config = {"extra": "allow"}

    project_id: str
