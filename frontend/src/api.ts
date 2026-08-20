import axios from 'axios';

// API base is configurable via Vite env; defaults to local backend for development.
const API_BASE_URL: string = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

export { API_BASE_URL };

export interface NoteEvent {
    pitch: number;
    start_time: number;
    duration?: number;
    end_time?: number;
    velocity: number;
    instrument: string;
    channel?: number;
    note_name?: string;
}

export interface BeatGrid {
    bpm: number;
    beats_per_bar: number;
    first_downbeat: number;
    onset_delay: number;
}

export interface StemsMap {
    vocals?: string;
    drums?: string;
    bass?: string;
    other?: string;
    instrumental?: string;
    /** Dynamic per-instrument stems keyed by instrument name (from transcription). */
    instrumental_parts?: Record<string, string>;
    /** General MIDI program (instrument) number per instrument name. */
    instrument_programs?: Record<string, number>;
}

export interface TimedWord {
    word: string;
    start: number;
    end: number;
}

export interface TimedLine {
    text: string;
    start: number;
    end: number;
    words: TimedWord[];
}

export interface Job {
    id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    title?: string;
    prompt: string;
    lyrics?: string;
    tags?: string;
    audio_path?: string;
    error_msg?: string;
    created_at: string;
    duration_ms?: number;
    seed?: number;
    is_favorite?: boolean;

    // v2 Generation & Provider
    model_provider?: string;
    llm_model?: string;
    parent_job_id?: string;
    temperature?: number;
    cfg_scale?: number;
    topk?: number;

    // v2 DAW Multitrack & MuScriptor Transcription Assets
    midi_path?: string;
    musicxml_path?: string;
    notes_json?: string;
    stems_json?: string;
    beat_grid_json?: string;
    timed_lyrics_json?: string;
    structured_caption_json?: string;
    voice_profile_id?: string;
    project_id?: string;
}

export interface Project {
    id: string;
    name: string;
    description?: string;
    tags?: string;
    bpm?: number;
    key_signature?: string;
    color?: string; // 'teal' | 'cyan' | 'amber' | 'emerald' | 'sky'
    icon?: string;
    created_at: string;
    updated_at: string;
    track_count?: number;
    total_duration_s?: number;
    stems_count?: number;
    midi_count?: number;
    jobs?: Job[];
}

export interface ProjectCreate {
    name: string;
    description?: string;
    tags?: string;
    bpm?: number;
    key_signature?: string;
    color?: string;
    icon?: string;
}

export interface ProjectUpdate {
    name?: string;
    description?: string;
    tags?: string;
    bpm?: number;
    key_signature?: string;
    color?: string;
    icon?: string;
}

export interface ModelVariant {
    id: string;
    name: string;
    architecture: string;
    quantization: string;
    size_gb: number;
    is_installed: boolean;
    local_path?: string;
    license: string;
    recommended_hardware: string;
    is_default: boolean;
}

export interface HardwareProfile {
    os_name: string;
    architecture: string;
    processor: string;
    has_cuda: boolean;
    has_mps: boolean;
    hardware_tier: string;
    tier_description: string;
    can_run_minimax_full: boolean;
    can_run_heartmula: boolean;
}

export interface GenerationCapabilities {
    provider_id: string;
    display_name: string;
    description: string;
    version: string;
    max_duration_sec: number;
    supports_structured_caption: boolean;
    supports_section_tags: boolean;
    supports_lora: boolean;
    supports_voice_conversion: boolean;
    supports_track_extension: boolean;
    supports_segment_repair: boolean;
    recommended_hardware: string;
    license_class: string;
    default_sample_rate: number;
}

export interface VoiceProfile {
    id: string;
    name: string;
    description: string;
    sample_audio_path?: string;
    status: 'ready' | 'training' | 'failed';
    created_at: string;
    consent_confirmed: boolean;
    f0_method: string;
    sample_rate: number;
    is_default?: boolean;
}

export const api = {
    toggleFavorite: async (jobId: string) => {
        const res = await axios.post(`${API_BASE_URL}/jobs/${jobId}/favorite`);
        return res.data;
    },

    checkHealth: async () => {
        const res = await axios.get(`${API_BASE_URL}/health`);
        return res.data;
    },

    getLyricsModels: async () => {
        const res = await axios.get(`${API_BASE_URL}/models/lyrics`);
        return res.data.models;
    },

    generateJob: async (
        prompt: string,
        durationMs: number,
        lyrics?: string,
        tags?: string,
        cfg_scale: number = 1.5,
        temperature: number = 1.0,
        topk: number = 50,
        llmModel?: string,
        parentJobId?: string,
        seed?: number,
        modelProvider: string = 'minimax_music3',
        voiceProfileId?: string,
        structuredCaption?: Record<string, string>,
        projectId?: string
    ) => {
        const res = await axios.post(`${API_BASE_URL}/generate/music`, {
            prompt,
            duration_ms: durationMs,
            lyrics,
            tags,
            cfg_scale,
            temperature,
            topk,
            llm_model: llmModel,
            parent_job_id: parentJobId,
            seed,
            model_provider: modelProvider,
            voice_profile_id: voiceProfileId,
            structured_caption: structuredCaption,
            project_id: projectId
        });
        return res.data;
    },

    generateLyrics: async (topic: string, modelName: string, currentLyrics?: string, tags?: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/lyrics`, {
            topic,
            model_name: modelName,
            seed_lyrics: currentLyrics,
            tags: tags
        });
        return res.data.lyrics;
    },

    chatLyrics: async (currentLyrics: string, userMessage: string, modelName: string, topic?: string, tags?: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/lyrics-chat`, {
            current_lyrics: currentLyrics,
            user_message: userMessage,
            model_name: modelName,
            topic: topic,
            tags: tags
        });
        return res.data;
    },

    enhancePrompt: async (concept: string, modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/enhance_prompt`, {
            concept,
            model_name: modelName
        });
        return res.data;
    },

    producerCompose: async (prompt: string, modelName?: string) => {
        const res = await axios.post(`${API_BASE_URL}/producer/compose`, {
            prompt,
            model_name: modelName
        });
        return res.data;
    },

    getInspiration: async (modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/evaluate_inspiration`, {
            model_name: modelName
        });
        return res.data;
    },

    getStylePresets: async (modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/styles`, {
            model_name: modelName
        });
        return res.data.styles;
    },

    renameJob: async (jobId: string, title: string) => {
        const res = await axios.patch(`${API_BASE_URL}/jobs/${jobId}`, { title });
        return res.data;
    },

    deleteJob: async (jobId: string) => {
        const res = await axios.delete(`${API_BASE_URL}/jobs/${jobId}`);
        return res.data;
    },

    inpaintTrack: async (jobId: string, startTime: number, endTime: number) => {
        const res = await axios.post(`${API_BASE_URL}/jobs/${jobId}/inpaint`, {
            start_time: startTime,
            end_time: endTime
        });
        return res.data;
    },

    cancelJob: async (jobId: string) => {
        const res = await axios.post(`${API_BASE_URL}/jobs/${jobId}/cancel`);
        return res.data;
    },

    getJobStatus: async (jobId: string) => {
        const res = await axios.get<Job>(`${API_BASE_URL}/jobs/${jobId}`);
        return res.data;
    },

    getHistory: async (limit: number = 50, offset: number = 0, status: string = 'all', search?: string) => {
        const res = await axios.get<Job[]>(`${API_BASE_URL}/history`, {
            params: { limit, offset, status: status === 'all' ? undefined : status, search }
        });
        return res.data;
    },

    getAudioUrl: (path: string) => {
        if (!path) return '';
        if (path.startsWith('http')) return path;
        return `${API_BASE_URL}${path}`;
    },

    getDownloadUrl: (jobId: string) => {
        return `${API_BASE_URL}/download_track/${jobId}`;
    },

    connectToEvents: (onMessage: (event: MessageEvent) => void) => {
        const eventSource = new EventSource(`${API_BASE_URL}/events`);
        eventSource.onmessage = onMessage;
        eventSource.addEventListener("job_update", onMessage);
        eventSource.addEventListener("job_progress", onMessage);
        return eventSource;
    },

    getLLMConfig: async () => {
        const res = await axios.get<LLMConfig>(`${API_BASE_URL}/config/llm`);
        return res.data;
    },

    updateLLMConfig: async (config: LLMConfig) => {
        const res = await axios.post<LLMConfig>(`${API_BASE_URL}/config/llm`, config);
        return res.data;
    },

    fetchModels: async (config: LLMConfig) => {
        const res = await axios.post<{ models: string[] }>(`${API_BASE_URL}/config/fetch-models`, config);
        return res.data.models;
    },

    getTrainingJobs: async () => {
        return trainingApi.listJobs();
    }
};

export const modelsApi = {
    getModelTree: async (): Promise<ModelVariant[]> => {
        const res = await axios.get(`${API_BASE_URL}/models/tree`);
        return res.data.models;
    },
    getCapabilities: async (): Promise<GenerationCapabilities[]> => {
        const res = await axios.get(`${API_BASE_URL}/models/capabilities`);
        return res.data.capabilities;
    },
    getHardwareProfile: async (): Promise<HardwareProfile> => {
        const res = await axios.get(`${API_BASE_URL}/models/hardware`);
        return res.data.hardware;
    },
    checkDependencies: async (modelId: string): Promise<{ missing: boolean; model_id: string; name: string; size_gb: number; message: string }> => {
        const res = await axios.get(`${API_BASE_URL}/models/check/${encodeURIComponent(modelId)}`);
        return res.data;
    },
    setActiveProvider: async (providerId: string) => {
        const res = await axios.post(`${API_BASE_URL}/models/active/${encodeURIComponent(providerId)}`);
        return res.data;
    }
};

export const voiceApi = {
    listProfiles: async (): Promise<VoiceProfile[]> => {
        const res = await axios.get(`${API_BASE_URL}/voice/profiles`);
        return res.data.profiles;
    },
    createProfile: async (data: { name: string; description: string; consent_confirmed: boolean; f0_method?: string; audio_file?: File | null }): Promise<VoiceProfile> => {
        const formData = new FormData();
        formData.append('name', data.name);
        formData.append('description', data.description);
        formData.append('consent_confirmed', String(data.consent_confirmed));
        if (data.f0_method) formData.append('f0_method', data.f0_method);
        if (data.audio_file) formData.append('audio_file', data.audio_file);
        const res = await axios.post(`${API_BASE_URL}/voice/profiles`, formData);
        return res.data.profile;
    },
    deleteProfile: async (profileId: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/voice/profiles/${encodeURIComponent(profileId)}`);
    }
};

export const workspaceApi = {
    uploadAndTranscribe: async (file: File): Promise<{ job_id: string; job: Job }> => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post(`${API_BASE_URL}/transcribe/upload`, formData);
        return res.data;
    },
    getExportUrl: (jobId: string, format: 'midi' | 'musicxml' | 'ableton' | 'lrc' | 'srt') => {
        return `${API_BASE_URL}/transcribe/export/${jobId}/${format}`;
    },
    applyMastering: async (jobId: string, targetLufs: number = -14.0): Promise<{ status: string; audio_path: string; lufs: number }> => {
        const res = await axios.post(`${API_BASE_URL}/mastering/match/${jobId}`, { target_lufs: targetLufs });
        return res.data;
    },
    saveNotes: async (jobId: string, notes: NoteEvent[]): Promise<void> => {
        await axios.post(`${API_BASE_URL}/workspace/${jobId}/notes`, notes);
    }
};

export interface ProviderConfig {
    api_key?: string;
    base_url?: string;
    model?: string;
}

export interface LLMConfig {
    provider?: string;
    openai?: ProviderConfig;
    gemini?: ProviderConfig;
    openrouter?: ProviderConfig;
    lmstudio?: ProviderConfig;
    ollama?: ProviderConfig;
    deepseek?: ProviderConfig;
    opencode?: ProviderConfig;
    omlx?: ProviderConfig;
}

export interface Style {
    name: string;
    type: 'official' | 'custom' | 'trained';
    description?: string;
    checkpoint_id?: string;
}

export interface PathsConfig {
    model_directory?: string;
    checkpoints_directory?: string;
    datasets_directory?: string;
}

export interface Dataset {
    id: string;
    name: string;
    styles: string[];
    audio_files: { filename: string; caption: string; preprocessed: boolean }[];
    status: string;
    created_at: string;
}

export interface TrainingJob {
    id: string;
    dataset_id: string;
    dataset_name?: string;
    config: {
        method: string;
        epochs: number;
        learning_rate: number;
        lora_rank: number;
    };
    status: string;
    progress: number;
    current_epoch: number;
    current_loss?: number;
    initial_loss?: number;
    final_loss?: number;
    total_epochs: number;
    checkpoint_id?: string;
    error?: string;
    message?: string;
    started_at?: string;
    completed_at?: string;
    created_at?: string;
}

export interface Checkpoint {
    id: string;
    name: string;
    styles: string[];
    method: string;
    created_at: string;
    size_bytes: number;
    is_active: boolean;
}

export const styleApi = {
    getStyles: async (): Promise<Style[]> => {
        const res = await axios.get(`${API_BASE_URL}/styles`);
        return res.data.styles;
    },
    addCustomStyle: async (name: string, description?: string): Promise<Style> => {
        const res = await axios.post(`${API_BASE_URL}/styles/custom`, { name, description });
        return res.data.style;
    },
    removeCustomStyle: async (name: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/styles/custom/${encodeURIComponent(name)}`);
    }
};

export const pathsApi = {
    getConfig: async (): Promise<PathsConfig> => {
        const res = await axios.get(`${API_BASE_URL}/config/paths`);
        return res.data;
    },
    updateConfig: async (paths: PathsConfig): Promise<PathsConfig> => {
        const res = await axios.post(`${API_BASE_URL}/config/paths`, paths);
        return res.data;
    },
    validate: async (paths: PathsConfig): Promise<Record<string, { valid: boolean; path: string }>> => {
        const res = await axios.post(`${API_BASE_URL}/config/paths/validate`, paths);
        return res.data;
    }
};

export const trainingApi = {
    createDataset: async (name: string, styles: string[]): Promise<Dataset> => {
        const res = await axios.post(`${API_BASE_URL}/training/datasets`, { name, styles });
        return res.data.dataset;
    },
    listDatasets: async (): Promise<Dataset[]> => {
        const res = await axios.get(`${API_BASE_URL}/training/datasets`);
        return res.data.datasets;
    },
    getDataset: async (id: string): Promise<Dataset> => {
        const res = await axios.get(`${API_BASE_URL}/training/datasets/${id}`);
        return res.data.dataset;
    },
    uploadAudio: async (datasetId: string, file: File, caption: string): Promise<void> => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('caption', caption);
        await axios.post(`${API_BASE_URL}/training/datasets/${datasetId}/audio`, formData);
    },
    deleteAudio: async (datasetId: string, filename: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/training/datasets/${datasetId}/audio/${encodeURIComponent(filename)}`);
    },
    updateAudioCaption: async (datasetId: string, filename: string, caption: string): Promise<void> => {
        await axios.put(`${API_BASE_URL}/training/datasets/${datasetId}/audio/${encodeURIComponent(filename)}`, { caption });
    },
    validateDataset: async (datasetId: string): Promise<{ valid: boolean; file_count: number; minimum_required: number }> => {
        const res = await axios.get(`${API_BASE_URL}/training/datasets/${datasetId}/validate`);
        return res.data;
    },
    updateDataset: async (datasetId: string, name: string, styles: string[]): Promise<Dataset> => {
        const res = await axios.put(`${API_BASE_URL}/training/datasets/${datasetId}`, { name, styles });
        return res.data.dataset;
    },
    deleteDataset: async (datasetId: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/training/datasets/${datasetId}`);
    },
    preprocessDataset: async (datasetId: string, force: boolean = true): Promise<{ success: boolean; processed_count?: number; message?: string }> => {
        const res = await axios.post(`${API_BASE_URL}/training/datasets/${datasetId}/preprocess`, { force });
        return res.data;
    },
    startJob: async (config: { dataset_id: string; method: string; epochs: number; learning_rate: number; lora_rank: number }): Promise<TrainingJob> => {
        const res = await axios.post(`${API_BASE_URL}/training/jobs`, config);
        return res.data.job;
    },
    cancelJob: async (jobId: string): Promise<void> => {
        await axios.post(`${API_BASE_URL}/training/jobs/${jobId}/cancel`);
    },
    listJobs: async (): Promise<TrainingJob[]> => {
        const res = await axios.get(`${API_BASE_URL}/training/jobs`);
        return res.data.jobs;
    },
    getJob: async (id: string): Promise<TrainingJob> => {
        const res = await axios.get(`${API_BASE_URL}/training/jobs/${id}`);
        return res.data.job;
    },
    getJobLogs: async (id: string, offset: number = 0): Promise<{ logs: string[]; offset: number }> => {
        const res = await axios.get(`${API_BASE_URL}/training/jobs/${id}/logs`, { params: { offset } });
        return res.data;
    },
    deleteJob: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/training/jobs/${id}`);
    },
    listCheckpoints: async (): Promise<Checkpoint[]> => {
        const res = await axios.get(`${API_BASE_URL}/training/checkpoints`);
        return res.data.checkpoints;
    },
    activateCheckpoint: async (id: string): Promise<void> => {
        await axios.post(`${API_BASE_URL}/training/checkpoints/${id}/activate`);
    },
    deactivateCheckpoint: async (): Promise<void> => {
        await axios.post(`${API_BASE_URL}/training/checkpoints/deactivate`);
    },
    deleteCheckpoint: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/training/checkpoints/${id}`);
    }
};

export const projectApi = {
    listProjects: async (): Promise<Project[]> => {
        const res = await axios.get(`${API_BASE_URL}/projects`);
        return res.data;
    },
    getProject: async (id: string): Promise<Project> => {
        const res = await axios.get(`${API_BASE_URL}/projects/${id}`);
        return res.data;
    },
    createProject: async (data: ProjectCreate): Promise<Project> => {
        const res = await axios.post(`${API_BASE_URL}/projects`, data);
        return res.data;
    },
    updateProject: async (id: string, data: ProjectUpdate): Promise<Project> => {
        const res = await axios.put(`${API_BASE_URL}/projects/${id}`, data);
        return res.data;
    },
    deleteProject: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/projects/${id}`);
    },
    addTrackToProject: async (projectId: string, jobId: string): Promise<void> => {
        await axios.post(`${API_BASE_URL}/projects/${projectId}/tracks`, { job_id: jobId });
    },
    removeTrackFromProject: async (projectId: string, jobId: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/projects/${projectId}/tracks/${jobId}`);
    }
};
