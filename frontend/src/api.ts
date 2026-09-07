import axios from 'axios';

// Phase-1 auth loop closure: attach optional bearer token to every request.
axios.interceptors.request.use((config) => {
    const token = localStorage.getItem('milimo_auth_token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
});

// API base is configurable via Vite env; defaults to local backend for development.
const API_BASE_URL: string = (import.meta as any).env?.VITE_API_URL ?? (
    (import.meta as any).env?.DEV ? 'http://localhost:8000' : ''
);

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
    [key: string]: any;
    vocals?: string;
    drums?: string;
    bass?: string;
    guitar?: string;
    piano?: string;
    other?: string;
    instrumental?: string;
    /** Dynamic per-instrument stems keyed by instrument name (from transcription). */
    instrumental_parts?: Record<string, string>;
    /** General MIDI program (instrument) number per instrument name. */
    instrument_programs?: Record<string, number>;
    stems_source?: string;
    sources_available?: string[];
    default_source?: string;
}

export interface StemMeta {
    label: string;
    icon: string;
    color: string;
    gradient: string;
}

export function getStemMeta(stemKey: string): StemMeta {
    const k = stemKey.toLowerCase();
    if (k.includes('vocal') || k === 'lead_vocals' || k === 'backing_vocals') {
        return { label: 'Vocals', icon: '🎤', color: '#0d9488', gradient: 'from-teal-500 to-cyan-500' };
    }
    if (k.includes('drum') || k === 'percussion' || k.includes('beat')) {
        return { label: 'Drums', icon: '🥁', color: '#ea580c', gradient: 'from-amber-500 to-orange-500' };
    }
    if (k.includes('bass') || k === 'synth_bass') {
        return { label: 'Bass', icon: '🎸', color: '#0284c7', gradient: 'from-sky-500 to-blue-500' };
    }
    if (k.includes('guitar') || k === 'acoustic_guitar' || k === 'electric_guitar') {
        return { label: 'Guitar', icon: '🎸', color: '#d97706', gradient: 'from-amber-600 to-yellow-500' };
    }
    if (k.includes('piano') || k === 'keys' || k === 'keyboard') {
        return { label: 'Piano & Keys', icon: '🎹', color: '#6366f1', gradient: 'from-violet-500 to-purple-500' };
    }
    if (k.includes('wind') || k.includes('flute') || k.includes('sax') || k.includes('brass') || k.includes('horn')) {
        return { label: 'Winds & Brass', icon: '🎷', color: '#10b981', gradient: 'from-emerald-500 to-teal-500' };
    }
    if (k.includes('string') || k.includes('violin') || k.includes('cello')) {
        return { label: 'Strings', icon: '🎻', color: '#ec4899', gradient: 'from-rose-500 to-pink-500' };
    }
    return {
        label: stemKey.charAt(0).toUpperCase() + stemKey.slice(1).replace(/_/g, ' '),
        icon: '🎛️',
        color: '#64748b',
        gradient: 'from-teal-500 to-emerald-500'
    };
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
    is_section?: boolean;
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

    // Visual Artwork Assets
    cover_image_path?: string;
    image_prompt?: string;
    video_path?: string;

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
    used_fallback_synth?: boolean;
    fallback_reason?: string | null;
    voice_profile_id?: string;
    project_id?: string;
    session_id?: string;
    mastered_path?: string;
}

export interface Project {
    id: string;
    name: string;
    description?: string;
    cover_image_path?: string;
    image_prompt?: string;
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
    cover_image_path?: string;
    image_prompt?: string;
    tags?: string;
    bpm?: number;
    key_signature?: string;
    color?: string;
    icon?: string;
}

export interface ProjectUpdate {
    name?: string;
    description?: string;
    cover_image_path?: string;
    image_prompt?: string;
    tags?: string;
    bpm?: number;
    key_signature?: string;
    color?: string;
    icon?: string;
}

export interface StudioSession {
    id: string;
    title: string;
    project_id?: string;
    active_job_id?: string;
    created_at: string;
    updated_at: string;
    message_count?: number;
    job_count?: number;
    jobs?: Job[];
    messages?: SessionMessage[];
}

export interface SessionMessage {
    id: string;
    session_id: string;
    role: 'user' | 'producer' | 'system';
    content: string;
    audio_attachment_path?: string;
    generated_job_id?: string;
    preset_data_json?: string;
    created_at: string;
}

export interface SessionCreate {
    title?: string;
    project_id?: string;
    active_job_id?: string;
}

export interface SessionUpdate {
    title?: string;
    project_id?: string;
    active_job_id?: string;
}

export interface SessionMessageCreate {
    content: string;
    role?: 'user' | 'producer';
    audio_attachment_path?: string;
    generated_job_id?: string;
    preset_data_json?: string;
}

export interface ModelDownloadStatus {
    id: string;
    repo_id: string;
    status: 'queued' | 'downloading' | 'completed' | 'cancelled' | 'error';
    total_files: number;
    files_done: number;
    current_file: string;
    received_bytes: number;
    total_bytes: number;
    progress_percent: number | null;
    local_dir: string;
    error: string;
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
    category?: 'audio' | 'image' | 'video';
    repo_id?: string;
    is_default: boolean;
    is_active?: boolean;
    is_custom?: boolean;
}

export interface HuggingFaceSearchResult {
    repo_id: string;
    name: string;
    author: string;
    downloads: number;
    likes: number;
    pipeline_tag: string;
    category: string;
    is_installed: boolean;
    last_modified?: string;
    size_bytes?: number;
    size_gb?: number;
    size_formatted?: string;
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
    acoustic_features?: {
        median_f0_hz?: number;
        spectral_centroid_hz?: number;
        spectral_rolloff_hz?: number;
        mean_rms?: number;
        duration_sec?: number;
        timbre_profile?: string;
    };
    dataset_files?: string[];
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
        projectId?: string,
        title?: string,
        isInstrumental?: boolean,
        coverImagePath?: string,
        imagePrompt?: string,
        sessionId?: string
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
            project_id: projectId,
            title,
            is_instrumental: isInstrumental,
            cover_image_path: coverImagePath,
            image_prompt: imagePrompt,
            session_id: sessionId
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

    rewriteCaption: async (concept: string, lyrics: string | undefined, tags: string | undefined, modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/rewrite_caption`, {
            concept,
            lyrics: lyrics || null,
            tags: tags || null,
            model_name: modelName
        });
        return res.data;
    },

    producerCompose: async (prompt: string, modelName?: string, signal?: AbortSignal) => {
        const res = await axios.post(`${API_BASE_URL}/producer/compose`, {
            prompt,
            model_name: modelName
        }, { signal });
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

    connectToEvents: (onMessage: (event: MessageEvent) => void, extraEventTypes: string[] = []) => {
        const token = localStorage.getItem('milimo_auth_token') || '';
        const url = token ? `${API_BASE_URL}/events?auth=${encodeURIComponent(token)}` : `${API_BASE_URL}/events`;
        const eventSource = new EventSource(url);
        eventSource.onmessage = onMessage;
        [...new Set(["job_update", "job_progress", ...extraEventTypes])]
            .forEach(t => eventSource.addEventListener(t, onMessage));
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
    startModelDownload: async (repoId: string): Promise<ModelDownloadStatus> => {
        const res = await axios.post(`${API_BASE_URL}/models/download`, { repo_id: repoId });
        return res.data;
    },
    getModelDownload: async (downloadId: string): Promise<ModelDownloadStatus> => {
        const res = await axios.get(`${API_BASE_URL}/models/downloads/${downloadId}`);
        return res.data;
    },
    cancelModelDownload: async (downloadId: string): Promise<void> => {
        await axios.post(`${API_BASE_URL}/models/downloads/${downloadId}/cancel`);
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
    },
    getActiveModel: async (): Promise<{ active_provider: string; active_model: ModelVariant }> => {
        const res = await axios.get(`${API_BASE_URL}/models/active`);
        return res.data;
    },
    selectActiveModel: async (modelId: string): Promise<{ status: string; active_model: ModelVariant }> => {
        const res = await axios.post(`${API_BASE_URL}/models/select`, { model_id: modelId });
        return res.data;
    },
    checkAutoInstall: async (): Promise<{ needs_download: boolean; recommended_repo_id: string | null }> => {
        const res = await axios.get(`${API_BASE_URL}/models/auto-install-check`);
        return res.data;
    },
    searchHuggingFace: async (query: string, pipeline?: string, limit: number = 20): Promise<{ query: string; count: number; models: HuggingFaceSearchResult[] }> => {
        const res = await axios.get(`${API_BASE_URL}/models/search`, {
            params: { q: query, pipeline: pipeline || undefined, limit }
        });
        return res.data;
    },
    deleteCustomModel: async (modelId: string): Promise<{ status: string; model_id: string }> => {
        const res = await axios.delete(`${API_BASE_URL}/models/custom/${encodeURIComponent(modelId)}`);
        return res.data;
    },
    updateCustomModel: async (modelId: string, updates: { category?: string; name?: string }): Promise<any> => {
        const res = await axios.patch(`${API_BASE_URL}/models/custom/${encodeURIComponent(modelId)}`, updates);
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
    has_key?: boolean;
    has_api_key?: boolean;
    base_url?: string;
    model?: string;
}

export interface LLMConfig {
    provider?: string;
    nvidia?: ProviderConfig;
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
    uploadAudio: async (datasetId: string, file: File, caption: string = ''): Promise<{ filename: string; caption: string }> => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('caption', caption);
        const res = await axios.post(`${API_BASE_URL}/training/datasets/${datasetId}/audio`, formData);
        return res.data.audio_file;
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
    duplicateProject: async (id: string): Promise<Project> => {
        const res = await axios.post(`${API_BASE_URL}/projects/${id}/duplicate`);
        return res.data;
    },
    exportProjectPackUrl: (id: string): string => {
        return `${API_BASE_URL}/projects/${id}/export`;
    },
    addTrackToProject: async (projectId: string, jobId: string): Promise<void> => {
        await axios.post(`${API_BASE_URL}/projects/${projectId}/tracks`, { job_id: jobId });
    },
    removeTrackFromProject: async (projectId: string, jobId: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/projects/${projectId}/tracks/${jobId}`);
    }
};

export const sessionApi = {
    listSessions: async (): Promise<StudioSession[]> => {
        const res = await axios.get(`${API_BASE_URL}/sessions`);
        return res.data;
    },
    createSession: async (data: SessionCreate = {}): Promise<StudioSession> => {
        const res = await axios.post(`${API_BASE_URL}/sessions`, data);
        return res.data;
    },
    getSession: async (id: string): Promise<StudioSession> => {
        const res = await axios.get(`${API_BASE_URL}/sessions/${id}`);
        return res.data;
    },
    updateSession: async (id: string, data: SessionUpdate): Promise<StudioSession> => {
        const res = await axios.patch(`${API_BASE_URL}/sessions/${id}`, data);
        return res.data;
    },
    deleteSession: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/sessions/${id}`);
    },
    sendChatMessage: async (id: string, message: SessionMessageCreate, signal?: AbortSignal): Promise<{
        session: StudioSession;
        user_message: SessionMessage;
        producer_message: SessionMessage;
        preset: any;
    }> => {
        const res = await axios.post(`${API_BASE_URL}/sessions/${id}/chat`, message, { signal });
        return res.data;
    }
};

export const coverApi = {
    uploadCoverImage: async (file: File): Promise<{ url: string; filename: string }> => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post(`${API_BASE_URL}/upload/image`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data;
    },
    generateCoverPrompt: async (params: { title?: string; description?: string; tags?: string; genre?: string }): Promise<{ prompt: string }> => {
        const res = await axios.post(`${API_BASE_URL}/generate/cover-prompt`, params);
        return res.data;
    },
    generateCoverImage: async (params: { prompt: string; style?: string; model_id?: string }): Promise<{ url: string; prompt: string }> => {
        const res = await axios.post(`${API_BASE_URL}/generate/cover-image`, params);
        return res.data;
    }
};

export interface SheetScoreItem {
    name: string;
    filename: string;
    type: 'musicxml' | 'pdf';
    url: string;
}

export const trackApi = {
    updateTrackMetadata: async (jobId: string, updates: Partial<Job>): Promise<Job> => {
        const res = await axios.patch(`${API_BASE_URL}/jobs/${jobId}`, updates);
        return res.data;
    },
    getStudioPackUrl: (jobId: string): string => {
        return `${API_BASE_URL}/jobs/${jobId}/studio-pack`;
    },
    voiceConvertTrack: async (
        jobId: string,
        voiceProfileId: string,
        options?: { pitch_shift?: number; dry_wet?: number; formant_preserve?: boolean }
    ): Promise<Job> => {
        const res = await axios.post(`${API_BASE_URL}/jobs/${jobId}/voice-convert`, {
            voice_profile_id: voiceProfileId,
            pitch_shift: options?.pitch_shift ?? 0,
            dry_wet: options?.dry_wet !== undefined ? options.dry_wet / 100 : 1.0,
            formant_preserve: options?.formant_preserve ?? true
        });
        return res.data;
    },
    getSheets: async (jobId: string): Promise<{ job_id: string; sheets: SheetScoreItem[] }> => {
        const res = await axios.get(`${API_BASE_URL}/tracks/${jobId}/sheets`);
        return res.data;
    },
    getTrackPeaks: async (jobId: string, buckets: number = 240): Promise<{ job_id: string; buckets: number; duration: number; peaks: number[] }> => {
        const res = await axios.get(`${API_BASE_URL}/tracks/${jobId}/peaks`, { params: { buckets } });
        return res.data;
    },
    updateMidiNotes: async (jobId: string, notes: NoteEvent[]): Promise<{ status: string; job: Job }> => {
        const res = await axios.post(`${API_BASE_URL}/tracks/${jobId}/midi`, notes);
        return res.data;
    },
    getLrcUrl: (jobId: string): string => {
        return `${API_BASE_URL}/tracks/${jobId}/lrc`;
    },
    realignLyrics: async (jobId: string, lyrics?: string): Promise<{ status: string; timed_lyrics: any[]; job: Job }> => {
        const res = await axios.post(`${API_BASE_URL}/tracks/${jobId}/realign_lyrics`, { lyrics });
        return res.data;
    }
};

// ── AI Agents & Artist Profiles (agent foundation surface) ──────────────────
export interface AgentInfo {
    name: string;
    display_name: string;
    description: string;
    input_schema: Record<string, unknown>;
}

export interface ArtistProfileT {
    id: string;
    project_id: string | null;
    name: string;
    bio: string;
    lore_json: string;
    tags: string;
    cover_image_path: string | null;
    default_provider: string | null;
    default_model: string | null;
    voice_profile_id: string | null;
    created_at: string;
    updated_at: string;
}

export interface AgentAssignmentT {
    id: string;
    profile_id: string;
    role: string;
    agent_name: string;
    model_provider: string | null;
    model: string | null;
    config_json: string;
}

export interface ReleaseT {
    id: string;
    profile_id: string;
    title: string;
    description: string;
    status: string;
    vision_json: string;
    track_order_json: string;
    cover_image_path: string | null;
    created_at: string;
}

export interface ProfileDetail {
    profile: ArtistProfileT;
    assignments: AgentAssignmentT[];
    releases: ReleaseT[];
}

export interface ExperiencerSeed {
    working_title: string;
    mood: string;
    story_seed: string;
    suggested_style_tags: string[];
    energy: number;
    placement_hint: string;
}

export interface ExperiencerVision {
    journey_title: string;
    concept_statement: string;
    life_journey_narrative: string;
    emotional_arc: { position: number; label: string; intensity: number; description?: string }[];
    song_seeds: ExperiencerSeed[];
    recurring_motifs: string[];
    listener_experience_notes: string;
}

export interface AgentRunEnvelope {
    run: {
        id: string;
        agent_name: string;
        status: string;
        tokens_in: number;
        tokens_out: number;
        latency_ms: number;
        attempts_json: string;
        output_json: string;
    };
    result: unknown;
}

export const agentsApi = {
    listAgents: async (): Promise<AgentInfo[]> => {
        const res = await axios.get(`${API_BASE_URL}/agents`);
        return res.data.agents;
    },
    runAgent: async (name: string, body: { input: Record<string, unknown>; session_id?: string; project_id?: string; profile_id?: string }): Promise<AgentRunEnvelope> => {
        const res = await axios.post(`${API_BASE_URL}/agents/${name}/run`, body);
        return res.data;
    },
    listRuns: async (profileId?: string, limit = 50): Promise<{ runs: AgentRunRow[]; total: number }> => {
        const res = await axios.get(`${API_BASE_URL}/agents/runs`, {
            params: { ...(profileId ? { profile_id: profileId } : {}), limit },
        });
        return res.data;
    },
    runStats: async (profileId?: string): Promise<RunStats> => {
        const res = await axios.get(`${API_BASE_URL}/agents/runs/stats`, {
            params: { ...(profileId ? { profile_id: profileId } : {}) },
        });
        return res.data;
    },
};

export interface TrackReview { verdict: 'pass' | 'revise' | 'concern' | 'unavailable'; score?: number | null; notes?: string; contradictions?: string[]; }
export interface ReleaseTrackT {
    id: string; title: string | null; status: string; duration_ms: number;
    seed: number | null; seed_slot?: number | null; artifacts: Record<string, string | null>;
    used_real_inference: boolean; review?: TrackReview | null; created_at: string;
}
// `status` = release lifecycle (planned|in_progress|completed); `rollup` = track completion rollup (completed|partial|pending)
export interface ReleaseTracksT { release_id: string; title: string; tracks: ReleaseTrackT[]; succeeded: number; total: number; status: string; rollup: string; }
export interface AgentRunRow {
    id: string;
    agent_name: string;
    status: string;
    progress?: number;
    error_message?: string;
    state_json?: string;
    input_json?: string;
    created_at?: string;
    latency_ms?: number;
    tokens_in?: number;
    tokens_out?: number;
}

export interface ProfileStats { crew_count: number; release_count: number; last_activity: string | null; }

export interface RunStats {
    total: number;
    statuses: Record<string, number>;
    success_rate: number | null;
    latency_ms: { p50: number | null; p95: number | null };
    tokens_in: number;
    tokens_out: number;
    by_agent: Record<string, { count: number; succeeded: number; failed: number; tokens_out: number }>;
}

export const albumApi = {
    produce: async (releaseId: string, autopilot: boolean, opts?: { budget?: { deadline_s?: number }; crew?: { stylist?: boolean; critic?: boolean } }): Promise<{ run_id: string; status: string; autopilot: boolean }> => {
        const res = await axios.post(`${API_BASE_URL}/releases/${releaseId}/produce`, {
            autopilot,
            ...(opts?.budget ? { budget: opts.budget } : {}),
            ...(opts?.crew ? { crew: opts.crew } : {}),
        });
        return res.data;
    },
    resume: async (runId: string, autopilot = false): Promise<{ run_id: string; status: string }> => {
        const res = await axios.post(`${API_BASE_URL}/agents/runs/${runId}/resume`, { autopilot });
        return res.data;
    },
    cancelRun: async (runId: string): Promise<{ id: string; status: string }> => {
        const res = await axios.post(`${API_BASE_URL}/agents/runs/${runId}/cancel`, {});
        return res.data;
    },
    getRun: async (runId: string): Promise<AgentRunRow> => {
        const res = await axios.get(`${API_BASE_URL}/agents/runs/${runId}`);
        return res.data.run;
    },
};

export const profilesApi = {
    list: async (opts?: { projectId?: string; withStats?: boolean; limit?: number; offset?: number; q?: string }): Promise<{ profiles: ArtistProfileT[]; total: number; stats?: Record<string, ProfileStats> }> => {
        const params: Record<string, unknown> = {};
        if (opts?.projectId) params.project_id = opts.projectId;
        if (opts?.withStats) params.with_stats = 1;
        if (opts?.limit) params.limit = opts.limit;
        if (opts?.offset) params.offset = opts.offset;
        if (opts?.q) params.q = opts.q;
        const res = await axios.get(`${API_BASE_URL}/profiles`, { params });
        return res.data;
    },
    create: async (body: { name: string; bio?: string; tags?: string; project_id?: string }): Promise<ArtistProfileT> => {
        const res = await axios.post(`${API_BASE_URL}/profiles`, body);
        return res.data;
    },
    get: async (id: string): Promise<ProfileDetail> => {
        const res = await axios.get(`${API_BASE_URL}/profiles/${id}`);
        return res.data;
    },
    update: async (id: string, body: Partial<Pick<ArtistProfileT, 'name' | 'bio' | 'tags' | 'lore_json' | 'voice_profile_id'>>): Promise<ArtistProfileT> => {
        const res = await axios.patch(`${API_BASE_URL}/profiles/${id}`, body);
        return res.data;
    },
    generateLore: async (id: string): Promise<{ run: string; lore: Record<string, unknown>; profile: ArtistProfileT }> => {
        const res = await axios.post(`${API_BASE_URL}/profiles/${id}/lore/generate`);
        return res.data;
    },
    delete: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/profiles/${id}`);
    },
    getReleaseTracks: async (releaseId: string): Promise<ReleaseTracksT> => {
        const res = await axios.get(`${API_BASE_URL}/releases/${releaseId}/tracks`);
        return res.data;
    },
    setCover: async (id: string, coverImagePath: string): Promise<ArtistProfileT> => {
        const res = await axios.patch(`${API_BASE_URL}/profiles/${id}/cover`, { cover_image_path: coverImagePath });
        return res.data;
    },
    setAssignments: async (id: string, assignments: { role: string; agent_name: string; model_provider?: string; model?: string }[]): Promise<AgentAssignmentT[]> => {
        const res = await axios.put(`${API_BASE_URL}/profiles/${id}/assignments`, { assignments });
        return res.data.assignments;
    },
    createRelease: async (body: { profile_id: string; title: string; description?: string; vision?: Record<string, unknown>; }): Promise<ReleaseT> => {
        const res = await axios.post(`${API_BASE_URL}/releases`, body);
        return res.data;
    },
};

export const releaseApi = {
    list: async (profileId: string, limit = 100): Promise<{ releases: ReleaseT[]; total: number }> => {
        const res = await axios.get(`${API_BASE_URL}/profiles/${profileId}/releases`, { params: { limit } });
        return res.data;
    },
    update: async (id: string, body: { title?: string; description?: string; status?: string; cover_image_path?: string }): Promise<ReleaseT> => {
        const res = await axios.patch(`${API_BASE_URL}/releases/${id}`, body);
        return res.data;
    },
    delete: async (id: string): Promise<{ status: string; jobs_detached: number }> => {
        const res = await axios.delete(`${API_BASE_URL}/releases/${id}`);
        return res.data;
    },
    setTrackOrder: async (releaseId: string, jobIds: string[]): Promise<{ status: string; track_order: string[] }> => {
        const res = await axios.patch(`${API_BASE_URL}/releases/${releaseId}/track-order`, { job_ids: jobIds });
        return res.data;
    },    retryTrack: async (releaseId: string, jobId: string): Promise<{ run_id: string; status: string; seed_slot: number }> => {
        const res = await axios.post(`${API_BASE_URL}/releases/${releaseId}/tracks/${jobId}/retry`);
        return res.data;
    },
    detachTrack: async (releaseId: string, jobId: string): Promise<{ status: string; message: string }> => {
        const res = await axios.delete(`${API_BASE_URL}/releases/${releaseId}/tracks/${jobId}`);
        return res.data;
    },
    attachTrack: async (releaseId: string, jobId: string): Promise<{ status: string; message: string }> => {
        const res = await axios.post(`${API_BASE_URL}/releases/${releaseId}/tracks`, { job_id: jobId });
        return res.data;
    },
};

export interface DbPlaylist {
    id: string;
    name: string;
    description: string;
    cover_color: string;
    created_at: string;
    updated_at: string;
    track_count: number;
    song_ids: string[];
    tracks?: Job[];
}

export interface PlaylistCreateInput {
    name: string;
    description?: string;
    cover_color?: string;
    song_ids?: string[];
}

export interface PlaylistUpdateInput {
    name?: string;
    description?: string;
    cover_color?: string;
}

export interface StudioProfile {
    id: string;
    artist_name: string;
    bio: string;
    email?: string;
    avatar_url?: string;
    social_links?: Record<string, string>;
    preferences?: Record<string, any>;
    created_at?: string;
    updated_at?: string;
}

export interface StudioProfileUpdateInput {
    artist_name?: string;
    bio?: string;
    email?: string;
    avatar_url?: string;
    social_links?: Record<string, string>;
    preferences?: Record<string, any>;
}

export const playlistApi = {
    list: async (): Promise<DbPlaylist[]> => {
        const res = await axios.get(`${API_BASE_URL}/playlists`);
        return res.data;
    },
    get: async (id: string): Promise<DbPlaylist> => {
        const res = await axios.get(`${API_BASE_URL}/playlists/${id}`);
        return res.data;
    },
    create: async (data: PlaylistCreateInput): Promise<DbPlaylist> => {
        const res = await axios.post(`${API_BASE_URL}/playlists`, data);
        return res.data;
    },
    update: async (id: string, data: PlaylistUpdateInput): Promise<DbPlaylist> => {
        const res = await axios.put(`${API_BASE_URL}/playlists/${id}`, data);
        return res.data;
    },
    delete: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/playlists/${id}`);
    },
    addTrack: async (playlistId: string, jobId: string): Promise<void> => {
        await axios.post(`${API_BASE_URL}/playlists/${playlistId}/tracks`, { job_id: jobId });
    },
    removeTrack: async (playlistId: string, jobId: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/playlists/${playlistId}/tracks/${jobId}`);
    }
};

export const studioProfileApi = {
    get: async (): Promise<StudioProfile> => {
        const res = await axios.get(`${API_BASE_URL}/profile/studio`);
        return res.data;
    },
    update: async (data: StudioProfileUpdateInput): Promise<StudioProfile> => {
        const res = await axios.put(`${API_BASE_URL}/profile/studio`, data);
        return res.data;
    }
};

export interface StoryboardScene {
    time: string;
    prompt: string;
    camera: string;
    lighting?: string;
}

export interface VideoClipSegment {
    clip_index: number;
    start_time: number;
    end_time: number;
    duration: number;
    time_str: string;
    is_vocal: boolean;
    scene_type: 'VOCAL_PERFORMANCE' | 'CINEMATIC_BROLL';
    lyrics: string;
    prompt: string;
    camera: string;
    lighting?: string;
}

export interface VideoPlanResult {
    status: string;
    job_id: string;
    total_clips: number;
    vocal_clips_count: number;
    broll_clips_count: number;
    max_clip_duration: number;
    model_max_duration?: number;
    model_name: string;
    clips: VideoClipSegment[];
}

export interface VideoTaskStatus {
    id: string;
    job_id: string;
    status: 'processing' | 'completed' | 'error';
    step: string;
    progress: number;
    total_clips: number;
    current_clip: number;
    video_url?: string | null;
    error?: string | null;
}

export interface VideoPlanParams {
    max_clip_duration?: number;
    bpm?: number;
    visual_style?: string;
    model_name?: string;
}

export interface VideoRenderParams {
    model_name?: string;
    visual_style?: string;
    resolution?: '720p' | '1080p';
    aspect_ratio?: '16:9' | '9:16';
    enable_lip_sync?: boolean;
    burn_lyrics?: boolean;
    subtitle_style?: string;
    max_clip_duration?: number;
    mode?: 'production_multiclip' | 'fast_preview';
    face_image_path?: string | null;
}

export const videoApi = {
    generateStoryboard: async (jobId: string, visualStyle: string = 'neon-cyberpunk'): Promise<StoryboardScene[]> => {
        const res = await axios.post(`${API_BASE_URL}/videos/storyboard/${jobId}`, { visual_style: visualStyle });
        return res.data.scenes;
    },
    planVideo: async (jobId: string, params: VideoPlanParams = {}): Promise<VideoPlanResult> => {
        const res = await axios.post(`${API_BASE_URL}/videos/plan/${jobId}`, params);
        return res.data;
    },
    renderAdvancedVideo: async (jobId: string, params: VideoRenderParams = {}): Promise<{ status: string; task_id: string; job_id: string }> => {
        const res = await axios.post(`${API_BASE_URL}/videos/render-advanced/${jobId}`, params);
        return res.data;
    },
    getVideoTaskStatus: async (taskId: string): Promise<VideoTaskStatus> => {
        const res = await axios.get(`${API_BASE_URL}/videos/tasks/${taskId}`);
        return res.data;
    },
    renderVideo: async (jobId: string, visualStyle: string = 'neon-cyberpunk', resolution: string = '720p'): Promise<{ status: string; video_url: string }> => {
        const res = await axios.post(`${API_BASE_URL}/videos/render/${jobId}`, { visual_style: visualStyle, resolution });
        return res.data;
    },
    getVideo: async (jobId: string): Promise<{ video_path: string | null; has_video: boolean }> => {
        const res = await axios.get(`${API_BASE_URL}/videos/${jobId}`);
        return res.data;
    }
};


