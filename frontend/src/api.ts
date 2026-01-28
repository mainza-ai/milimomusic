import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface Job {
    id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    title?: string;
    prompt: string;
    lyrics?: string;
    tags?: string; // Added field
    audio_path?: string;
    error_msg?: string;
    created_at: string;
    duration_ms?: number;
    seed?: number; // Added field
    is_favorite?: boolean;
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
        seed?: number
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
            seed
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
        return res.data; // { message, lyrics }
    },

    enhancePrompt: async (concept: string, modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/enhance_prompt`, {
            concept,
            model_name: modelName
        });
        return res.data; // { topic, tags }
    },

    getInspiration: async (modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/evaluate_inspiration`, {
            model_name: modelName
        });
        return res.data; // { topic, tags }
    },

    getStylePresets: async (modelName: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/styles`, {
            model_name: modelName
        });
        return res.data.styles; // string[]
    },

    generateMusic: async (
        tags: string,
        lyrics: string,
        durationMs: number = 240000,
        temperature: number = 1.0,
        cfgScale: number = 1.5,
        topk: number = 50,
        prompt: string,
        llmModel: string = "llama3.2:3b-instruct-fp16"
    ) => {
        const res = await axios.post(`${API_BASE_URL}/generate/music`, {
            lyrics,
            tags,
            duration_ms: durationMs,
            temperature,
            cfg_scale: cfgScale,
            topk,
            prompt,
            llm_model: llmModel
        });
        return res.data; // { job_id, status }
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
        return `${API_BASE_URL}${path}`;
    },

    getDownloadUrl: (jobId: string) => {
        return `${API_BASE_URL}/download_track/${jobId}`;
    },

    connectToEvents: (onMessage: (event: MessageEvent) => void) => {
        const eventSource = new EventSource(`${API_BASE_URL}/events`);
        eventSource.onmessage = onMessage;

        // Custom event listeners
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
}

// Style Management Types
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
    dataset_name?: string;  // Persists even if dataset is deleted
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
    initial_loss?: number;   // First loss value at start of training
    final_loss?: number;     // Final loss when training completes
    total_epochs: number;
    checkpoint_id?: string;
    error?: string;
    message?: string;
    started_at?: string;     // ISO timestamp when training started
    completed_at?: string;   // ISO timestamp when training finished
    created_at?: string;     // ISO timestamp when job was created
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

// Extended API
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
    // Datasets
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

    // Jobs
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

    // Checkpoints
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
