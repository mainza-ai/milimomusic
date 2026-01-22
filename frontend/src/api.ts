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
}

export const api = {
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
        parentJobId?: string,
        seed?: number // Added field
    ) => {
        const res = await axios.post(`${API_BASE_URL}/generate/music`, {
            prompt,
            duration_ms: durationMs,
            lyrics,
            tags,
            cfg_scale,
            parent_job_id: parentJobId,
            seed
        });
        return res.data;
    },

    generateLyrics: async (topic: string, modelName: string, currentLyrics?: string) => {
        const res = await axios.post(`${API_BASE_URL}/generate/lyrics`, {
            topic,
            model_name: modelName,
            seed_lyrics: currentLyrics
        });
        return res.data.lyrics;
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
        llmModel: string = "llama3"
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
