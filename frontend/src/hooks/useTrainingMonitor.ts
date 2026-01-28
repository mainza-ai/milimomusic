import { useState, useEffect, useCallback } from 'react';
import { api, type TrainingJob } from '../api';

export function useTrainingMonitor() {
    const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
    const [lastUpdated, setLastUpdated] = useState<number>(Date.now());

    const checkStatus = useCallback(async () => {
        try {
            const jobs = await api.getTrainingJobs();
            // Find any running or preprocessing job
            // We assume mostly one active job at a time for simplicity in this MVP
            const running = jobs.find(j => j.status === 'running' || j.status === 'preprocessing');
            setActiveJob(running || null);
            setLastUpdated(Date.now());
        } catch (e) {
            console.error("Failed to monitor training status", e);
        }
    }, []);

    useEffect(() => {
        // Initial check
        checkStatus();

        // Poll every 3 seconds
        const interval = setInterval(checkStatus, 3000);
        return () => clearInterval(interval);
    }, [checkStatus]);

    // Expose a manual refresh if needed
    return {
        activeJob,
        refresh: checkStatus,
        lastUpdated
    };
}
