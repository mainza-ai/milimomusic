import React, { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';

interface TaskProgress {
    jobId: string;
    stage: string;
    progress: number;
    message: string;
}

export const FloatingStatusWidget: React.FC = () => {
    const [task, setTask] = useState<TaskProgress | null>(null);

    useEffect(() => {
        const handleProgress = (e: any) => {
            const data = e.detail;
            if (data && (data.progress !== undefined || data.msg || data.stage)) {
                setTask({
                    jobId: data.job_id || 'active',
                    stage: data.stage || 'Synthesizing Audio',
                    progress: data.progress || 0,
                    message: data.msg || data.message || 'Processing audio tensor frames...'
                });

                if (data.progress >= 100) {
                    setTimeout(() => setTask(null), 3000);
                }
            }
        };

        window.addEventListener('milimo_progress', handleProgress);
        return () => window.removeEventListener('milimo_progress', handleProgress);
    }, []);

    if (!task) return null;

    return (
        <div className="fixed bottom-6 right-6 z-50 bg-white/90 dark:bg-[#141620]/95 backdrop-blur-2xl border border-teal-500/30 rounded-3xl shadow-apple-lg p-4 w-80 text-slate-800 dark:text-slate-200 animate-slide-up select-none">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping" />
                    <span className="text-xs font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                        {task.stage}
                    </span>
                </div>
                <Sparkles size={14} className="text-teal-500 dark:text-teal-400 animate-pulse" />
            </div>

            <p className="text-xs text-slate-600 dark:text-slate-300 truncate mb-2 font-mono">
                {task.message}
            </p>

            <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden shadow-inner">
                <div
                    className="bg-gradient-to-r from-teal-500 to-cyan-400 h-full transition-all duration-300 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, task.progress))}%` }}
                />
            </div>
            <div className="flex justify-between text-[10px] text-slate-500 dark:text-slate-400 font-mono mt-1 font-semibold">
                <span>Pipeline Active</span>
                <span>{task.progress}%</span>
            </div>
        </div>
    );
};
