import React, { useEffect } from 'react';
import { GlassCard } from './GlassCard';
import { CheckCircle2, AlertCircle } from 'lucide-react';

interface ToastProps {
    message: string;
    type?: 'success' | 'error';
    onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ message, type = 'success', onClose }) => {
    useEffect(() => {
        const timer = setTimeout(onClose, 3000);
        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className="fixed top-6 right-6 z-[100] animate-in slide-in-from-top-5 fade-in duration-300">
            <GlassCard className="px-4 py-3 flex items-center gap-3 !bg-white/90 shadow-xl border-l-4 border-l-indigo-500">
                {type === 'success' ? (
                    <CheckCircle2 className="w-5 h-5 text-indigo-600" />
                ) : (
                    <AlertCircle className="w-5 h-5 text-red-500" />
                )}
                <span className="text-sm font-medium text-slate-700">{message}</span>
            </GlassCard>
        </div>
    );
};
