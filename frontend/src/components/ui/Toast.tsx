import React, { useEffect } from 'react';
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
        <div className="fixed top-6 right-6 z-[110] animate-slide-down">
            <div
                role="status"
                className={`px-4 py-3 flex items-center gap-3 rounded-2xl border shadow-apple-2xl backdrop-blur-2xl bg-white/90 dark:bg-[#141620]/95 ${
                    type === 'success'
                        ? 'border-teal-500/30'
                        : 'border-rose-500/40'
                }`}
            >
                {type === 'success' ? (
                    <CheckCircle2 className="w-5 h-5 text-teal-600 dark:text-teal-400 flex-shrink-0" />
                ) : (
                    <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0" />
                )}
                <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{message}</span>
            </div>
        </div>
    );
};
