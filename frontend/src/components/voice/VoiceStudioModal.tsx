import React from 'react';
import { Mic, X } from 'lucide-react';
import type { Job } from '../../api';
import { VocalStudioView } from '../views/VocalStudioView';

interface VoiceStudioModalProps {
    isOpen: boolean;
    onClose: () => void;
    initialTrack?: Job | null;
    initialStemPath?: string;
    onOpenWorkspace?: (job: Job) => void;
}

/**
 * Voice & Vocal Production Studio.
 *
 * The former "YuE2 My Music" training tab has been removed (Decision D1): its
 * trainer wrote a 20-byte placeholder file and reported it as a trained LoRA.
 */
export const VoiceStudioModal: React.FC<VoiceStudioModalProps> = ({
    isOpen,
    onClose,
    initialTrack,
    initialStemPath,
    onOpenWorkspace
}) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/70 backdrop-blur-md">
            <div className="w-full max-w-6xl bg-white dark:bg-[#12141c] border border-black/[0.08] dark:border-white/10 rounded-3xl shadow-apple-2xl flex flex-col max-h-[92vh] overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="px-6 py-4 border-b border-black/[0.06] dark:border-white/10 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center">
                            <Mic size={20} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                                <span>Voice &amp; Vocal Production Studio</span>
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">
                                Clone singer identities, live vocal booth recording, and neural singing voice conversion
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        aria-label="Close modal"
                        title="Close"
                        className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
                    <VocalStudioView
                        isModal={true}
                        initialTrack={initialTrack}
                        initialStemPath={initialStemPath}
                        onOpenWorkspace={onOpenWorkspace}
                        onCloseModal={onClose}
                    />
                </div>
            </div>
        </div>
    );
};
