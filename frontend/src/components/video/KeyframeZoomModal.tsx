import React from 'react';
import { X, Download, Sparkles } from 'lucide-react';
import { api, type VideoClipSegment } from '../../api';

interface KeyframeZoomModalProps {
    isOpen: boolean;
    onClose: () => void;
    clipIndex: number | null;
    keyframeUrl: string | null;
    clipSegment?: VideoClipSegment;
}

export const KeyframeZoomModal: React.FC<KeyframeZoomModalProps> = ({
    isOpen,
    onClose,
    clipIndex,
    keyframeUrl,
    clipSegment,
}) => {
    if (!isOpen || !keyframeUrl) return null;

    const fullUrl = api.getAudioUrl(keyframeUrl);

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
            <div className="w-full max-w-3xl bg-slate-950 border border-white/15 rounded-3xl shadow-2xl flex flex-col overflow-hidden">
                {/* Header */}
                <div className="px-6 py-3.5 border-b border-white/10 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                        <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-teal-500/20 text-teal-300">
                            Keyframe Still #{clipIndex}
                        </span>
                        {clipSegment && (
                            <span className="text-xs text-slate-400 font-mono">
                                {clipSegment.time_str} ({clipSegment.duration.toFixed(1)}s)
                            </span>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => api.downloadUrlAsFile(fullUrl, `scene_${clipIndex}_keyframe.png`)}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-white transition-colors"
                            title="Download still image"
                        >
                            <Download size={16} />
                        </button>
                        <button
                            type="button"
                            onClick={onClose}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-white transition-colors"
                        >
                            <X size={16} />
                        </button>
                    </div>
                </div>

                {/* Keyframe Stills Image Canvas */}
                <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
                    <img
                        src={fullUrl}
                        alt={`Keyframe ${clipIndex}`}
                        className="w-full h-full object-contain"
                    />
                </div>

                {/* Footer Prompt Details */}
                {clipSegment && (
                    <div className="p-4 bg-slate-900/80 border-t border-white/10 space-y-1.5 text-xs">
                        <div className="flex items-center justify-between">
                            <span className="font-bold text-slate-200 flex items-center gap-1.5">
                                <Sparkles size={13} className="text-teal-400" />
                                <span>Scene Prompt</span>
                            </span>
                            <span className="text-[10px] text-slate-400 font-mono">
                                🎥 {clipSegment.camera}
                            </span>
                        </div>
                        <p className="text-slate-300 font-medium">
                            {clipSegment.prompt}
                        </p>
                        {clipSegment.lyrics && (
                            <p className="text-[11px] italic text-cyan-400">
                                "{clipSegment.lyrics}"
                            </p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};
