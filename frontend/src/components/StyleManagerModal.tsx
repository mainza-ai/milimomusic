import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, Trash2, Music, Beaker, Rocket, Loader2, GraduationCap } from 'lucide-react';
import { styleApi, type Style } from '../api';


interface StyleManagerModalProps {
    isOpen: boolean;
    onClose: () => void;
    onStylesChange?: () => void;
    onOpenTraining?: () => void;
}

export const StyleManagerModal: React.FC<StyleManagerModalProps> = ({
    isOpen,
    onClose,
    onStylesChange,
    onOpenTraining
}) => {
    const [activeTab, setActiveTab] = useState<'official' | 'custom'>('official');
    const [styles, setStyles] = useState<Style[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [newStyleName, setNewStyleName] = useState('');
    const [newStyleDesc, setNewStyleDesc] = useState('');
    const [isAdding, setIsAdding] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const officialStyles = styles.filter(s => s.type === 'official');
    const customStyles = styles.filter(s => s.type === 'custom' || s.type === 'trained');

    useEffect(() => {
        if (isOpen) {
            loadStyles();
        }
    }, [isOpen]);

    const loadStyles = async () => {
        setIsLoading(true);
        try {
            const data = await styleApi.getStyles();
            setStyles(data);
        } catch (e) {
            console.error('Failed to load styles', e);
        } finally {
            setIsLoading(false);
        }
    };

    const handleAddStyle = async () => {
        if (!newStyleName.trim()) return;

        setIsAdding(true);
        setError(null);
        try {
            await styleApi.addCustomStyle(newStyleName.trim(), newStyleDesc.trim() || undefined);
            setNewStyleName('');
            setNewStyleDesc('');
            await loadStyles();
            onStylesChange?.();
        } catch (e: any) {
            setError(e.response?.data?.detail || 'Failed to add style');
        } finally {
            setIsAdding(false);
        }
    };

    const handleDeleteStyle = async (name: string) => {
        try {
            await styleApi.removeCustomStyle(name);
            await loadStyles();
            onStylesChange?.();
        } catch (e) {
            console.error('Failed to delete style', e);
        }
    };

    const getStyleIcon = (style: Style) => {
        if (style.type === 'trained') return <Rocket className="w-3.5 h-3.5 text-purple-500" />;
        if (style.type === 'custom') return <Beaker className="w-3.5 h-3.5 text-amber-500" />;
        return <Music className="w-3.5 h-3.5 text-cyan-500" />;
    };

    const getStyleBadge = (style: Style) => {
        if (style.type === 'trained') return (
            <span className="text-[10px] bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded-full font-bold">
                🚀 Trained
            </span>
        );
        if (style.type === 'custom') return (
            <span className="text-[10px] bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full font-bold">
                ⚗️ Custom
            </span>
        );
        return null;
    };

    const handleOpenTraining = () => {
        if (onOpenTraining) {
            onClose(); // Close this modal
            onOpenTraining(); // Open the main one
        }
    };

    return (
        <>
            {createPortal(
                <AnimatePresence>
                    {isOpen && (
                        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                className="bg-white/90 backdrop-blur-2xl rounded-xl border border-white/50 shadow-2xl w-full max-w-2xl overflow-hidden flex flex-col max-h-[80vh] glass-panel"
                            >
                                {/* Header */}
                                <div className="flex items-center justify-between p-6 border-b border-white/20 bg-gradient-to-r from-cyan-50/80 to-purple-50/80">
                                    <h2 className="text-xl font-bold text-slate-800 flex items-center gap-3 tracking-tight">
                                        <span className="p-2 bg-white/50 rounded-lg shadow-sm text-lg">🎨</span>
                                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
                                            Style Manager
                                        </span>
                                    </h2>
                                    <button onClick={onClose} className="p-2 rounded-full hover:bg-white/50 text-slate-400 hover:text-red-500 transition-colors">
                                        <X className="w-5 h-5" />
                                    </button>
                                </div>

                                {/* Tabs */}
                                <div className="flex border-b border-white/20 bg-white/30 p-2 gap-2">
                                    <button
                                        onClick={() => setActiveTab('official')}
                                        className={`flex-1 py-2.5 text-sm font-bold rounded-lg transition-all flex items-center justify-center gap-2 ${activeTab === 'official'
                                            ? 'bg-white shadow-sm text-cyan-700 ring-1 ring-black/5'
                                            : 'text-slate-500 hover:text-slate-700 hover:bg-white/40'
                                            }`}
                                    >
                                        🎵 Official ({officialStyles.length})
                                    </button>
                                    <button
                                        onClick={() => setActiveTab('custom')}
                                        className={`flex-1 py-2.5 text-sm font-bold rounded-lg transition-all flex items-center justify-center gap-2 ${activeTab === 'custom'
                                            ? 'bg-white shadow-sm text-cyan-700 ring-1 ring-black/5'
                                            : 'text-slate-500 hover:text-slate-700 hover:bg-white/40'
                                            }`}
                                    >
                                        ⚗️ Custom ({customStyles.length})
                                    </button>
                                </div>

                                {/* Content */}
                                <div className="flex-1 overflow-y-auto p-4">
                                    {isLoading ? (
                                        <div className="flex items-center justify-center py-12">
                                            <Loader2 className="w-6 h-6 animate-spin text-cyan-500" />
                                        </div>
                                    ) : activeTab === 'official' ? (
                                        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                                            {officialStyles.map(style => (
                                                <div
                                                    key={style.name}
                                                    className="flex items-center gap-2 px-3 py-2 bg-slate-50 rounded-lg border border-slate-100 text-sm text-slate-700"
                                                >
                                                    {getStyleIcon(style)}
                                                    <span className="truncate">{style.name}</span>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="space-y-4">
                                            {/* Add New Style Form */}
                                            <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-4 border border-amber-100">
                                                <h4 className="text-sm font-bold text-amber-800 mb-3 flex items-center gap-2">
                                                    <Plus className="w-4 h-4" />
                                                    Add Custom Style
                                                </h4>
                                                <div className="flex gap-2">
                                                    <input
                                                        type="text"
                                                        value={newStyleName}
                                                        onChange={(e) => setNewStyleName(e.target.value)}
                                                        placeholder="Style name (e.g., Samba)"
                                                        className="flex-1 px-3 py-2 border border-amber-200 rounded-md focus:ring-2 focus:ring-amber-400 focus:outline-none text-sm bg-white"
                                                    />
                                                    <button
                                                        onClick={handleAddStyle}
                                                        disabled={isAdding || !newStyleName.trim()}
                                                        className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-md font-bold text-sm shadow-md transition-all disabled:opacity-50 flex items-center gap-2"
                                                    >
                                                        {isAdding ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                                                        Add
                                                    </button>
                                                </div>
                                                {error && (
                                                    <p className="text-red-600 text-xs mt-2">{error}</p>
                                                )}
                                                <p className="text-amber-700 text-xs mt-2">
                                                    ⚠️ Custom styles work best after fine-tuning. Use the Training Studio to train new styles.
                                                </p>
                                            </div>

                                            {/* Custom Styles List */}
                                            {customStyles.length === 0 ? (
                                                <div className="text-center py-8 text-slate-400">
                                                    <Beaker className="w-10 h-10 mx-auto mb-2 opacity-50" />
                                                    <p className="text-sm">No custom styles yet</p>
                                                </div>
                                            ) : (
                                                <div className="space-y-2">
                                                    {customStyles.map(style => (
                                                        <div
                                                            key={style.name}
                                                            className="flex items-center justify-between px-4 py-3 bg-white rounded-lg border border-slate-200 hover:border-slate-300 transition-colors"
                                                        >
                                                            <div className="flex items-center gap-3">
                                                                {getStyleIcon(style)}
                                                                <span className="font-medium text-slate-700">{style.name}</span>
                                                                {getStyleBadge(style)}
                                                            </div>
                                                            <button
                                                                onClick={() => handleDeleteStyle(style.name)}
                                                                className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                                                            >
                                                                <Trash2 className="w-4 h-4" />
                                                            </button>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {/* Footer */}
                                <div className="p-4 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between">
                                    <p className="text-xs text-slate-500">
                                        🎵 Official • ⚗️ Custom • 🚀 Trained
                                    </p>
                                    <button
                                        onClick={handleOpenTraining}
                                        className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white rounded-md font-bold text-sm transition-all flex items-center gap-2 shadow-md"
                                    >
                                        <GraduationCap className="w-4 h-4" />
                                        Training Studio
                                    </button>
                                </div>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>,
                document.body
            )}
        </>
    );
};
