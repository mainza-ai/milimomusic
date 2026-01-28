import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, FolderOpen, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { pathsApi, type PathsConfig } from '../api';

interface PathsSettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const PathsSettingsModal: React.FC<PathsSettingsModalProps> = ({ isOpen, onClose }) => {
    const [config, setConfig] = useState<PathsConfig>({
        model_directory: '',
        checkpoints_directory: '',
        datasets_directory: ''
    });
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [validation, setValidation] = useState<Record<string, { valid: boolean; path: string }>>({});
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        if (isOpen) {
            loadConfig();
        }
    }, [isOpen]);

    const loadConfig = async () => {
        setIsLoading(true);
        try {
            const data = await pathsApi.getConfig();
            setConfig(data);
            // Validate on load
            const validationResult = await pathsApi.validate(data);
            setValidation(validationResult);
        } catch (e) {
            console.error('Failed to load paths config', e);
        } finally {
            setIsLoading(false);
        }
    };

    const handleChange = (field: keyof PathsConfig, value: string) => {
        setConfig(prev => ({ ...prev, [field]: value }));
        setMessage(null);
    };

    const handleValidate = async () => {
        try {
            const result = await pathsApi.validate(config);
            setValidation(result);
        } catch (e) {
            console.error('Failed to validate paths', e);
        }
    };

    const handleSave = async () => {
        setIsSaving(true);
        setMessage(null);
        try {
            await pathsApi.updateConfig(config);
            setMessage({ type: 'success', text: 'Paths saved successfully!' });
            // Re-validate after save
            const result = await pathsApi.validate(config);
            setValidation(result);
        } catch (e: any) {
            setMessage({ type: 'error', text: e.response?.data?.detail || 'Failed to save paths' });
        } finally {
            setIsSaving(false);
        }
    };

    const pathFields: { key: keyof PathsConfig; label: string; description: string }[] = [
        {
            key: 'model_directory',
            label: 'Model Directory',
            description: 'Base directory for HeartMuLa model files'
        },
        {
            key: 'checkpoints_directory',
            label: 'Checkpoints Directory',
            description: 'Where trained model checkpoints are saved'
        },
        {
            key: 'datasets_directory',
            label: 'Datasets Directory',
            description: 'Where training datasets and audio files are stored'
        }
    ];

    return createPortal(
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="bg-white rounded-xl shadow-2xl w-full max-w-lg overflow-hidden"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-slate-100 bg-gradient-to-r from-slate-700 to-slate-800">
                            <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                <FolderOpen className="w-5 h-5" />
                                Path Settings
                            </h2>
                            <button
                                onClick={onClose}
                                className="p-1 rounded-full hover:bg-white/20 text-white/80 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="p-6 space-y-5">
                            {isLoading ? (
                                <div className="flex items-center justify-center py-12">
                                    <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
                                </div>
                            ) : (
                                <>
                                    {pathFields.map(field => (
                                        <div key={field.key}>
                                            <label className="block text-sm font-medium text-slate-700 mb-1">
                                                {field.label}
                                            </label>
                                            <div className="relative">
                                                <input
                                                    type="text"
                                                    value={config[field.key] || ''}
                                                    onChange={(e) => handleChange(field.key, e.target.value)}
                                                    onBlur={handleValidate}
                                                    placeholder={`/path/to/${field.key.replace('_', '-')}`}
                                                    className={`w-full px-3 py-2 pr-10 border rounded-md focus:ring-2 focus:outline-none text-sm ${validation[field.key]?.valid === false
                                                            ? 'border-red-300 focus:ring-red-400'
                                                            : validation[field.key]?.valid === true
                                                                ? 'border-green-300 focus:ring-green-400'
                                                                : 'border-slate-200 focus:ring-cyan-400'
                                                        }`}
                                                />
                                                {validation[field.key] && (
                                                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                                                        {validation[field.key].valid ? (
                                                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                                                        ) : (
                                                            <AlertCircle className="w-4 h-4 text-red-500" />
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-xs text-slate-500 mt-1">{field.description}</p>
                                            {validation[field.key]?.valid === false && (
                                                <p className="text-xs text-red-500 mt-1">
                                                    Directory does not exist or is not accessible
                                                </p>
                                            )}
                                        </div>
                                    ))}

                                    {/* Message */}
                                    {message && (
                                        <div className={`p-3 rounded-lg text-sm flex items-center gap-2 ${message.type === 'success'
                                                ? 'bg-green-50 text-green-700 border border-green-200'
                                                : 'bg-red-50 text-red-700 border border-red-200'
                                            }`}>
                                            {message.type === 'success' ? (
                                                <CheckCircle2 className="w-4 h-4" />
                                            ) : (
                                                <AlertCircle className="w-4 h-4" />
                                            )}
                                            {message.text}
                                        </div>
                                    )}
                                </>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="p-4 border-t border-slate-100 bg-slate-50 flex justify-end gap-3">
                            <button
                                onClick={onClose}
                                className="px-4 py-2 text-slate-600 hover:bg-slate-200 rounded-md transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleSave}
                                disabled={isSaving}
                                className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-md font-bold text-sm transition-colors disabled:opacity-50 flex items-center gap-2"
                            >
                                {isSaving ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Save className="w-4 h-4" />
                                )}
                                Save
                            </button>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>,
        document.body
    );
};
