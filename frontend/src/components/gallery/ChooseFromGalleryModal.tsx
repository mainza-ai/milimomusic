import React, { useState, useEffect } from 'react';
import {
    X,
    Search,
    Heart,
    Sparkles,
    Image as ImageIcon,
    Check,
    Loader2
} from 'lucide-react';
import { imageStudioApi, coverApi, type VisualAsset } from '../../api';
import { toast } from '../../utils/toast';

interface ChooseFromGalleryModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectAsset: (asset: VisualAsset) => void;
    title?: string;
    targetJobTitle?: string;
}

export const ChooseFromGalleryModal: React.FC<ChooseFromGalleryModalProps> = ({
    isOpen,
    onClose,
    onSelectAsset,
    title = 'Choose Artwork from Gallery',
    targetJobTitle
}) => {
    const [assets, setAssets] = useState<VisualAsset[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [typeFilter, setTypeFilter] = useState<string>('all');
    const [favoriteOnly, setFavoriteOnly] = useState(false);
    const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);

    useEffect(() => {
        if (!isOpen) return;
        loadGallery();
    }, [isOpen, favoriteOnly, typeFilter]);

    const loadGallery = async () => {
        setIsLoading(true);
        try {
            const res = await imageStudioApi.getGallery({
                asset_type: typeFilter === 'all' ? undefined : typeFilter,
                favorite_only: favoriteOnly || undefined,
                limit: 100
            });
            setAssets(res.assets || []);
        } catch (err) {
            console.error('Failed to load gallery assets:', err);
            toast('Failed to load image gallery', 'error');
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    const filteredAssets = assets.filter(a => {
        if (!searchQuery.trim()) return true;
        const q = searchQuery.toLowerCase();
        return (
            (a.title && a.title.toLowerCase().includes(q)) ||
            (a.prompt && a.prompt.toLowerCase().includes(q)) ||
            (a.style && a.style.toLowerCase().includes(q))
        );
    });

    const handleConfirmSelection = (asset: VisualAsset) => {
        onSelectAsset(asset);
        onClose();
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/70 backdrop-blur-md animate-fade-in">
            <div
                className="bg-white dark:bg-[#141622] border border-black/10 dark:border-white/10 rounded-3xl w-full max-w-4xl max-h-[88vh] flex flex-col shadow-2xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/[0.08] bg-black/[0.02] dark:bg-white/[0.02]">
                    <div className="flex items-center space-x-3">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-500/20 to-cyan-500/20 text-teal-600 dark:text-teal-400 flex items-center justify-center border border-teal-500/30">
                            <ImageIcon size={18} />
                        </div>
                        <div>
                            <h2 className="text-base font-bold text-slate-900 dark:text-white">
                                {title}
                            </h2>
                            {targetJobTitle && (
                                <p className="text-xs text-slate-500 dark:text-slate-400">
                                    Setting cover for: <span className="font-semibold text-teal-600 dark:text-teal-400">{targetJobTitle}</span>
                                </p>
                            )}
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                        title="Close Modal"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Filters Strip */}
                <div className="px-6 py-3 border-b border-black/[0.04] dark:border-white/[0.06] flex flex-wrap items-center justify-between gap-3 bg-black/[0.01] dark:bg-white/[0.01]">
                    <div className="flex items-center space-x-2 flex-1 min-w-[200px]">
                        <div className="relative flex-1 max-w-sm">
                            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                            <input
                                type="text"
                                placeholder="Search gallery by prompt or title…"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-teal-500"
                            />
                        </div>

                        {/* Filter Tabs */}
                        <div className="flex items-center gap-1 bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10 text-xs">
                            {[
                                { id: 'all', label: 'All' },
                                { id: 'album_cover', label: 'Covers' },
                                { id: 'concept_art', label: 'Concepts' },
                                { id: 'scene_keyframe', label: 'Keyframes' },
                            ].map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setTypeFilter(tab.id)}
                                    className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all ${
                                        typeFilter === tab.id
                                            ? 'bg-white dark:bg-white/20 text-teal-600 dark:text-teal-300 shadow-sm'
                                            : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                                    }`}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <button
                        onClick={() => setFavoriteOnly(!favoriteOnly)}
                        className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
                            favoriteOnly
                                ? 'bg-rose-500/10 border-rose-500/20 text-rose-500'
                                : 'bg-black/[0.03] dark:bg-white/5 border-black/[0.06] dark:border-white/10 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                        }`}
                    >
                        <Heart size={13} className={favoriteOnly ? 'fill-rose-500' : ''} />
                        <span>Favorites</span>
                    </button>
                </div>

                {/* Content Grid */}
                <div className="flex-1 overflow-y-auto p-6 min-h-[320px]">
                    {isLoading ? (
                        <div className="flex flex-col items-center justify-center h-64 space-y-3">
                            <Loader2 size={28} className="animate-spin text-teal-500" />
                            <p className="text-xs text-slate-500 dark:text-slate-400">Loading gallery artwork…</p>
                        </div>
                    ) : filteredAssets.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-64 space-y-3 text-center">
                            <div className="w-12 h-12 rounded-2xl bg-black/5 dark:bg-white/5 flex items-center justify-center text-slate-400">
                                <Sparkles size={24} />
                            </div>
                            <div className="max-w-xs">
                                <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200">No artwork found</h3>
                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                    {searchQuery ? 'Try adjusting your search terms or filters.' : 'Generate images in the Image Studio to build your personal local art vault.'}
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                            {filteredAssets.map((asset) => {
                                const isSelected = selectedAssetId === asset.id;
                                const imgUrl = coverApi.getCoverUrl(asset.image_url);
                                return (
                                    <div
                                        key={asset.id}
                                        onClick={() => setSelectedAssetId(asset.id)}
                                        onDoubleClick={() => handleConfirmSelection(asset)}
                                        className={`group relative rounded-2xl overflow-hidden border cursor-pointer transition-all ${
                                            isSelected
                                                ? 'ring-2 ring-teal-500 border-teal-500 shadow-md scale-[1.02]'
                                                : 'border-black/[0.08] dark:border-white/10 hover:border-black/20 dark:hover:border-white/20 hover:scale-[1.01]'
                                        }`}
                                    >
                                        <div className="aspect-square bg-slate-900 overflow-hidden relative">
                                            <img
                                                src={imgUrl}
                                                alt={asset.title || asset.prompt}
                                                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                                                loading="lazy"
                                            />
                                            {asset.is_favorite && (
                                                <div className="absolute top-2 right-2 p-1 rounded-full bg-black/60 backdrop-blur-sm text-rose-400 shadow">
                                                    <Heart size={11} className="fill-rose-400" />
                                                </div>
                                            )}
                                            <div className="absolute bottom-2 left-2 px-1.5 py-0.5 rounded bg-black/60 backdrop-blur-sm text-[9px] font-mono text-white/90">
                                                {asset.aspect_ratio || '1:1'}
                                            </div>
                                            {isSelected && (
                                                <div className="absolute inset-0 bg-teal-500/25 flex items-center justify-center">
                                                    <div className="w-8 h-8 rounded-full bg-teal-500 text-slate-950 flex items-center justify-center shadow-lg font-bold">
                                                        <Check size={18} />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        <div className="p-2.5 bg-white/95 dark:bg-[#161826]/95">
                                            <p className="text-xs font-semibold text-slate-800 dark:text-slate-200 truncate">
                                                {asset.title || asset.prompt.slice(0, 30)}
                                            </p>
                                            <p className="text-[10px] text-slate-400 font-mono mt-0.5 truncate">
                                                {asset.asset_type.replace('_', ' ')} · {new Date(asset.created_at).toLocaleDateString()}
                                            </p>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Footer Actions */}
                <div className="px-6 py-4 border-t border-black/[0.06] dark:border-white/[0.08] bg-black/[0.02] dark:bg-white/[0.02] flex items-center justify-between">
                    <span className="text-xs text-slate-400 font-mono">
                        {filteredAssets.length} image{filteredAssets.length === 1 ? '' : 's'} available
                    </span>
                    <div className="flex items-center space-x-2">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 dark:text-slate-400 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={() => {
                                const selected = assets.find(a => a.id === selectedAssetId);
                                if (selected) handleConfirmSelection(selected);
                            }}
                            disabled={!selectedAssetId}
                            className="px-5 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-1.5"
                        >
                            <Check size={14} />
                            <span>Apply Selected Artwork</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
