import React, { useState, useEffect, useMemo } from 'react';
import {
    Sparkles,
    Image as ImageIcon,
    Download,
    Trash2,
    Heart,
    Copy,
    Check,
    Search,
    RefreshCw,
    Sliders,
    Disc,
    Maximize2,
    X,
    ChevronDown,
    ChevronUp,
    Loader2
} from 'lucide-react';
import {
    imageStudioApi,
    coverApi,
    type VisualAsset,
    type Job
} from '../../api';
import { toast } from '../../utils/toast';

interface ImageStudioViewProps {
    songs?: Job[];
    onSetSongCover?: (songId: string, coverUrl: string) => void;
}

const STYLE_PRESETS = [
    { id: 'cinematic concept art', label: 'Cinematic Concept', desc: '8k dramatic lighting, volumetric atmosphere' },
    { id: 'photorealistic album cover, 8k, high fashion editorial', label: 'Photorealistic Cover', desc: 'Sharp studio lighting, glossy finish' },
    { id: 'neon cyberpunk synthwave, retro-futuristic', label: 'Cyberpunk Synthwave', desc: 'Vibrant neons, nocturnal cityscape' },
    { id: 'minimalist vinyl record jacket art, graphic design swiss style', label: 'Minimalist Vinyl', desc: 'Clean typography space, bold color blocking' },
    { id: 'dark fantasy surrealism, oil painting texture', label: 'Dark Surrealism', desc: 'Ethereal mood, textured brushstrokes' },
    { id: 'vintage 70s analog 35mm film photography, kodachrome warmth', label: '70s Film Analog', desc: 'Warm grain, vintage lens flares' },
    { id: 'vibrant anime aesthetic, makoto shinkai skies, detailed clouds', label: 'Anime Studio', desc: 'Luminous skies, anime production art' },
    { id: 'abstract geometric bauhaus art, modernist constructivism', label: 'Geometric Bauhaus', desc: 'Abstract shapes, modern contrast' },
];

const ASPECT_RATIOS = [
    { id: '1:1', label: '1:1 Square', sub: 'Album Covers', iconClass: 'w-4 h-4 rounded-sm border-2' },
    { id: '16:9', label: '16:9 Wide', sub: 'Video / Cinema', iconClass: 'w-5 h-3 rounded-sm border-2' },
    { id: '9:16', label: '9:16 Vertical', sub: 'Shorts / Stories', iconClass: 'w-3 h-5 rounded-sm border-2' },
    { id: '4:3', label: '4:3 Classic', sub: 'Retro Photo', iconClass: 'w-4 h-3 rounded-sm border-2' },
];

const STARTER_PROMPTS = [
    "Futuristic jazz saxophonist playing atop a rainy Tokyo skyscraper at dusk, purple and amber neon reflections",
    "Minimalist cassette tape melting into a cosmic nebula of liquid soundwaves, gold and deep obsidian black",
    "Ethereal choir performing inside an ancient cathedral carved into a crystalline ice cavern, god rays",
    "Afrofuturistic cyborg singer with glowing fiber-optic braids, high-fashion metallic studio portrait",
    "Lonely analog synthesizer resting on a serene Norwegian fjord shoreline during the midnight sun",
];

export const ImageStudioView: React.FC<ImageStudioViewProps> = ({
    songs = [],
    onSetSongCover
}) => {
    // Generation Form State
    const [prompt, setPrompt] = useState('');
    const [title, setTitle] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('');
    const [selectedStyle, setSelectedStyle] = useState(STYLE_PRESETS[0].id);
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [assetType, setAssetType] = useState<'album_cover' | 'concept_art' | 'scene_keyframe' | 'character_plate'>('concept_art');
    const [targetSongId, setTargetSongId] = useState<string>('');
    const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);

    // Gallery State
    const [assets, setAssets] = useState<VisualAsset[]>([]);
    const [isLoadingGallery, setIsLoadingGallery] = useState(false);
    const [galleryFilter, setGalleryFilter] = useState<'all' | 'album_cover' | 'concept_art' | 'scene_keyframe' | 'favorites'>('all');
    const [searchQuery, setSearchQuery] = useState('');

    // Modal / Lightbox State
    const [activeLightboxAsset, setActiveLightboxAsset] = useState<VisualAsset | null>(null);
    const [targetTrackModalAsset, setTargetTrackModalAsset] = useState<VisualAsset | null>(null);
    const [copiedPromptId, setCopiedPromptId] = useState<string | null>(null);

    useEffect(() => {
        loadGallery();
    }, []);

    const loadGallery = async () => {
        setIsLoadingGallery(true);
        try {
            const res = await imageStudioApi.getGallery({ limit: 150 });
            setAssets(res.assets || []);
        } catch (err) {
            console.error('Failed to load gallery:', err);
            toast('Failed to load image gallery', 'error');
        } finally {
            setIsLoadingGallery(false);
        }
    };

    const handleGenerate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt.trim()) {
            toast('Please enter an image prompt', 'error');
            return;
        }

        setIsGenerating(true);
        try {
            toast('Generating visual asset with local studio engine...', 'info');
            const res = await imageStudioApi.generateImage({
                prompt: prompt.trim(),
                title: title.trim() || undefined,
                negative_prompt: negativePrompt.trim() || undefined,
                style: selectedStyle,
                aspect_ratio: aspectRatio,
                asset_type: assetType,
                linked_job_id: targetSongId || undefined
            });

            if (res.success && res.asset) {
                toast(`Artwork generated successfully with ${res.engine || 'local studio engine'}!`, 'success');
                // Prepend to gallery
                setAssets(prev => [res.asset, ...prev]);

                // If user selected a target song, assign it as cover
                if (targetSongId) {
                    await imageStudioApi.setAsCover(res.asset.id, targetSongId);
                    onSetSongCover?.(targetSongId, res.asset.image_url);
                    toast('Applied directly as track album cover!', 'success');
                }
            }
        } catch (err: any) {
            console.error('Generation error:', err);
            toast(err?.response?.data?.detail || err?.message || 'Failed to generate image', 'error');
        } finally {
            setIsGenerating(false);
        }
    };

    const handleToggleFavorite = async (asset: VisualAsset, e?: React.MouseEvent) => {
        if (e) e.stopPropagation();
        const nextFav = !asset.is_favorite;
        // Optimistic update
        setAssets(prev => prev.map(a => a.id === asset.id ? { ...a, is_favorite: nextFav } : a));
        if (activeLightboxAsset?.id === asset.id) {
            setActiveLightboxAsset(prev => prev ? { ...prev, is_favorite: nextFav } : null);
        }

        try {
            await imageStudioApi.updateAsset(asset.id, { is_favorite: nextFav });
        } catch (err) {
            // Revert on error
            setAssets(prev => prev.map(a => a.id === asset.id ? { ...a, is_favorite: !nextFav } : a));
            toast('Failed to update favorite', 'error');
        }
    };

    const handleDeleteAsset = async (assetId: string, e?: React.MouseEvent) => {
        if (e) e.stopPropagation();
        if (!window.confirm('Are you sure you want to delete this artwork from your local gallery?')) return;

        try {
            await imageStudioApi.deleteAsset(assetId);
            setAssets(prev => prev.filter(a => a.id !== assetId));
            if (activeLightboxAsset?.id === assetId) setActiveLightboxAsset(null);
            toast('Visual asset deleted', 'info');
        } catch (err) {
            toast('Failed to delete asset', 'error');
        }
    };

    const handleSetAsSongCover = async (asset: VisualAsset, songId: string) => {
        try {
            await imageStudioApi.setAsCover(asset.id, songId);
            onSetSongCover?.(songId, asset.image_url);
            setTargetTrackModalAsset(null);
            toast('Album cover updated successfully!', 'success');
        } catch (err) {
            toast('Failed to set album cover', 'error');
        }
    };

    const handleCopyPrompt = (asset: VisualAsset, e?: React.MouseEvent) => {
        if (e) e.stopPropagation();
        navigator.clipboard.writeText(asset.prompt);
        setCopiedPromptId(asset.id);
        toast('Prompt copied to clipboard', 'info');
        setTimeout(() => setCopiedPromptId(null), 2000);
    };

    const filteredAssets = useMemo(() => {
        return assets.filter(a => {
            // Category / Tab filter
            if (galleryFilter === 'favorites' && !a.is_favorite) return false;
            if (galleryFilter !== 'all' && galleryFilter !== 'favorites' && a.asset_type !== galleryFilter) return false;

            // Search query filter
            if (searchQuery.trim()) {
                const q = searchQuery.toLowerCase();
                const matches = (a.title && a.title.toLowerCase().includes(q)) ||
                                (a.prompt && a.prompt.toLowerCase().includes(q)) ||
                                (a.style && a.style.toLowerCase().includes(q));
                if (!matches) return false;
            }
            return true;
        });
    }, [assets, galleryFilter, searchQuery]);

    return (
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 space-y-8 max-w-7xl mx-auto w-full min-w-0">
            {/* Studio Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-black/[0.06] dark:border-white/[0.08]">
                <div>
                    <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-teal-500 to-cyan-500 flex items-center justify-center text-slate-950 font-bold shadow-md shadow-teal-500/20">
                            <ImageIcon size={22} />
                        </div>
                        <div>
                            <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                                Standalone Image Generation Studio
                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                                    Local-First FLUX & MLX
                                </span>
                            </h1>
                            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                                Generate concepts, album covers, and cinematic keyframes on-device with continuous local file gallery storage.
                            </p>
                        </div>
                    </div>
                </div>

                <div className="flex items-center space-x-2">
                    <button
                        onClick={loadGallery}
                        disabled={isLoadingGallery}
                        className="px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-semibold text-xs flex items-center gap-1.5 transition-colors border border-black/[0.06] dark:border-white/10"
                        title="Refresh Gallery"
                    >
                        <RefreshCw size={13} className={isLoadingGallery ? 'animate-spin' : ''} />
                        <span>Refresh Vault</span>
                    </button>
                </div>
            </div>

            {/* Split Creator Layout: Generation Command Deck + Flow Gallery */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* 1. Generator Control Deck (5 Cols on Desktop) */}
                <div className="lg:col-span-5 bg-white/95 dark:bg-[#141622]/95 rounded-3xl border border-black/[0.08] dark:border-white/10 p-5 sm:p-6 shadow-apple-lg space-y-5">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                            <Sparkles size={14} className="text-teal-500" />
                            Artwork Synthesis Deck
                        </span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20">
                            Zero Cloud Latency
                        </span>
                    </div>

                    <form onSubmit={handleGenerate} className="space-y-4">
                        {/* Prompt Input */}
                        <div>
                            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                                Visual Prompt
                            </label>
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                rows={3}
                                placeholder="Describe your album cover or visual concept in rich sensory detail…"
                                className="apple-input w-full text-xs p-3 leading-relaxed rounded-xl resize-none focus:ring-teal-500"
                            />
                        </div>

                        {/* Starter Prompts Carousel */}
                        <div>
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                                Inspiration Seeds
                            </span>
                            <div className="flex flex-wrap gap-1.5">
                                {STARTER_PROMPTS.slice(0, 3).map((seed, idx) => (
                                    <button
                                        key={idx}
                                        type="button"
                                        onClick={() => setPrompt(seed)}
                                        className="text-[11px] text-left px-2.5 py-1 rounded-lg bg-black/[0.03] dark:bg-white/5 hover:bg-teal-500/10 hover:text-teal-600 dark:hover:text-teal-300 text-slate-600 dark:text-slate-400 border border-black/[0.04] dark:border-white/5 transition-all truncate max-w-full"
                                    >
                                        ✨ {seed.slice(0, 42)}…
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Style Preset Selector */}
                        <div>
                            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5 flex items-center justify-between">
                                <span>Visual Style Preset</span>
                                <span className="text-[10px] text-slate-400 font-mono font-normal">Pitched aesthetic</span>
                            </label>
                            <div className="grid grid-cols-2 gap-2">
                                {STYLE_PRESETS.map((preset) => {
                                    const isSelected = selectedStyle === preset.id;
                                    return (
                                        <button
                                            key={preset.id}
                                            type="button"
                                            onClick={() => setSelectedStyle(preset.id)}
                                            className={`p-2.5 rounded-xl border text-left transition-all ${
                                                isSelected
                                                    ? 'bg-teal-500/10 border-teal-500/40 text-teal-900 dark:text-teal-200 shadow-sm'
                                                    : 'bg-black/[0.02] dark:bg-white/[0.03] border-black/[0.05] dark:border-white/5 text-slate-600 dark:text-slate-400 hover:border-black/15 dark:hover:border-white/15'
                                            }`}
                                        >
                                            <div className="text-xs font-bold truncate">{preset.label}</div>
                                            <div className="text-[10px] text-slate-400 truncate mt-0.5">{preset.desc}</div>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Aspect Ratio Selector */}
                        <div>
                            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                                Canvas Aspect Ratio
                            </label>
                            <div className="grid grid-cols-4 gap-2">
                                {ASPECT_RATIOS.map((ar) => {
                                    const isSelected = aspectRatio === ar.id;
                                    return (
                                        <button
                                            key={ar.id}
                                            type="button"
                                            onClick={() => setAspectRatio(ar.id)}
                                            className={`p-2 rounded-xl border flex flex-col items-center justify-center text-center transition-all ${
                                                isSelected
                                                    ? 'bg-teal-500/10 border-teal-500/40 text-teal-700 dark:text-teal-300 shadow-sm'
                                                    : 'bg-black/[0.02] dark:bg-white/[0.03] border-black/[0.05] dark:border-white/5 text-slate-600 dark:text-slate-400 hover:border-black/15'
                                            }`}
                                        >
                                            <div className={`${ar.iconClass} border-current mb-1 opacity-75`} />
                                            <span className="text-[11px] font-bold">{ar.id}</span>
                                            <span className="text-[9px] text-slate-400 truncate">{ar.sub}</span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Asset Type Selector */}
                        <div>
                            <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                                Asset Role
                            </label>
                            <div className="grid grid-cols-2 gap-2">
                                {[
                                    { id: 'concept_art', label: 'Concept Art', desc: 'Ideation & Moodboard' },
                                    { id: 'album_cover', label: 'Album Cover', desc: 'Square Release Art' },
                                    { id: 'scene_keyframe', label: 'Video Keyframe', desc: 'Timeline Visual Plate' },
                                    { id: 'character_plate', label: 'Performer Plate', desc: 'Avatar / Lip-Sync Base' },
                                ] .map((type) => (
                                    <button
                                        key={type.id}
                                        type="button"
                                        onClick={() => setAssetType(type.id as any)}
                                        className={`p-2 rounded-xl border text-left transition-all ${
                                            assetType === type.id
                                                ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-800 dark:text-cyan-200'
                                                : 'bg-black/[0.02] dark:bg-white/[0.03] border-black/[0.05] dark:border-white/5 text-slate-600 dark:text-slate-400'
                                        }`}
                                    >
                                        <div className="text-[11px] font-bold">{type.label}</div>
                                        <div className="text-[9px] text-slate-400">{type.desc}</div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Optional Target Song Link */}
                        {songs.length > 0 && (
                            <div>
                                <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                                    Link Immediately to Song (Optional)
                                </label>
                                <select
                                    value={targetSongId}
                                    onChange={(e) => setTargetSongId(e.target.value)}
                                    className="apple-input w-full text-xs py-2 px-3 rounded-xl"
                                >
                                    <option value="">-- Do not link (save to gallery only) --</option>
                                    {songs.map((song) => (
                                        <option key={song.id} value={song.id}>
                                            🎵 {song.title || song.prompt.slice(0, 30)} ({song.id.slice(0, 6)})
                                        </option>
                                    ))}
                                </select>
                            </div>
                        )}

                        {/* Advanced Expandable Controls */}
                        <div className="pt-1">
                            <button
                                type="button"
                                onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                                className="flex items-center justify-between w-full text-xs font-semibold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition-colors py-1"
                            >
                                <span className="flex items-center gap-1.5">
                                    <Sliders size={13} />
                                    Advanced Configuration
                                </span>
                                {isAdvancedOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            </button>

                            {isAdvancedOpen && (
                                <div className="mt-2.5 p-3 rounded-2xl bg-black/[0.03] dark:bg-white/[0.03] border border-black/[0.05] dark:border-white/5 space-y-3 animate-fade-in">
                                    <div>
                                        <label className="block text-[11px] font-bold text-slate-600 dark:text-slate-400 mb-1">
                                            Asset Title
                                        </label>
                                        <input
                                            type="text"
                                            value={title}
                                            onChange={(e) => setTitle(e.target.value)}
                                            placeholder="Optional display title"
                                            className="apple-input w-full text-xs py-1.5 px-2.5 rounded-lg"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-[11px] font-bold text-slate-600 dark:text-slate-400 mb-1">
                                            Negative Prompt
                                        </label>
                                        <input
                                            type="text"
                                            value={negativePrompt}
                                            onChange={(e) => setNegativePrompt(e.target.value)}
                                            placeholder="blurry, distorted text, ugly artifacts, watermark"
                                            className="apple-input w-full text-xs py-1.5 px-2.5 rounded-lg"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Submit Action Button */}
                        <button
                            type="submit"
                            disabled={isGenerating || !prompt.trim()}
                            className="w-full py-3 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-2xl flex items-center justify-center space-x-2 transition-all shadow-md shadow-teal-500/20 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isGenerating ? (
                                <>
                                    <Loader2 size={16} className="animate-spin" />
                                    <span>Synthesizing Local Artwork…</span>
                                </>
                            ) : (
                                <>
                                    <Sparkles size={16} />
                                    <span>Generate Studio Artwork</span>
                                </>
                            )}
                        </button>
                    </form>
                </div>

                {/* 2. Visual Asset Vault & Google Flow Gallery (7 Cols on Desktop) */}
                <div className="lg:col-span-7 space-y-4">
                    {/* Gallery Filters & Search Strip */}
                    <div className="bg-white/95 dark:bg-[#141622]/95 rounded-2xl border border-black/[0.08] dark:border-white/10 p-3 shadow-apple-sm flex flex-wrap items-center justify-between gap-3">
                        {/* Search Input */}
                        <div className="relative flex-1 min-w-[180px]">
                            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Filter gallery by keyword…"
                                className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-teal-500"
                            />
                        </div>

                        {/* Filter Categories */}
                        <div className="flex items-center gap-1 bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10 text-xs">
                            {[
                                { id: 'all', label: 'All' },
                                { id: 'album_cover', label: 'Covers' },
                                { id: 'concept_art', label: 'Concepts' },
                                { id: 'scene_keyframe', label: 'Keyframes' },
                                { id: 'favorites', label: '★ Liked' },
                            ].map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setGalleryFilter(tab.id as any)}
                                    className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all ${
                                        galleryFilter === tab.id
                                            ? 'bg-white dark:bg-white/20 text-teal-600 dark:text-teal-300 shadow-sm'
                                            : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                                    }`}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Gallery Grid */}
                    {isLoadingGallery ? (
                        <div className="flex flex-col items-center justify-center h-80 space-y-3 bg-white/50 dark:bg-[#141622]/50 rounded-3xl border border-black/[0.06] dark:border-white/[0.08]">
                            <Loader2 size={32} className="animate-spin text-teal-500" />
                            <p className="text-xs text-slate-500 dark:text-slate-400">Loading your visual asset library…</p>
                        </div>
                    ) : filteredAssets.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-80 space-y-3 text-center bg-white/50 dark:bg-[#141622]/50 rounded-3xl border border-black/[0.06] dark:border-white/[0.08] p-6">
                            <div className="w-14 h-14 rounded-2xl bg-black/5 dark:bg-white/5 flex items-center justify-center text-slate-400">
                                <ImageIcon size={28} />
                            </div>
                            <div className="max-w-sm">
                                <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200">No artwork found</h3>
                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                    {searchQuery ? 'No images matched your search criteria.' : 'Create your first piece of local visual art using the deck on the left!'}
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 gap-4">
                            {filteredAssets.map((asset) => {
                                const imgUrl = coverApi.getCoverUrl(asset.image_url);
                                return (
                                    <div
                                        key={asset.id}
                                        className="group relative rounded-2xl overflow-hidden border border-black/[0.08] dark:border-white/10 bg-white dark:bg-[#161826] shadow-apple-sm hover:shadow-apple-md transition-all flex flex-col"
                                    >
                                        {/* Image Box */}
                                        <div
                                            className="relative aspect-square overflow-hidden bg-slate-950 cursor-pointer"
                                            onClick={() => setActiveLightboxAsset(asset)}
                                        >
                                            <img
                                                src={imgUrl}
                                                alt={asset.title || asset.prompt}
                                                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                                                loading="lazy"
                                            />

                                            {/* Hover HUD Overlay */}
                                            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity p-2.5 flex flex-col justify-between">
                                                {/* Top Overlay Actions */}
                                                <div className="flex items-center justify-between">
                                                    <button
                                                        onClick={(e) => handleToggleFavorite(asset, e)}
                                                        className={`p-1.5 rounded-xl backdrop-blur-md transition-transform hover:scale-110 ${
                                                            asset.is_favorite
                                                                ? 'bg-rose-500/20 text-rose-400'
                                                                : 'bg-black/40 text-white/80 hover:text-rose-400'
                                                        }`}
                                                        title="Favorite Artwork"
                                                    >
                                                        <Heart size={14} className={asset.is_favorite ? 'fill-rose-400' : ''} />
                                                    </button>

                                                    <div className="flex items-center space-x-1">
                                                        <button
                                                            onClick={(e) => handleCopyPrompt(asset, e)}
                                                            className="p-1.5 rounded-xl bg-black/40 text-white/80 hover:text-white backdrop-blur-md hover:scale-110 transition-transform"
                                                            title="Copy Prompt"
                                                        >
                                                            {copiedPromptId === asset.id ? <Check size={14} className="text-teal-400" /> : <Copy size={14} />}
                                                        </button>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setActiveLightboxAsset(asset);
                                                            }}
                                                            className="p-1.5 rounded-xl bg-black/40 text-white/80 hover:text-white backdrop-blur-md hover:scale-110 transition-transform"
                                                            title="Enlarge View"
                                                        >
                                                            <Maximize2 size={14} />
                                                        </button>
                                                        <button
                                                            onClick={(e) => handleDeleteAsset(asset.id, e)}
                                                            className="p-1.5 rounded-xl bg-black/40 text-white/80 hover:text-rose-400 backdrop-blur-md hover:scale-110 transition-transform"
                                                            title="Delete Artwork"
                                                        >
                                                            <Trash2 size={14} />
                                                        </button>
                                                    </div>
                                                </div>

                                                {/* Bottom Overlay Action: Quick Set as Cover */}
                                                <div>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            setTargetTrackModalAsset(asset);
                                                        }}
                                                        className="w-full py-1.5 px-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-[10px] flex items-center justify-center gap-1 shadow-sm transition-all"
                                                    >
                                                        <Disc size={12} />
                                                        <span>Set as Album Cover</span>
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Badges */}
                                            <div className="absolute bottom-2 left-2 flex items-center gap-1 pointer-events-none group-hover:opacity-0 transition-opacity">
                                                <span className="px-1.5 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[9px] font-mono text-white/90">
                                                    {asset.aspect_ratio || '1:1'}
                                                </span>
                                                <span className="px-1.5 py-0.5 rounded bg-teal-500/80 backdrop-blur-sm text-[9px] font-mono text-slate-950 font-bold">
                                                    {asset.asset_type.replace('_', ' ')}
                                                </span>
                                            </div>
                                            {asset.is_favorite && (
                                                <div className="absolute top-2 right-2 p-1 rounded-full bg-black/70 text-rose-400 pointer-events-none group-hover:opacity-0 transition-opacity">
                                                    <Heart size={11} className="fill-rose-400" />
                                                </div>
                                            )}
                                        </div>

                                        {/* Meta Footer */}
                                        <div className="p-2.5">
                                            <p className="text-xs font-semibold text-slate-900 dark:text-slate-100 truncate">
                                                {asset.title || asset.prompt.slice(0, 32)}
                                            </p>
                                            <p className="text-[10px] text-slate-400 font-mono mt-0.5 truncate">
                                                {new Date(asset.created_at).toLocaleDateString()} · {asset.style?.split(',')[0] || 'Concept'}
                                            </p>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>

            {/* Lightbox High-Resolution Modal */}
            {activeLightboxAsset && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md animate-fade-in"
                    onClick={() => setActiveLightboxAsset(null)}
                >
                    <div
                        className="bg-white dark:bg-[#141624] border border-black/10 dark:border-white/10 rounded-3xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/[0.08]">
                            <div>
                                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                                    {activeLightboxAsset.title || 'Studio Artwork Inspection'}
                                </h3>
                                <p className="text-xs text-slate-400 font-mono">
                                    {activeLightboxAsset.width || 1024} × {activeLightboxAsset.height || 1024} · {activeLightboxAsset.aspect_ratio}
                                </p>
                            </div>
                            <button
                                onClick={() => setActiveLightboxAsset(null)}
                                className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                            >
                                <X size={18} />
                            </button>
                        </div>

                        <div className="p-6 overflow-y-auto flex-1 grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                            <div className="md:col-span-7 bg-black rounded-2xl overflow-hidden flex items-center justify-center max-h-[500px]">
                                <img
                                    src={coverApi.getCoverUrl(activeLightboxAsset.image_url)}
                                    alt={activeLightboxAsset.title || activeLightboxAsset.prompt}
                                    className="max-h-[500px] w-auto object-contain"
                                />
                            </div>

                            <div className="md:col-span-5 space-y-4">
                                <div>
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                                        Prompt
                                    </span>
                                    <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed bg-black/[0.03] dark:bg-white/[0.03] p-3 rounded-xl border border-black/[0.05] dark:border-white/5">
                                        {activeLightboxAsset.prompt}
                                    </p>
                                </div>

                                {activeLightboxAsset.negative_prompt && (
                                    <div>
                                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                                            Negative Prompt
                                        </span>
                                        <p className="text-xs text-slate-500 font-mono bg-black/[0.03] dark:bg-white/[0.03] p-2.5 rounded-xl">
                                            {activeLightboxAsset.negative_prompt}
                                        </p>
                                    </div>
                                )}

                                <div className="space-y-1.5 text-xs">
                                    <div className="flex justify-between py-1 border-b border-black/[0.04] dark:border-white/[0.06]">
                                        <span className="text-slate-400">Style Preset</span>
                                        <span className="font-semibold text-slate-800 dark:text-slate-200">{activeLightboxAsset.style || 'Custom'}</span>
                                    </div>
                                    <div className="flex justify-between py-1 border-b border-black/[0.04] dark:border-white/[0.06]">
                                        <span className="text-slate-400">Role</span>
                                        <span className="font-semibold text-teal-600 dark:text-teal-400">{activeLightboxAsset.asset_type}</span>
                                    </div>
                                    <div className="flex justify-between py-1 border-b border-black/[0.04] dark:border-white/[0.06]">
                                        <span className="text-slate-400">Engine / Architecture</span>
                                        <span className="font-mono text-slate-600 dark:text-slate-400">FLUX.2 / MLX Local</span>
                                    </div>
                                </div>

                                <div className="pt-2 space-y-2">
                                    <button
                                        onClick={() => {
                                            setTargetTrackModalAsset(activeLightboxAsset);
                                            setActiveLightboxAsset(null);
                                        }}
                                        className="w-full py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center justify-center gap-1.5 shadow-md shadow-teal-500/20"
                                    >
                                        <Disc size={15} />
                                        <span>Assign as Album Cover to Track…</span>
                                    </button>

                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={(e) => handleCopyPrompt(activeLightboxAsset, e)}
                                            className="flex-1 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-center gap-1.5 border border-black/[0.06] dark:border-white/10"
                                        >
                                            <Copy size={13} />
                                            <span>Copy Prompt</span>
                                        </button>
                                        <a
                                            href={coverApi.getCoverUrl(activeLightboxAsset.image_url)}
                                            download={`${activeLightboxAsset.title || 'milimo_artwork'}.png`}
                                            className="flex-1 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-center gap-1.5 border border-black/[0.06] dark:border-white/10"
                                        >
                                            <Download size={13} />
                                            <span>Download</span>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Target Track Selection Modal */}
            {targetTrackModalAsset && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/75 backdrop-blur-md animate-fade-in"
                    onClick={() => setTargetTrackModalAsset(null)}
                >
                    <div
                        className="bg-white dark:bg-[#141624] border border-black/10 dark:border-white/10 rounded-3xl max-w-lg w-full p-6 shadow-2xl space-y-4"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                                <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center border border-teal-500/20">
                                    <Disc size={18} />
                                </div>
                                <div>
                                    <h3 className="text-base font-bold text-slate-900 dark:text-white">
                                        Assign Album Cover
                                    </h3>
                                    <p className="text-xs text-slate-500 dark:text-slate-400">
                                        Choose which track should use this artwork
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={() => setTargetTrackModalAsset(null)}
                                className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        <div className="max-h-72 overflow-y-auto space-y-2 pr-1">
                            {songs.length === 0 ? (
                                <p className="text-xs text-slate-400 text-center py-4">No tracks found in library.</p>
                            ) : (
                                songs.map((s) => (
                                    <button
                                        key={s.id}
                                        onClick={() => handleSetAsSongCover(targetTrackModalAsset, s.id)}
                                        className="w-full text-left p-3 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] hover:bg-teal-500/10 dark:hover:bg-teal-500/15 border border-black/[0.05] dark:border-white/5 hover:border-teal-500/30 transition-all flex items-center justify-between group"
                                    >
                                        <div className="min-w-0 pr-2">
                                            <p className="text-xs font-bold text-slate-800 dark:text-slate-200 group-hover:text-teal-600 dark:group-hover:text-teal-300 truncate">
                                                {s.title || s.prompt.slice(0, 30)}
                                            </p>
                                            <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                                                {s.tags?.split(',')[0] || 'Pop'} · ID: {s.id.slice(0, 8)}
                                            </p>
                                        </div>
                                        <span className="text-[11px] font-bold text-teal-600 dark:text-teal-400 opacity-0 group-hover:opacity-100 transition-opacity">
                                            Apply Cover →
                                        </span>
                                    </button>
                                ))
                            )}
                        </div>

                        <div className="flex justify-end pt-2">
                            <button
                                onClick={() => setTargetTrackModalAsset(null)}
                                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 dark:text-slate-400 hover:bg-black/5 dark:hover:bg-white/10"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
