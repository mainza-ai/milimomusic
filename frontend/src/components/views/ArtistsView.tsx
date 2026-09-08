import { toast } from '../../utils/toast';
import { API_BASE_URL } from '../../api';
import React, { useState, useEffect, useRef } from 'react';
import {
    Users, Plus, ArrowLeft, Trash2, Save, Loader2, Sparkles,
    Mic2, UserCog, Disc3, CheckCircle2, AlertTriangle, X, Copy, History,
    Upload, Tag, BookOpen, ChevronDown, ChevronRight
} from 'lucide-react';
import {
    agentsApi, profilesApi, albumApi, coverApi, api, releaseApi, projectApi, styleApi, voiceApi,
    type ReleaseTracksT, type ReleaseTrackT, type Job,
    type AgentInfo, type ArtistProfileT, type ProfileStats, type AgentRunRow, type Project, type Style, type VoiceProfile, type RunStats,
    type ProfileDetail, type ExperiencerVision, type AgentRunEnvelope
} from '../../api';
import { useValidatedForm } from '../../hooks/useValidatedForm';
import { useAudioEngine } from '../../context/AudioEngineContext';
import { StaticWaveform } from '../ui/StaticWaveform';
import { Modal } from '../ui/primitives';

const ROLES = ['world_builder', 'experiencer', 'songwriter', 'producer'];
const ROLE_LABELS: Record<string, string> = {
    world_builder: 'World Builder', experiencer: 'Experiencer',
    songwriter: 'Songwriter', producer: 'Producer', stylist: 'Stylist', critic: 'Critic',
};
const LLM_PROVIDERS = ['opencode', 'nvidia', 'deepseek', 'openai', 'gemini', 'openrouter', 'omlx', 'ollama', 'lmstudio'];

type RunPhase = 'idle' | 'running' | 'done' | 'error';

/** Backend datetimes are naive UTC; make them honest local dates (Safari-safe). */
const fmtUtcDate = (ts?: string | null): string => {
    if (!ts) return '';
    try {
        return new Date(ts.replace(' ', 'T').slice(0, 23) + 'Z').toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch { return ts.slice(0, 10); }
};

interface ArtistsViewProps {
    /** Deep-link support: ?view=artists&id=<profile> lands on that artist. */
    initialProfileId?: string | null;
}

export const ArtistsView: React.FC<ArtistsViewProps> = ({ initialProfileId }) => {
    const [profiles, setProfiles] = useState<ArtistProfileT[]>([]);
    const [stats, setStats] = useState<Record<string, ProfileStats>>({});
    const [isLoadingList, setIsLoadingList] = useState(true);
    const [detail, setDetail] = useState<ProfileDetail | null>(null);
    const [isDetailLoading, setIsDetailLoading] = useState(false);
    // Guided create (A1): 4-step modal — identity, bio, tags, cover.
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [createStep, setCreateStep] = useState(0);
    const [createProjectId, setCreateProjectId] = useState('');
    const [createCover, setCreateCover] = useState<File | null>(null);
    const [createCoverPreview, setCreateCoverPreview] = useState('');
    const [createBusy, setCreateBusy] = useState(false);
    const [projects, setProjects] = useState<Project[]>([]);
    const [styleChips, setStyleChips] = useState<Style[]>([]);
    const createForm = useValidatedForm({ name: '', bio: '', tags: '' }, {
        name: v => (!v.trim() ? 'Give the artist a name.' : v.trim().length < 2 ? 'At least 2 characters.' : null),
        bio: v => (v.trim().length > 0 && v.trim().length < 20
            ? 'Give the bio a little more for the crew to ground on (20+ chars).' : null),
        tags: () => null,
    });
    const [agentsRegistry, setAgentsRegistry] = useState<AgentInfo[]>([]);

    // identity edit state (D3: shared validated form pattern)
    const identityForm = useValidatedForm({ name: '', bio: '', tags: '' }, {
        name: v => (!v.trim() ? 'Name cannot be empty.' : v.trim().length < 2 ? 'At least 2 characters.' : null),
        bio: () => null,
        tags: () => null,
    });
    const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
    const [saveError, setSaveError] = useState('');
    // A1: artist voice identity — singing voice applied to every album track
    const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([]);
    const [voiceSaving, setVoiceSaving] = useState(false);

    const handleChangeVoice = async (voiceId: string) => {
        if (!detail) return;
        setVoiceSaving(true);
        try {
            const updated = await profilesApi.update(detail.profile.id, { voice_profile_id: voiceId || null });
            setDetail({ ...detail, profile: updated });
            setProfiles(prev => prev.map(p => p.id === updated.id ? updated : p));
            toast(voiceId ? 'Singing voice linked — future album tracks will use it.' : 'Voice unlinked — future tracks use the provider default.', 'success');
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Voice update failed'), 'error');
        } finally {
            setVoiceSaving(false);
        }
    };

    // crew add row
    const [crewRole, setCrewRole] = useState('experiencer');
    const [crewAgent, setCrewAgent] = useState('experiencer');
    // Per-assignment model override (item #9): assignment > artist default > global.
    const [crewProvider, setCrewProvider] = useState('');
    const [crewModel, setCrewModel] = useState('');

    // release create
    const [newReleaseTitle, setNewReleaseTitle] = useState('');

    // Album production run
    interface AlbumRunState {
        runId: string;
        releaseTitle: string;
        status: string;
        progress: number;
        message: string;
    }
    const [albumRun, setAlbumRun] = useState<AlbumRunState | null>(null);
    const albumRunIdRef = useRef<string | null>(null);
    const [autopilot, setAutopilot] = useState(false);
    // B3: wall-clock budget cap handed to the orchestrator (off = uncapped)
    const [budgetMin, setBudgetMin] = useState<'off' | '15' | '30' | '60'>('off');
    // 3A: optional crew agents for this run (cost control — default off)
    const [crewStylist, setCrewStylist] = useState(false);
    const [crewCritic, setCrewCritic] = useState(false);

    const startAlbum = async (releaseId: string, title: string) => {
        try {
            const budget = budgetMin === 'off' ? undefined : { deadline_s: Number(budgetMin) * 60 };
            const res = await albumApi.produce(releaseId, autopilot, {
                budget,
                crew: { stylist: crewStylist, critic: crewCritic },
            });
            albumRunIdRef.current = res.run_id;
            setAlbumRun({ runId: res.run_id, releaseTitle: title, status: 'queued', progress: 0, message: 'Imagining the journey…' });
        } catch (e: any) {
            setAlbumRun({ runId: '-', releaseTitle: title, status: 'failed', progress: 0, message: '' });
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Could not start album run'), 'error');
        }
    };

    const retryTrack = async (releaseId: string, jobId: string, title: string) => {
        try {
            const res = await releaseApi.retryTrack(releaseId, jobId);
            albumRunIdRef.current = res.run_id;
            setAlbumRun({ runId: res.run_id, releaseTitle: title || 'Track', status: 'queued', progress: 0, message: 'Reproducing failed track…' });
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Retry failed'), 'error');
        }
    };

    const detachTrack = async (releaseId: string, jobId: string, title: string) => {
        if (!confirm(`Detach "${title}" from this release? The audio file will remain in your project.`)) return;
        try {
            await releaseApi.detachTrack(releaseId, jobId);
            toast(`Detached "${title}"`, 'success');
            if (openTracks === releaseId) {
                try { setTracks(await profilesApi.getReleaseTracks(releaseId)); } catch { /* optimistic */ }
            }
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Detach failed'), 'error');
        }
    };

    const approveNextTrack = async () => {
        if (!albumRun || albumRun.runId === '-') return;
        setAlbumRun(r => r ? { ...r, status: 'running', message: 'Approved — producing next track…' } : r);
        await albumApi.resume(albumRun.runId).catch((e) => { toast(String(e?.response?.data?.detail?.error?.message || e?.message || "Request failed"), "error"); return null; });
    };

    const cancelAlbum = async () => {
        if (!albumRun || albumRun.runId === '-') return;
        await albumApi.cancelRun(albumRun.runId).catch((e) => { toast(String(e?.response?.data?.detail?.error?.message || e?.message || "Request failed"), "error"); return null; });
        setAlbumRun(r => r ? { ...r, status: 'cancelling', message: 'Cancelling…' } : r);
    };

    // Experiencer run
    const [briefTitle, setBriefTitle] = useState('');
    const [briefConcept, setBriefConcept] = useState('');
    const [briefTarget, setBriefTarget] = useState(10);
    const [briefDirection, setBriefDirection] = useState('');
    const [runPhase, setRunPhase] = useState<RunPhase>('idle');
    const [runError, setRunError] = useState<string>('');
    const [runStage, setRunStage] = useState('');
    const [vision, setVision] = useState<ExperiencerVision | null>(null);
    const [elapsed, setElapsed] = useState(0);
    const elapsedTimer = useRef<number | undefined>(undefined);
    const expRunIdRef = useRef<string | null>(null);

    useEffect(() => {
        agentsApi.listAgents().then(setAgentsRegistry).catch(console.error);
    }, []);

    const openProfile = async (id: string) => {
        setIsDetailLoading(true);
        syncArtistUrl(id);
        try {
            const d = await profilesApi.get(id);
            setDetail(d);
            identityForm.setAll({ name: d.profile.name, bio: d.profile.bio, tags: d.profile.tags });
            setVision(null);
            setRunPhase('idle');
            setRunError('');
            albumRunIdRef.current = null;
            setAlbumRun(null);
            // Run history (C5) + aggregates (3D) + active-run recovery.
            agentsApi.listRuns(d.profile.id, 50).then(async ({ runs }) => {
                setRunHistory(runs.slice(0, 10));
                const active = envelope_active(runs);
                if (!active) return;
                const full = await albumApi.getRun(active.id).catch(() => null);
                let releaseTitle = 'This album';
                try {
                    const cfg = JSON.parse(full?.input_json || '{}');
                    releaseTitle = d.releases.find(r => r.id === cfg.release_id)?.title || releaseTitle;
                } catch { /* input_json unreadable — generic title */ }
                albumRunIdRef.current = active.id;
                setAlbumRun({
                    runId: active.id, releaseTitle,
                    status: active.status, progress: active.progress || 0,
                    message: 'Recovered an in-flight album run.',
                });
            }).catch(console.error);
            agentsApi.runStats(d.profile.id).then(setRunStats).catch(console.error);
        } finally {
            setIsDetailLoading(false);
        }
    };

    // A4: cover art generation from lore/tags — procedural covers endpoint.
    const [coverGenBusy, setCoverGenBusy] = useState<'profile' | string | null>(null);
    const generateCover = async (target: 'profile' | string) => {
        if (!detail) return;
        setCoverGenBusy(target);
        try {
            let prompt: string;
            if (target === 'profile') {
                let loreBits = '';
                try {
                    const lore = JSON.parse(detail.profile.lore_json || '{}');
                    loreBits = [lore.era_setting, lore.appearance].filter(Boolean).join(', ');
                } catch { /* raw/freeform lore — not usable as a structured prompt */ }
                prompt = `High-end artistic album cover art for the artist '${detail.profile.name}'`
                    + (loreBits ? `, ${loreBits}` : '')
                    + (detail.profile.tags ? `, style: ${detail.profile.tags}` : '')
                    + ', minimalist, cinematic lighting, modern abstract aesthetics, award-winning graphic design';
            } else {
                const rel = detail.releases.find(r => r.id === target);
                prompt = `High-end artistic album cover art for the release '${rel?.title || 'Untitled'}', by ${detail.profile.name}`
                    + (detail.profile.tags ? `, style: ${detail.profile.tags}` : '')
                    + ', minimalist, cinematic lighting, modern abstract aesthetics, award-winning graphic design';
            }
            const { url } = await coverApi.generateCoverImage({ prompt });
            if (target === 'profile') {
                const updated = await profilesApi.setCover(detail.profile.id, url);
                setDetail({ ...detail, profile: updated });
                setProfiles(prev => prev.map(p => p.id === updated.id ? updated : p));
            } else {
                const updated = await releaseApi.update(target, { cover_image_path: url });
                setDetail(d => d ? { ...d, releases: d.releases.map(r => r.id === target ? updated : r) } : d);
            }
            toast('Cover art generated.', 'success');
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Cover generation failed'), 'error');
        } finally {
            setCoverGenBusy(null);
        }
    };

    /** Hand the finished track to the main studio (full player + DAW view). */
    const openInStudio = (jobId: string) => {
        try {
            const url = new URL(window.location.href);
            url.searchParams.set('view', 'track-detail');
            url.searchParams.set('track', jobId);
            window.history.pushState({ view: 'track-detail', trackId: jobId }, '', url.toString());
            window.dispatchEvent(new PopStateEvent('popstate'));
        } catch { /* deep-link best effort */ }
    };

    // Waveform click: seek inside the current track, or start this row's
    // track at the clicked position (single fetch, user-initiated).
    const { playTrack, swapCurrentTrack, currentTrack, currentTime, duration, seek } = useAudioEngine();
    const handleWaveSeek = async (tr: ReleaseTrackT, fraction: number) => {
        try {
            if (currentTrack?.id === tr.id && duration > 0) {
                seek(fraction * duration);
                return;
            }
            const job = await api.getJobStatus(tr.id);
            if (job.audio_path) {
                await playTrack(job, [job], fraction * (tr.duration_ms / 1000));
            }
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Seek failed'), 'error');
        }
    };
    const [playFetchingId, setPlayFetchingId] = useState<string | null>(null);
    const playFromTracklist = async (jobId: string, albumTracks?: ReleaseTrackT[]) => {
        const anchor = albumTracks?.find(t => t.id === jobId);
        const anchorAudio = anchor?.artifacts?.audio;
        if (!anchorAudio) {
            toast('This track has no audio yet.', 'error');
            return;
        }
        // Minimal playable Job for instant start; hydrated below.
        const optimistic = { id: jobId, title: anchor?.title || 'Track', audio_path: anchorAudio } as Job;
        setPlayFetchingId(jobId);
        void playTrack(optimistic, [optimistic]);
        try {
            const job = await api.getJobStatus(jobId);
            if (job.audio_path) {
                let playlistQueue: Job[] = [job];
                if (albumTracks && albumTracks.length > 1) {
                    try {
                        const targetIdx = albumTracks.findIndex(t => t.id === jobId);
                        const ordered = targetIdx >= 0 ? [...albumTracks.slice(targetIdx), ...albumTracks.slice(0, targetIdx)] : albumTracks;
                        const otherTracks = ordered.filter(t => t.id !== jobId && t.status === 'completed' && t.artifacts?.audio);
                        const fetched = await Promise.all(otherTracks.map(t => api.getJobStatus(t.id).catch(() => null)));
                        const valid = fetched.filter((j): j is Job => j !== null && !!j.audio_path);
                        playlistQueue = [job, ...valid];
                    } catch {
                        playlistQueue = [job];
                    }
                }
                // Swap the optimistic entry for the hydrated Job in place —
                // no element restart, no stealing back a user pause.
                swapCurrentTrack(job, playlistQueue);
            } else {
                toast('This track has no audio yet.', 'error');
            }
        } catch (e: any) {
            // Optimistic playback already started; only the hydration failed.
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Track details failed to load (playing preview)'), 'error');
        } finally {
            setPlayFetchingId(null);
        }
    };

    /** The album run a reload should reattach to, if any. */
    const envelope_active = (runs: AgentRunRow[]) => runs.find(r => r.agent_name === 'album_orchestrator'
        && ['queued', 'running', 'awaiting_approval'].includes(r.status));

    const refreshDetail = () => detail && openProfile(detail.profile.id);    const openCreateModal = () => {
        setCreateStep(0);
        createForm.reset();
        setCreateProjectId('');
        setCreateCover(null);
        setCreateCoverPreview('');
        setIsCreateOpen(true);
        projectApi.listProjects().then(setProjects).catch(console.error);
        styleApi.getStyles().then(s => setStyleChips(s.filter(st => st.type !== 'trained').slice(0, 10))).catch(console.error);
    };

    const toggleChip = (chip: string) => {
        const current = createForm.values.tags;
        const parts = current.split(',').map(s => s.trim()).filter(Boolean);
        const idx = parts.findIndex(s => s.toLowerCase() === chip.toLowerCase());
        if (idx >= 0) parts.splice(idx, 1);
        else parts.push(chip);
        createForm.setField('tags', parts.join(', '));
    };

    const submitCreate = async () => {
        createForm.markSubmitAttempted();
        if (!createForm.values.name.trim()) { setCreateStep(0); return; }
        setCreateBusy(true);
        try {
            let p = await profilesApi.create({
                name: createForm.values.name.trim(),
                bio: createForm.values.bio.trim(),
                tags: createForm.values.tags.trim(),
                ...(createProjectId ? { project_id: createProjectId } : {}),
            });
            if (createCover) {
                try {
                    const { url } = await coverApi.uploadCoverImage(createCover);
                    p = await profilesApi.setCover(p.id, url);
                } catch {
                    toast('Artist created, but the cover upload failed — add it from their page.', 'error');
                }
            }
            setProfiles(prev => [p, ...prev]);
            setIsCreateOpen(false);
            createForm.reset();
            setCreateStep(0);
            setCreateCover(null);
            setCreateCoverPreview('');
            setCreateProjectId('');
            toast(`${p.name} created.`, 'success');
            loadArtists({ page: 0, sortBy, q: debouncedSearch }); // new artist is newest — page 0
            openProfile(p.id);
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Create failed'), 'error');
        } finally {
            setCreateBusy(false);
        }
    };

    const [openTracks, setOpenTracks] = useState<string | null>(null);
    const [tracks, setTracks] = useState<ReleaseTracksT | null>(null);
    const [openReleaseMeta, setOpenReleaseMeta] = useState<{ track_total: number; active_run: boolean } | null>(null);
    const toggleTracks = async (rid: string) => {
        if (openTracks === rid) { setOpenTracks(null); setTracks(null); setOpenReleaseMeta(null); return; }
        setOpenTracks(rid); setTracks(null); setOpenReleaseMeta(null);
        try { setTracks(await profilesApi.getReleaseTracks(rid)); }
        catch (e: any) { toast(String(e?.response?.data?.detail?.error?.message || 'Failed to load tracks'), 'error'); }
        // Live release meta (single-fetch): surfaces a running album run.
        try {
            const meta = await releaseApi.get(rid);
            setOpenReleaseMeta({ track_total: meta.track_total, active_run: meta.active_run });
        } catch { /* header already shows tracks payload; meta is enhancement */ }
    };
    // B2: curate the tracklist order (optimistic move, honest revert on failure).
    // Tombstone rows (status 'missing', synthetic ids) are display-only and
    // never submitted — the server 422s unknown track ids.
    const moveTrack = async (index: number, dir: -1 | 1) => {
        if (!tracks || !openTracks) return;
        const arr = [...tracks.tracks];
        const j = index + dir;
        if (j < 0 || j >= arr.length) return;
        [arr[index], arr[j]] = [arr[j], arr[index]];
        setTracks({ ...tracks, tracks: arr });
        try {
            await releaseApi.setTrackOrder(openTracks, arr.filter(t => t.status !== 'missing').map(t => t.id));
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Reorder failed'), 'error');
            try { setTracks(await profilesApi.getReleaseTracks(openTracks)); } catch { /* keep optimistic view */ }
        }
    };
    const [search, setSearch] = useState('');
    const [sortBy, setSortBy] = useState<'activity' | 'name'>('activity');
    const [runHistory, setRunHistory] = useState<AgentRunRow[]>([]);
    const [runStats, setRunStats] = useState<RunStats | null>(null);
    // 3C: server-side search + pagination — the URL is the only truth for filtering
    const PAGE_SIZE = 24;
    const [page, setPage] = useState(0);
    const [total, setTotal] = useState(0);
    const [debouncedSearch, setDebouncedSearch] = useState('');
    useEffect(() => {
        const t = window.setTimeout(() => {
            setDebouncedSearch(search);
            setPage(0); // a new query restarts at the first page
        }, 250);
        return () => window.clearTimeout(t);
    }, [search]);

    const loadArtists = React.useCallback(async (opts: { page: number; sortBy: 'activity' | 'name'; q: string }) => {
        setIsLoadingList(true);
        try {
            const data = await profilesApi.list({
                withStats: true, limit: PAGE_SIZE, offset: opts.page * PAGE_SIZE,
                q: opts.q || undefined,
            });
            setProfiles(data.profiles);
            setStats(data.stats || {});
            setTotal(data.total);
            if (sortBy === 'name') data.profiles.sort((a, b) => a.name.localeCompare(b.name));
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Could not load artists'), 'error');
        } finally {
            setIsLoadingList(false);
        }
    }, [sortBy]);

    useEffect(() => {
        loadArtists({ page, sortBy, q: debouncedSearch });
    }, [page, sortBy, debouncedSearch, loadArtists]);

    const visibleProfiles = profiles; // server already filtered + paginated

    // ── Deep-link + URL truth (C6): the URL always reflects the open artist ──
    const syncArtistUrl = (profileId: string | null) => {
        try {
            const url = new URL(window.location.href);
            if (profileId) url.searchParams.set('id', profileId);
            else url.searchParams.delete('id');
            window.history.replaceState({}, '', url.toString());
        } catch { /* URL sync is best-effort */ }
    };

    const prevDeepLinkRef = useRef<string | null | undefined>(undefined);
    useEffect(() => {
        voiceApi.listProfiles().then(setVoiceProfiles).catch(console.error);
    }, []);
    useEffect(() => {
        const prev = prevDeepLinkRef.current;
        prevDeepLinkRef.current = initialProfileId ?? null;
        if (initialProfileId && (!detail || detail.profile.id !== initialProfileId)) {
            openProfile(initialProfileId);
        } else if (typeof prev === 'string' && initialProfileId === null && detail) {
            // Popstate back to the artists list — the URL no longer carries the
            // id, so the detail must close to match (dirty-guard still applies).
            if (!identityDirty || window.confirm('You have unsaved identity changes. Discard them?')) {
                setDetail(null);
            } else {
                syncArtistUrl(detail.profile.id);
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [initialProfileId]);

    const closeDetail = () => {
        // Dirty-guard (C3): identity edits must not vanish silently.
        if (identityDirty && !window.confirm('You have unsaved identity changes. Discard them?')) return;
        setDetail(null);
        syncArtistUrl(null);
    };
    const [coverBusy, setCoverBusy] = useState(false);
    const uploadCover = async (file: File) => {
        if (!detail) return;
        setCoverBusy(true);
        try {
            const { url } = await coverApi.uploadCoverImage(file);
            const updated = await profilesApi.setCover(detail.profile.id, url);
            setDetail({ ...detail, profile: updated });
            setProfiles(prev => prev.map(pp => pp.id === updated.id ? updated : pp));
            toast("Artist identity image updated.", "success");
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || "Cover upload failed"), "error");
        } finally {
            setCoverBusy(false);
        }
    };

    const identityDirty = !!detail && (
        identityForm.values.name !== detail.profile.name || identityForm.values.bio !== detail.profile.bio || identityForm.values.tags !== detail.profile.tags);
    useEffect(() => {
        if (!identityDirty) return;
        const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ''; };
        window.addEventListener('beforeunload', warn);
        return () => window.removeEventListener('beforeunload', warn);
    }, [identityDirty]);

    const handleSaveIdentity = async () => {
        if (!detail || !identityForm.isValid) return;
        setSaveState('saving');
        setSaveError('');
        try {
            const updated = await profilesApi.update(detail.profile.id, {
                name: identityForm.values.name.trim(), bio: identityForm.values.bio, tags: identityForm.values.tags
            });
            setDetail({ ...detail, profile: updated });
            setProfiles(prev => prev.map(p => p.id === updated.id ? updated : p));
            setSaveState('saved');
            setTimeout(() => setSaveState(s => (s === 'saved' ? 'idle' : s)), 2000);
        } catch (e: any) {
            setSaveState('error');
            setSaveError(String(e?.response?.data?.detail?.error?.message || e?.message || 'Save failed — check your connection and retry.'));
        }
    };

    const handleDeleteProfile = async () => {
        if (!detail) return;
        if (!window.confirm(`Delete artist "${detail.profile.name}"? Their crew is removed and album containers are deleted; finished tracks remain in Explore as standalone tracks.`)) return;
        try {
            await profilesApi.delete(detail.profile.id);
            setDetail(null);
            syncArtistUrl(null);
            loadArtists({ page, sortBy, q: debouncedSearch }); // refetch current page
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Delete failed'), 'error');
        }
    };

    const addCrewMember = async () => {
        if (!detail) return;
        if (detail.assignments.some(a => a.role === crewRole)) {
            toast(String(`This artist already has a ${crewRole} assigned. Remove them first.`), "error");
            return;
        }
        try {
            await profilesApi.setAssignments(detail.profile.id, [
                ...detail.assignments.map(a => ({ role: a.role, agent_name: a.agent_name })),
                {
                    role: crewRole, agent_name: crewAgent,
                    ...(crewProvider ? {
                        model_provider: crewProvider,
                        ...(crewModel.trim() ? { model: crewModel.trim() } : {}),
                    } : {}),
                },
            ]);
            setCrewProvider('');
            setCrewModel('');
            refreshDetail();
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Could not add crew member'), 'error');
        }
    };

    const removeCrewMember = async (assignmentId: string) => {
        if (!detail) return;
        const remaining = detail.assignments.filter(a => a.id !== assignmentId)
            .map(a => ({ role: a.role, agent_name: a.agent_name }));
        try {
            const updated = await profilesApi.setAssignments(detail.profile.id, remaining);
            setDetail({ ...detail, assignments: updated });
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Could not remove crew member'), 'error');
        }
    };

    // ── Experiencer run ────────────────────────────────────────────────────
    // Live stage text for experiencer runs via SSE run_progress events.
    // Filtered to THIS run's id — concurrent album/agent runs must not
    // corrupt the stage readout.
    useEffect(() => {
        if (runPhase !== 'running') return;
        const es = api.connectToEvents((event: MessageEvent) => {
            try {
                const d = JSON.parse(event.data);
                if (expRunIdRef.current && d.run_id !== expRunIdRef.current) return;
                if (d.message) setRunStage(String(d.message));
            } catch { /* non-JSON frame */ }
        }, ['run_progress']);
        return () => es.close();
    }, [runPhase]);

    // Album run: live events + polling fallback.
    useEffect(() => {
        const es = api.connectToEvents((event: MessageEvent) => {
            try {
                const d = JSON.parse(event.data);
                if (!albumRunIdRef.current || d.run_id !== albumRunIdRef.current) return;
                setAlbumRun(prev => prev ? ({
                    ...prev,
                    status: ['done', 'failed', 'cancelled', 'budget_exceeded'].includes(d.phase)
                        ? d.phase : (d.phase === 'awaiting_approval' ? 'awaiting_approval' : prev.status),
                    progress: typeof d.progress === 'number' ? d.progress : prev.progress,
                    message: d.message ? String(d.message) : prev.message,
                }) : prev);
            } catch { /* non-JSON frame */ }
        }, ['run_progress', 'run_update']);
        return () => es.close();
    }, []);

    const albumActive = !!albumRun && ['queued', 'running', 'awaiting_approval', 'cancelling'].includes(albumRun.status);
    useEffect(() => {
        if (!albumActive || !albumRun || albumRun.runId === '-') return;
        const t = setInterval(async () => {
            const row = await albumApi.getRun(albumRun.runId).catch((e) => { toast(String(e?.response?.data?.detail?.error?.message || e?.message || "Request failed"), "error"); return null; });
            if (row) setAlbumRun(prev => prev ? ({ ...prev, status: row.status, progress: row.progress ?? prev.progress }) : prev);
        }, 5000);
        return () => clearInterval(t);
    }, [albumActive, albumRun?.runId]);

    // B4: while a run is live, keep release rows honest (planned → in_progress → completed)
    useEffect(() => {
        if (!albumActive || !detail) return;
        const pid = detail.profile.id;
        const t = setInterval(async () => {
            try {
                const data = await releaseApi.list(pid);
                setDetail(d => d ? { ...d, releases: data.releases } : d);
            } catch { /* transient poll error — next tick retries */ }
        }, 8000);
        return () => clearInterval(t);
    }, [albumActive, detail?.profile.id]);

    const startRunTimer = () => {
        setElapsed(0);
        window.clearInterval(elapsedTimer.current);
        elapsedTimer.current = window.setInterval(() => setElapsed(s => s + 1), 1000);
    };
    const stopRunTimer = () => window.clearInterval(elapsedTimer.current);

    const runExperiencer = async () => {
        if (!detail || !briefTitle.trim() || !briefConcept.trim()) return;
        setRunPhase('running');
        setRunError('');
        setVision(null);
        startRunTimer();
        try {
            const envelope: AgentRunEnvelope = await agentsApi.runAgent('experiencer', {
                input: {
                    album_title: briefTitle.trim(),
                    album_concept: briefConcept.trim(),
                    artist_name: detail.profile.name,
                    artist_bio: detail.profile.bio,
                    tags: detail.profile.tags,
                    track_target: briefTarget,
                    extra_direction: briefDirection.trim(),
                },
                profile_id: detail.profile.id,
            });
            expRunIdRef.current = envelope.run.id;
            const out = JSON.parse(envelope.run.output_json).output as ExperiencerVision;
            setVision(out);
            setRunPhase('done');
        } catch (e: unknown) {
            const err = e as { response?: { data?: { detail?: { error?: { message?: string; attempts?: { provider: string; error_type: string }[] } } } } };
            const errBody = err.response?.data?.detail?.error;
            let msg = errBody?.message || 'Agent run failed.';
            if (errBody?.attempts?.length) {
                msg += '\nAttempts: ' + errBody.attempts.map(a => `${a.provider} (${a.error_type})`).join(', ');
            }
            setRunError(msg);
            setRunPhase('error');
        } finally {
            stopRunTimer();
        }
    };

    const saveVisionAsRelease = async () => {
        if (!detail || !vision) return;
        try {
            await profilesApi.createRelease({
                profile_id: detail.profile.id,
                title: vision.journey_title,
                description: vision.concept_statement,
                vision: vision as unknown as Record<string, unknown>,
            });
            toast('Vision saved as a release — producing it will use these exact seeds.', 'success');
            refreshDetail();
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Could not save release'), 'error');
        }
    };

    // World lore edit state (F10)
    const [editLore, setEditLore] = useState('');
    const [loreSaving, setLoreSaving] = useState(false);
    const [loreGenerating, setLoreGenerating] = useState(false);
    const loreBaselineRef = useRef('');
    useEffect(() => {
        if (detail) {
            let next = '';
            try {
                const raw = detail.profile.lore_json || '{}';
                const parsed = JSON.parse(raw);
                next = parsed && typeof parsed === 'object' ? JSON.stringify(parsed, null, 2) : String(raw);
            } catch {
                next = String(detail.profile.lore_json || '');
            }
            setEditLore(next);
            loreBaselineRef.current = next;
        }
    }, [detail?.profile.id, detail?.profile.lore_json]);

    const handleSaveLore = async () => {
        if (!detail) return;
        setLoreSaving(true);
        try {
            const updated = await profilesApi.update(detail.profile.id, { lore_json: editLore });
            setDetail({ ...detail, profile: updated });
            loreBaselineRef.current = editLore;
            toast('World lore saved — the crew will ground on it.', 'success');
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Lore save failed'), 'error');
        } finally {
            setLoreSaving(false);
        }
    };

    const handleGenerateLore = async () => {
        if (!detail) return;
        if (editLore !== loreBaselineRef.current && !window.confirm('Regenerating will replace the lore — including your unsaved edits. Continue?')) return;
        setLoreGenerating(true);
        try {
            const res = await profilesApi.generateLore(detail.profile.id);
            const pretty = JSON.stringify(res.lore, null, 2);
            setEditLore(pretty);
            loreBaselineRef.current = pretty;
            setDetail(d => d ? { ...d, profile: res.profile } : d);
            toast('World lore generated — review and save it.', 'success');
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Lore generation failed'), 'error');
        } finally {
            setLoreGenerating(false);
        }
    };

    // Release edit state (F8: full release lifecycle in the UI)
    const [editReleaseId, setEditReleaseId] = useState<string | null>(null);
    const [editReleaseTitle, setEditReleaseTitle] = useState('');
    const [editReleaseDesc, setEditReleaseDesc] = useState('');

    const handleRenameRelease = async (rid: string) => {
        if (!editReleaseTitle.trim()) return;
        try {
            const updated = await releaseApi.update(rid, {
                title: editReleaseTitle.trim(),
                description: editReleaseDesc.trim(),
            });
            setDetail(d => d ? { ...d, releases: d.releases.map(r => r.id === rid ? updated : r) } : d);
            setEditReleaseId(null);
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Rename failed'), 'error');
        }
    };

    const handleDeleteRelease = async (rid: string, title: string) => {
        if (!window.confirm(`Delete release "${title}"? Its finished tracks remain in Explore as standalone tracks.`)) return;
        try {
            await releaseApi.delete(rid);
            setDetail(d => d ? { ...d, releases: d.releases.filter(r => r.id !== rid) } : d);
            if (openTracks === rid) { setOpenTracks(null); setTracks(null); }
            toast('Release deleted.', 'success');
        } catch (e: any) {
            toast(String(e?.response?.data?.detail?.error?.message || e?.message || 'Delete failed'), 'error');
        }
    };

    if (isDetailLoading) {
        return <div className="flex-1 flex items-center justify-center"><Loader2 size={22} className="animate-spin text-teal-500" /></div>;
    }

    // ── LIST MODE ──────────────────────────────────────────────────────────
    if (!detail) {
        return (
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 max-w-6xl mx-auto w-full animate-fade-in">
                <div className="flex items-center justify-between gap-4 mb-6">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 border border-teal-500/20 flex items-center justify-center">
                            <Users size={18} className="text-teal-600 dark:text-teal-400" />
                        </div>
                        <div>
                            <h1 className="text-xl font-extrabold tracking-tight text-slate-900 dark:text-white">Artist Profiles</h1>
                            <p className="text-xs text-slate-500 dark:text-slate-400">Each artist carries their own identity and AI crew.</p>
                        </div>
                    </div>
                    <button
                        onClick={openCreateModal}
                        className="px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all"
                    >
                        <Plus size={14} /> New Artist
                    </button>
                </div>

                {isLoadingList ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4" role="list" aria-label="Loading artists">
                        {[0, 1, 2, 3, 4, 5].map(i => (
                            <div key={i} className="p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] space-y-3" role="presentation">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-slate-200 dark:bg-white/10 animate-pulse" />
                                    <div className="h-3.5 w-24 rounded bg-slate-200 dark:bg-white/10 animate-pulse" />
                                </div>
                                <div className="h-2.5 w-full rounded bg-slate-200 dark:bg-white/10 animate-pulse" />
                                <div className="h-2.5 w-2/3 rounded bg-slate-200 dark:bg-white/10 animate-pulse" />
                            </div>
                        ))}
                    </div>
                ) : profiles.length === 0 && debouncedSearch ? (
                    <p className="text-xs text-slate-500 italic py-8 text-center">No artists match “{debouncedSearch}”.</p>
                ) : profiles.length === 0 ? (
                    <div className="py-20 text-center space-y-3">
                        <Users size={36} className="mx-auto text-slate-300 dark:text-slate-600" />
                        <p className="text-sm font-bold text-slate-700 dark:text-slate-200">No artists yet</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto">
                            Create an artist profile, assign their AI crew, and let the Experiencer imagine their first album.
                        </p>
                        <button onClick={openCreateModal}
                            className="mt-1 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all">
                            <Plus size={14} /> Create your first artist
                        </button>
                    </div>
                ) : (
                    <div>
                        <div className="flex items-center gap-2 mb-3">
                            <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search artists, styles…" className="apple-input text-xs flex-1" aria-label="Search artists" />
                            <select value={sortBy} onChange={e => setSortBy(e.target.value as 'activity' | 'name')}
                                className="apple-input !py-1.5 !px-2 text-[11px] font-mono w-auto" aria-label="Sort artists">
                                <option value="activity">Recent activity</option>
                                <option value="name">Name</option>
                            </select>
                        </div>
                        {visibleProfiles.length === 0 ? (
                            <p className="text-xs text-slate-500 italic py-8 text-center">No artists match “{search}”.</p>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4" role="list" aria-label="Artist profiles">
                                {visibleProfiles.map(p => (
                                    <button
                                        key={p.id}
                                        role="listitem"
                                        aria-label={`Open artist ${p.name}`}
                                        onClick={() => openProfile(p.id)}
                                        className="text-left p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm hover:shadow-apple-md backdrop-blur-xl transition-all hover:-translate-y-0.5 focus-visible:ring-2 focus-visible:ring-teal-500/60 outline-none"
                                    >
                                        <div className="flex items-center gap-3">
                                            {p.cover_image_path ? (
                                                <img src={`${API_BASE_URL}${p.cover_image_path}`} alt="" className="w-10 h-10 rounded-xl object-cover border border-black/10 dark:border-white/10" />
                                            ) : (
                                                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-400/40 to-fuchsia-500/40 flex items-center justify-center"><UserCog size={16} className="text-slate-500" /></div>
                                            )}
                                            <h3 className="text-sm font-extrabold text-slate-900 dark:text-white truncate">{p.name}</h3>
                                        </div>
                                        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 line-clamp-2 min-h-[2em]">
                                            {p.bio || 'No bio yet.'}
                                        </p>
                                        {p.tags && (
                                            <div className="flex flex-wrap gap-1 mt-2">
                                                {p.tags.split(',').slice(0, 4).map(t => t.trim()).filter(Boolean).map(t => (
                                                    <span key={t} className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20">{t}</span>
                                                ))}
                                            </div>
                                        )}
                                        {stats[p.id] && (
                                            <div className="flex items-center gap-2 mt-2 text-[10px] font-mono text-slate-400">
                                                <span>{stats[p.id].crew_count} crew</span>
                                                <span aria-hidden="true">·</span>
                                                <span>{stats[p.id].release_count} releases</span>
                                                {stats[p.id].last_activity && (
                                                    <>
                                                        <span aria-hidden="true">·</span>
                                                        <span>active {fmtUtcDate(stats[p.id].last_activity)}</span>
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </button>
                                ))}
                            </div>
                        )}
                        {total > PAGE_SIZE && (
                            <div className="flex items-center justify-between mt-4" aria-label="Artist pages">
                                <span className="text-[10px] font-mono text-slate-400">
                                    {page * PAGE_SIZE + 1}–{Math.min(total, (page + 1) * PAGE_SIZE)} of {total}
                                </span>
                                <div className="flex items-center gap-1.5">
                                    <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                                        className="text-[10px] font-bold px-2.5 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-300 hover:bg-black/[0.08] disabled:opacity-40 transition-colors">Prev</button>
                                    <button onClick={() => setPage(p => p + 1)} disabled={(page + 1) * PAGE_SIZE >= total}
                                        className="text-[10px] font-bold px-2.5 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-300 hover:bg-black/[0.08] disabled:opacity-40 transition-colors">Next</button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Guided create stepper (A1): identity → bio → tags → cover */}
                <Modal isOpen={isCreateOpen} onClose={() => { if (!createBusy) setIsCreateOpen(false); }}
                    title="New Artist Profile" widthClass="max-w-md">
                    <div className="p-6 space-y-4">
                            <div className="flex items-center justify-end">
                                <div className="flex items-center gap-1.5" aria-label={`Step ${createStep + 1} of 4`}>
                                    {[0, 1, 2, 3].map(i => (
                                        <span key={i} className={`w-1.5 h-1.5 rounded-full transition-colors ${i === createStep ? 'bg-teal-500 w-4' : i < createStep ? 'bg-teal-500/50' : 'bg-slate-300 dark:bg-slate-600'}`} />
                                    ))}
                                </div>
                            </div>

                            {createStep === 0 && (
                                <div className="space-y-3">
                                    <label className="block">
                                        <span className="text-[10px] font-mono font-bold uppercase text-slate-400 block mb-1">Artist name</span>
                                        <input autoFocus value={createForm.values.name}
                                            onChange={e => createForm.setField('name', e.target.value)}
                                            onBlur={() => createForm.markTouched('name')}
                                            placeholder="e.g. Nalo Rivers" className="apple-input text-sm" />
                                        {createForm.showError('name') && (
                                            <span className="text-[10px] text-rose-500 font-mono mt-1 block" role="alert">{createForm.showError('name')}</span>
                                        )}
                                    </label>
                                    <label className="block">
                                        <span className="text-[10px] font-mono font-bold uppercase text-slate-400 block mb-1">Project (optional)</span>
                                        <select value={createProjectId} onChange={e => setCreateProjectId(e.target.value)}
                                            className="apple-input text-xs" aria-label="Owning project">
                                            <option value="">No project — standalone artist</option>
                                            {projects.map(pr => <option key={pr.id} value={pr.id}>{pr.name}</option>)}
                                        </select>
                                    </label>
                                </div>
                            )}

                            {createStep === 1 && (
                                <div className="space-y-2">
                                    <label className="block">
                                        <span className="flex items-center justify-between mb-1">
                                            <span className="text-[10px] font-mono font-bold uppercase text-slate-400">Bio / identity</span>
                                            <span className={`text-[10px] font-mono ${createForm.values.bio.trim().length > 0 && createForm.values.bio.trim().length < 20 ? 'text-amber-500' : 'text-slate-400'}`}>{createForm.values.bio.trim().length} chars</span>
                                        </span>
                                        <textarea value={createForm.values.bio} rows={4}
                                            onChange={e => createForm.setField('bio', e.target.value)}
                                            onBlur={() => createForm.markTouched('bio')}
                                            placeholder="Who is this artist? Where do they come from, what do they sound like, what do they care about?" className="apple-input text-xs" />
                                        {createForm.showError('bio') && (
                                            <span className="text-[10px] text-rose-500 font-mono mt-1 block" role="alert">{createForm.showError('bio')}</span>
                                        )}
                                    </label>
                                    <button type="button" onClick={() => createForm.setField('bio', 'Raised between two cities and a river of late-night radio. Writes about distance, memory, and the small hours. Voice like worn velvet over steady drums.')}
                                        className="text-[10px] font-bold text-teal-600 dark:text-teal-400 hover:underline">
                                        Use an example
                                    </button>
                                </div>
                            )}

                            {createStep === 2 && (
                                <div className="space-y-2">
                                    <label className="block">
                                        <span className="text-[10px] font-mono font-bold uppercase text-slate-400 block mb-1">Style tags (comma-separated)</span>
                                        <input value={createForm.values.tags}
                                            onChange={e => createForm.setField('tags', e.target.value)}
                                            placeholder="e.g. indie folk, warm, fingerpicked guitar" className="apple-input text-xs font-mono" />
                                    </label>
                                    {styleChips.length > 0 && (
                                        <div className="flex flex-wrap gap-1.5 pt-1">
                                            {styleChips.map(s => {
                                                const active = createForm.values.tags.toLowerCase().includes(s.name.toLowerCase());
                                                return (
                                                    <button key={s.name} type="button" onClick={() => toggleChip(s.name)}
                                                        className={`text-[10px] font-mono px-2 py-0.5 rounded-full border transition-colors ${active
                                                            ? 'bg-teal-500 text-slate-950 border-teal-500'
                                                            : 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border-teal-500/20 hover:bg-teal-500/20'}`}>
                                                        {s.name}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            )}

                            {createStep === 3 && (
                                <div className="space-y-3">
                                    <span className="text-[10px] font-mono font-bold uppercase text-slate-400 block">Identity image (optional)</span>
                                    <div className="flex items-center gap-3">
                                        {createCoverPreview ? (
                                            <img src={createCoverPreview} alt="Cover preview" className="w-14 h-14 rounded-2xl object-cover border border-black/10 dark:border-white/10" />
                                        ) : (
                                            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-400/30 to-fuchsia-500/30 flex items-center justify-center"><UserCog size={20} className="text-slate-500" /></div>
                                        )}
                                        <label className="text-[10px] font-bold px-2 py-1.5 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500 hover:text-slate-950 transition-colors cursor-pointer">
                                            {createCover ? 'Change image' : 'Choose image'}
                                            <input type="file" accept="image/*" className="hidden"
                                                onChange={e => {
                                                    const f = e.target.files?.[0];
                                                    if (f) {
                                                        setCreateCover(f);
                                                        setCreateCoverPreview(URL.createObjectURL(f));
                                                    }
                                                }} />
                                        </label>
                                        {createCover && (
                                            <button type="button" onClick={() => { setCreateCover(null); setCreateCoverPreview(''); }}
                                                className="text-[10px] font-bold text-slate-400 hover:text-rose-500">Remove</button>
                                        )}
                                    </div>
                                    <p className="text-[10px] text-slate-400">You can always add this later from the artist's page.</p>
                                </div>
                            )}

                            <div className="flex justify-between items-center pt-1">
                                {createStep > 0 ? (
                                    <button onClick={() => setCreateStep(s => s - 1)} disabled={createBusy}
                                        className="px-3 py-1.5 text-xs font-bold rounded-xl text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 disabled:opacity-40">
                                        Back
                                    </button>
                                ) : (
                                    <button onClick={() => setIsCreateOpen(false)} disabled={createBusy}
                                        className="px-3 py-1.5 text-xs font-bold rounded-xl text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 disabled:opacity-40">
                                        Cancel
                                    </button>
                                )}
                                {createStep < 3 ? (
                                    <button onClick={() => setCreateStep(s => s + 1)}
                                        disabled={createStep === 0 ? !!createForm.errors.name : createStep === 1 ? !!createForm.errors.bio : false}
                                        className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold text-xs disabled:opacity-40 active:scale-[0.98] transition-all">
                                        Next
                                    </button>
                                ) : (
                                    <button onClick={submitCreate} disabled={createBusy || !!createForm.errors.name || !!createForm.errors.bio}
                                        className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold text-xs disabled:opacity-40 active:scale-[0.98] transition-all">
                                        {createBusy ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                                        Create Artist
                                    </button>
                                )}
                            </div>
                    </div>
                </Modal>
            </div>
        );
    }

    // ── DETAIL MODE ────────────────────────────────────────────────────────
    const hasExperiencerCrew = detail.assignments.some(a => a.agent_name === 'experiencer');

    return (
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 max-w-5xl mx-auto w-full animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between gap-4 mb-6">
                <div className="flex items-center gap-3 min-w-0">
                    <button onClick={closeDetail} aria-label="Back to artists"
                        className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 border border-black/[0.06] dark:border-white/10 text-xs font-semibold transition-all">
                        <ArrowLeft size={14} />
                        <span>Artists</span>
                    </button>
                    <div className="min-w-0">
                        <div className="flex items-center gap-2">
                            <h1 className="text-xl sm:text-2xl font-black tracking-tight text-slate-900 dark:text-white truncate">{detail.profile.name}</h1>
                            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 font-bold shrink-0">
                                Artist
                            </span>
                        </div>
                        <p className="text-xs font-mono text-slate-400 truncate mt-0.5">
                            {detail.assignments.length} crew · {detail.releases.length} releases
                        </p>
                    </div>
                </div>
                <button onClick={handleDeleteProfile}
                    className="p-2.5 rounded-xl text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all"
                    title="Delete artist profile">
                    <Trash2 size={16} />
                </button>
            </div>

            {/* Identity editor */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 sm:p-6 mb-6 space-y-5">
                {/* Section Header with Title and Aligned Save Button */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-black/[0.05] dark:border-white/[0.06]">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 border border-teal-500/20 flex items-center justify-center text-teal-600 dark:text-teal-400 shrink-0">
                            <UserCog size={16} />
                        </div>
                        <div>
                            <h2 className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                                Artist Identity
                                {identityDirty && (
                                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/20 font-medium">
                                        Unsaved edits
                                    </span>
                                )}
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">Core persona, sonic branding, and vocal style</p>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
                        {saveState === 'saving' && (
                            <span className="text-xs text-teal-500 font-mono flex items-center gap-1.5 mr-1">
                                <Loader2 size={13} className="animate-spin" /> Saving…
                            </span>
                        )}
                        {saveState === 'saved' && (
                            <span className="text-xs text-emerald-500 font-mono flex items-center gap-1.5 mr-1">
                                <CheckCircle2 size={13} /> Saved
                            </span>
                        )}
                        <button
                            onClick={handleSaveIdentity}
                            disabled={saveState === 'saving' || !identityDirty || !identityForm.isValid}
                            className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all shadow-sm active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed ${
                                identityDirty && identityForm.isValid
                                    ? 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 shadow-teal-500/20'
                                    : 'bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 text-slate-700 dark:text-slate-200'
                            }`}
                        >
                            <Save size={13} />
                            <span>Save Identity</span>
                        </button>
                    </div>
                </div>

                {saveState === 'error' && saveError && (
                    <p className="text-[11px] font-mono text-rose-600 dark:text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-xl p-2.5" role="alert">{saveError}</p>
                )}

                {/* Avatar & Visual Branding Header Block */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-4 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/5">
                    <div className="relative group shrink-0">
                        {detail.profile.cover_image_path ? (
                            <img
                                src={`${API_BASE_URL}${detail.profile.cover_image_path}`}
                                alt={detail.profile.name}
                                className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl object-cover border border-black/10 dark:border-white/10 shadow-sm"
                            />
                        ) : (
                            <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-br from-teal-400/20 via-cyan-500/15 to-fuchsia-500/20 border border-teal-500/20 flex items-center justify-center">
                                <UserCog size={28} className="text-teal-600/70 dark:text-teal-400/70" />
                            </div>
                        )}
                    </div>
                    <div className="flex-1 min-w-0 space-y-1.5">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Portrait & Cover Art</h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                            Upload a custom portrait or generate neural concept art grounded in the artist's backstory and style.
                        </p>
                        <div className="flex flex-wrap items-center gap-2 pt-1">
                            <label className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 text-slate-700 dark:text-slate-200 transition-colors cursor-pointer">
                                <Upload size={12} className="text-teal-500" />
                                <span>{coverBusy ? 'Uploading…' : (detail.profile.cover_image_path ? 'Change image' : 'Add identity image')}</span>
                                <input type="file" accept="image/*" className="hidden" disabled={coverBusy}
                                    onChange={e => { const f = e.target.files?.[0]; if (f) uploadCover(f); }} />
                            </label>
                            <button
                                onClick={() => generateCover('profile')}
                                disabled={coverGenBusy !== null}
                                className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-xl bg-fuchsia-500/10 text-fuchsia-600 dark:text-fuchsia-400 hover:bg-fuchsia-500 hover:text-slate-950 border border-fuchsia-500/20 disabled:opacity-50 transition-colors"
                                title="Generate identity art from this artist's lore and tags"
                            >
                                <Sparkles size={12} />
                                <span>{coverGenBusy === 'profile' ? 'Imagining…' : 'Generate art'}</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Form Fields: Grid Layout with clear labels */}
                <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Artist Name */}
                        <div className="space-y-1.5">
                            <label htmlFor="identity-name" className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                                <span>Artist Name</span>
                                <span className="text-[11px] text-slate-400 font-normal">Public identity</span>
                            </label>
                            <input
                                id="identity-name"
                                value={identityForm.values.name}
                                onChange={e => identityForm.setField('name', e.target.value)}
                                onBlur={() => identityForm.markTouched('name')}
                                placeholder="e.g. Nova Eclipse"
                                className="w-full apple-input text-sm font-semibold"
                                aria-label="Artist name"
                            />
                            {identityForm.showError('name') && (
                                <span className="text-[10px] text-rose-500 font-mono block" role="alert">{identityForm.showError('name')}</span>
                            )}
                        </div>

                        {/* Singing Voice (paired in 2-col grid for clean hierarchy) */}
                        <div className="space-y-1.5">
                            <label htmlFor="identity-voice" className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                                <span className="flex items-center gap-1.5"><Mic2 size={12} className="text-teal-500" /> Singing Voice</span>
                                {voiceSaving && <span className="text-[10px] text-teal-500 font-mono flex items-center gap-1"><Loader2 size={10} className="animate-spin" /> saving</span>}
                            </label>
                            <select
                                id="identity-voice"
                                value={detail.profile.voice_profile_id || ''}
                                onChange={e => handleChangeVoice(e.target.value)}
                                disabled={voiceSaving}
                                className="w-full apple-input text-xs"
                                aria-label="Singing voice"
                            >
                                <option value="">Provider default (no custom voice)</option>
                                {voiceProfiles.filter(v => v.status === 'ready').map(v => (
                                    <option key={v.id} value={v.id}>{v.name}</option>
                                ))}
                            </select>
                            {voiceProfiles.filter(v => v.status === 'ready').length === 0 && (
                                <span className="text-[10px] text-slate-400 block">No voice profiles yet — create one in Voice Lab.</span>
                            )}
                        </div>
                    </div>

                    {/* Artist Bio */}
                    <div className="space-y-1.5">
                        <label htmlFor="identity-bio" className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                            <span>Artist Bio & Concept</span>
                            <span className="text-[11px] text-slate-400 font-normal">Grounds the AI crew on backstory and character</span>
                        </label>
                        <textarea
                            id="identity-bio"
                            value={identityForm.values.bio}
                            onChange={e => identityForm.setField('bio', e.target.value)}
                            rows={3}
                            placeholder="Bio — who is this artist? Where do they come from, what do they sound like, what do they care about? The crew reads this for grounding."
                            className="w-full apple-input text-xs leading-relaxed resize-y"
                            aria-label="Artist bio"
                        />
                    </div>

                    {/* Genre & Style Tags */}
                    <div className="space-y-1.5">
                        <label htmlFor="identity-tags" className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                            <span className="flex items-center gap-1.5"><Tag size={12} className="text-teal-500" /> Style & Genre Tags</span>
                            <span className="text-[11px] text-slate-400 font-normal">Comma-separated</span>
                        </label>
                        <input
                            id="identity-tags"
                            value={identityForm.values.tags}
                            onChange={e => identityForm.setField('tags', e.target.value)}
                            placeholder="e.g. ambient, synthwave, electronic, cinematic percussion"
                            className="w-full apple-input text-xs font-mono"
                            aria-label="Artist style tags"
                        />
                        {identityForm.values.tags && (
                            <div className="flex flex-wrap gap-1.5 pt-1">
                                {identityForm.values.tags.split(',').map(t => t.trim()).filter(Boolean).map(t => (
                                    <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20">
                                        {t}
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* World Lore (Canonical Artist Lore Document) */}
                    <details className="group rounded-xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.04] dark:border-white/5 transition-all">
                        <summary className="p-3.5 flex items-center justify-between cursor-pointer select-none text-xs font-semibold text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors">
                            <div className="flex items-center gap-2">
                                <BookOpen size={14} className="text-teal-500" />
                                <span>World Lore & Canon</span>
                                <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                                    detail.profile.lore_json && detail.profile.lore_json !== '{}'
                                        ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
                                        : 'bg-black/[0.04] dark:bg-white/5 text-slate-400 border-black/[0.06] dark:border-white/10'
                                }`}>
                                    {detail.profile.lore_json && detail.profile.lore_json !== '{}' ? 'Configured' : 'Empty'}
                                </span>
                            </div>
                            <ChevronDown size={14} className="text-slate-400 transition-transform group-open:rotate-180" />
                        </summary>
                        <div className="p-3.5 pt-0 space-y-3 border-t border-black/[0.04] dark:border-white/5 mt-1">
                            <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed pt-2">
                                Structured canon as JSON or freeform narrative text. The World-Builder and Experiencer agents ground on this document during album generation.
                            </p>
                            <textarea
                                value={editLore}
                                onChange={e => setEditLore(e.target.value)}
                                rows={5}
                                placeholder={'Structured lore as JSON or freeform text — the crew reads this as canonical history.\ne.g. {"hometown": "Lusaka", "era": "1970s", "themes": ["resilience", "night drive"]}'}
                                className="w-full apple-input !bg-black/[0.02] dark:!bg-black/20 text-xs font-mono leading-relaxed"
                                aria-label="World lore"
                            />
                            <div className="flex items-center justify-end gap-2">
                                <button
                                    onClick={handleGenerateLore}
                                    disabled={loreGenerating || loreSaving}
                                    className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500 hover:text-slate-950 border border-teal-500/20 disabled:opacity-50 transition-colors"
                                >
                                    <Sparkles size={12} />
                                    <span>{loreGenerating ? 'Imagining…' : 'Generate with World-Builder'}</span>
                                </button>
                                <button
                                    onClick={handleSaveLore}
                                    disabled={loreSaving || loreGenerating}
                                    className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 text-slate-700 dark:text-slate-200 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 disabled:opacity-50 transition-colors"
                                >
                                    <Save size={12} />
                                    <span>{loreSaving ? 'Saving…' : 'Save Lore'}</span>
                                </button>
                            </div>
                        </div>
                    </details>
                </div>
            </section>

            {/* AI Crew */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 sm:p-6 mb-6 space-y-5">
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-black/[0.05] dark:border-white/[0.06]">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 border border-teal-500/20 flex items-center justify-center text-teal-600 dark:text-teal-400 shrink-0">
                            <Users size={16} />
                        </div>
                        <div>
                            <h2 className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                                AI Creative Crew
                                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-500 dark:text-slate-400 font-normal">
                                    {detail.assignments.length} assigned
                                </span>
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">Specialized autonomous agents collaborating on the artist's discography</p>
                        </div>
                    </div>
                </div>

                {/* Crew Cards: Compact Responsive Grid */}
                {detail.assignments.length === 0 ? (
                    <div className="p-6 text-center rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-dashed border-black/[0.08] dark:border-white/[0.08]">
                        <Users size={24} className="mx-auto text-slate-400 mb-2" />
                        <p className="text-xs font-bold text-slate-700 dark:text-slate-300">No agents assigned yet</p>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 max-w-sm mx-auto mt-0.5">
                            Assign at least an Experiencer to unlock album concept generation and lifecycle production.
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        {detail.assignments.map(a => (
                            <div
                                key={a.id}
                                className="p-3.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.05] dark:border-white/[0.06] hover:border-teal-500/30 dark:hover:border-teal-500/30 transition-all flex flex-col justify-between gap-3 group relative"
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div className="min-w-0">
                                        <span className="text-xs font-bold text-slate-900 dark:text-white truncate block">
                                            {ROLE_LABELS[a.role] || a.role}
                                        </span>
                                        <span className="text-[10px] font-mono text-slate-400 block mt-0.5">
                                            agent: <span className="text-slate-600 dark:text-slate-300 font-medium">{a.agent_name}</span>
                                        </span>
                                    </div>
                                    <button
                                        onClick={() => removeCrewMember(a.id)}
                                        aria-label={`Remove ${a.role}`}
                                        className="p-1 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors opacity-70 group-hover:opacity-100"
                                        title={`Remove ${ROLE_LABELS[a.role] || a.role}`}
                                    >
                                        <X size={13} />
                                    </button>
                                </div>

                                <div className="pt-2 border-t border-black/[0.04] dark:border-white/5 flex items-center justify-between gap-2">
                                    {a.model_provider ? (
                                        <span
                                            className="text-[9px] font-mono text-teal-600 dark:text-teal-400 bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 rounded-md truncate max-w-full"
                                            title={`Pinned Model: ${a.model_provider}${a.model ? `/${a.model}` : ''}`}
                                        >
                                            pinned: {a.model_provider}{a.model ? `/${a.model}` : ''}
                                        </span>
                                    ) : (
                                        <span className="text-[9px] font-mono text-slate-400 bg-black/[0.03] dark:bg-white/[0.04] px-2 py-0.5 rounded-md">
                                            Default failover chain
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Add Crew Member Panel */}
                <div className="p-4 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.05] dark:border-white/[0.06] space-y-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Assign Crew Member
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-2.5 items-end">
                        <div className="sm:col-span-1 md:col-span-2 space-y-1">
                            <label className="text-[10px] font-semibold text-slate-600 dark:text-slate-400 block">Role</label>
                            <select
                                value={crewRole}
                                onChange={e => setCrewRole(e.target.value)}
                                className="w-full apple-input !py-1.5 !px-2.5 text-xs font-medium"
                                aria-label="Crew role"
                            >
                                {ROLES.map(r => <option key={r} value={r}>{ROLE_LABELS[r] || r}</option>)}
                            </select>
                        </div>
                        <div className="sm:col-span-1 md:col-span-2 space-y-1">
                            <label className="text-[10px] font-semibold text-slate-600 dark:text-slate-400 block">Agent Persona</label>
                            <select
                                value={crewAgent}
                                onChange={e => setCrewAgent(e.target.value)}
                                className="w-full apple-input !py-1.5 !px-2.5 text-xs font-medium"
                                aria-label="Agent"
                            >
                                {agentsRegistry.map(a => <option key={a.name} value={a.name}>{a.display_name}</option>)}
                            </select>
                        </div>
                        <div className="sm:col-span-2 md:col-span-1">
                            <button
                                onClick={addCrewMember}
                                aria-label="Add crew member"
                                className="w-full h-[34px] px-3 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm active:scale-[0.98] transition-all"
                            >
                                <Plus size={14} /> <span>Add</span>
                            </button>
                        </div>
                    </div>

                    {/* Optional Model Override Collapsible */}
                    <details className="group pt-1">
                        <summary className="text-[11px] text-slate-500 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 cursor-pointer select-none font-medium flex items-center gap-1.5 transition-colors">
                            <ChevronRight size={12} className="transition-transform group-open:rotate-90" />
                            <span>Model override for new assignment (optional)</span>
                        </summary>
                        <div className="pt-2 pl-4 space-y-2">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                <select
                                    value={crewProvider}
                                    onChange={e => setCrewProvider(e.target.value)}
                                    className="w-full apple-input !py-1.5 !px-2 text-xs font-mono"
                                    aria-label="Override provider"
                                >
                                    <option value="">— artist default —</option>
                                    {LLM_PROVIDERS.map(p => <option key={p} value={p}>{p}</option>)}
                                </select>
                                <input
                                    value={crewModel}
                                    onChange={e => setCrewModel(e.target.value)}
                                    placeholder="model id (optional)"
                                    aria-label="Override model"
                                    className="w-full apple-input !py-1.5 !px-2 text-xs font-mono"
                                />
                            </div>
                            <p className="text-[10px] text-slate-400 leading-normal">
                                Pinned here → attempted FIRST for this artist, global failover chain stays behind it.
                            </p>
                        </div>
                    </details>
                </div>
            </section>

            {/* Experiencer Studio */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 sm:p-6 mb-6 space-y-5">
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-black/[0.05] dark:border-white/[0.06]">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 border border-teal-500/20 flex items-center justify-center text-teal-600 dark:text-teal-400 shrink-0">
                            <Sparkles size={16} />
                        </div>
                        <div>
                            <h2 className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                                Experiencer Studio
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">Conceptualize journey arcs, emotional beats, and full album narrative visions</p>
                        </div>
                    </div>
                </div>

                {!hasExperiencerCrew && (
                    <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-700 dark:text-amber-300 flex items-center gap-2">
                        <AlertTriangle size={14} className="shrink-0 text-amber-500" />
                        <span>Assign an experiencer above to ground runs in this artist.</span>
                    </div>
                )}

                <div className="space-y-4">
                    <div className="space-y-1.5">
                        <label htmlFor="brief-title" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                            Album Title
                        </label>
                        <input
                            id="brief-title"
                            value={briefTitle}
                            onChange={e => setBriefTitle(e.target.value)}
                            placeholder="Album title"
                            aria-label="Album title"
                            className="w-full apple-input text-sm font-semibold"
                        />
                    </div>

                    <div className="space-y-1.5">
                        <label htmlFor="brief-concept" className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                            <span>Album Concept & Vision</span>
                            <span className="text-[11px] text-slate-400 font-normal">The narrative premise the Experiencer inhabits</span>
                        </label>
                        <textarea
                            id="brief-concept"
                            value={briefConcept}
                            onChange={e => setBriefConcept(e.target.value)}
                            rows={3}
                            placeholder="Album concept — the premise the experiencer will live inside…"
                            className="w-full apple-input text-xs leading-relaxed resize-y"
                            aria-label="Album concept"
                        />
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 block">
                                Tracks
                            </label>
                            <input
                                type="number"
                                min={1}
                                max={30}
                                value={briefTarget}
                                onChange={e => setBriefTarget(Math.max(1, Math.min(30, parseInt(e.target.value) || 1)))}
                                className="w-full apple-input !py-1.5 text-xs font-mono"
                            />
                        </div>
                        <div className="sm:col-span-2 space-y-1.5">
                            <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 block">
                                Extra direction (optional)
                            </label>
                            <input
                                value={briefDirection}
                                onChange={e => setBriefDirection(e.target.value)}
                                placeholder="mood, references, constraints…"
                                className="w-full apple-input !py-1.5 text-xs"
                            />
                        </div>
                    </div>
                </div>

                <div className="flex items-center justify-between gap-3 pt-2">
                    {runPhase === 'running' ? (
                        <span className="text-xs font-mono text-slate-500 flex items-center gap-2">
                            <Loader2 size={13} className="animate-spin text-teal-500" />
                            {runStage || 'Imagining'} · {elapsed}s (large models: 1-4 min)
                        </span>
                    ) : <span />}
                    <button
                        onClick={runExperiencer}
                        disabled={runPhase === 'running' || !briefTitle.trim() || !briefConcept.trim()}
                        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all disabled:opacity-40 disabled:pointer-events-none"
                    >
                        {runPhase === 'running' ? <Loader2 size={13} className="animate-spin" /> : <Mic2 size={13} />}
                        <span>{runPhase === 'running' ? 'Imagining…' : 'Run Experiencer'}</span>
                    </button>
                </div>

                {runError && (
                    <pre className="text-[11px] font-mono text-rose-600 dark:text-rose-400 whitespace-pre-wrap bg-rose-500/10 border border-rose-500/20 rounded-xl p-3.5 select-text">{runError}</pre>
                )}
            </section>

            {/* Vision artifact */}
            {vision && (
                <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-teal-500/30 shadow-apple-lg backdrop-blur-xl p-6 space-y-5 relative overflow-hidden">
                    <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-teal-500 via-cyan-400 to-sky-500" />
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <p className="text-[10px] font-mono font-bold uppercase tracking-widest text-teal-600 dark:text-teal-400 flex items-center gap-1.5">
                                <Sparkles size={11} /> Experiencer Vision
                            </p>
                            <h2 className="text-lg font-extrabold text-slate-900 dark:text-white mt-1">{vision.journey_title}</h2>
                        </div>
                        <button onClick={() => navigator.clipboard.writeText(JSON.stringify(vision, null, 2)).catch(() => {})}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-teal-600 hover:bg-teal-500/10 transition-colors" title="Copy vision JSON">
                            <Copy size={13} />
                        </button>
                    </div>
                    <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">{vision.concept_statement}</p>

                    <div>
                        <h3 className="text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400 mb-2">Life Journey</h3>
                        <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed whitespace-pre-line select-text max-h-48 overflow-y-auto custom-scrollbar pr-2">{vision.life_journey_narrative}</p>
                    </div>

                    <div>
                        <h3 className="text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400 mb-2">Emotional Arc</h3>
                        <div className="flex flex-wrap gap-2">
                            {vision.emotional_arc.map(b => (
                                <div key={b.position} className="flex-1 min-w-[130px] p-2.5 rounded-xl bg-black/[0.03] dark:bg-white/[0.04] border border-black/[0.05] dark:border-white/5">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[11px] font-extrabold capitalize text-slate-700 dark:text-slate-200">{b.label}</span>
                                        <span className="text-[9px] font-mono text-slate-400 tabular-nums">{Math.round(b.intensity * 100)}%</span>
                                    </div>
                                    <div className="mt-1.5 h-1 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                                        <div className="h-full rounded-full bg-gradient-to-r from-teal-500 to-cyan-400" style={{ width: `${b.intensity * 100}%` }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div>
                        <h3 className="text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400 mb-2">Song Seeds ({vision.song_seeds.length})</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {vision.song_seeds.map((s, i) => (
                                <div key={i} className="p-4 rounded-2xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.05] dark:border-white/5 space-y-2">
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="text-xs font-extrabold text-slate-900 dark:text-white truncate">{s.working_title}</span>
                                        <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 border border-teal-500/20 flex-shrink-0">{s.placement_hint}</span>
                                    </div>
                                    <p className="text-[11px] italic text-slate-500 dark:text-slate-400">{s.mood}</p>
                                    <p className="text-[11px] text-slate-600 dark:text-slate-300 leading-relaxed select-text">{s.story_seed}</p>
                                    <div className="flex items-center gap-2">
                                        <span className="text-[9px] font-mono text-slate-400">energy</span>
                                        <div className="flex-1 h-1 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                                            <div className="h-full bg-gradient-to-r from-teal-500 to-amber-400" style={{ width: `${s.energy * 100}%` }} />
                                        </div>
                                    </div>
                                    {s.suggested_style_tags.length > 0 && (
                                        <div className="flex flex-wrap gap-1 pt-0.5">
                                            {s.suggested_style_tags.map(t => (
                                                <span key={t} className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border border-cyan-500/20">{t}</span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {vision.recurring_motifs.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                            {vision.recurring_motifs.map(m => (
                                <span key={m} className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-violet-500/10 text-violet-600 dark:text-violet-300 border border-violet-500/20">{m}</span>
                            ))}
                        </div>
                    )}
                    {vision.listener_experience_notes && (
                        <blockquote className="border-l-2 border-teal-500 pl-3 text-xs italic text-slate-600 dark:text-slate-300">
                            {vision.listener_experience_notes}
                        </blockquote>
                    )}

                    <button onClick={saveVisionAsRelease}
                        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 text-xs font-bold text-slate-700 dark:text-slate-200 active:scale-[0.98] transition-all">
                        <Disc3 size={13} /> Save as Release
                    </button>
                </section>
            )}

            {/* Releases */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 sm:p-6 mb-6 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-black/[0.05] dark:border-white/[0.06]">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 border border-teal-500/20 flex items-center justify-center text-teal-600 dark:text-teal-400 shrink-0">
                            <Disc3 size={16} />
                        </div>
                        <div>
                            <h2 className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                                Releases & Discography
                                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-500 dark:text-slate-400 font-normal">
                                    {detail.releases.length} releases
                                </span>
                            </h2>
                            <p className="text-xs text-slate-500 dark:text-slate-400">Multi-track albums, EPs, and neural production pipeline</p>
                        </div>
                    </div>
                </div>
                    {albumRun && (
                        <div className={`p-2.5 rounded-xl border ${albumRun.status === 'failed' ? 'border-red-500/30 bg-red-500/5'
                            : albumRun.status === 'done' ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-fuchsia-500/30 bg-fuchsia-500/5'}`}>
                            <div className="flex items-center justify-between gap-2 mb-1">
                                <span className="text-[11px] font-bold text-slate-800 dark:text-slate-100 truncate">Album · {albumRun.releaseTitle}</span>
                                <div className="flex items-center gap-1.5 shrink-0">
                                    {albumRun.status === 'awaiting_approval' && (
                                        <button onClick={approveNextTrack}
                                            className="text-[10px] font-bold px-2 py-1 rounded-lg bg-emerald-500 text-slate-950 hover:bg-emerald-400 transition-colors">Approve next track</button>)}
                                    {['queued', 'running', 'awaiting_approval', 'cancelling'].includes(albumRun.status) && (
                                        <button onClick={cancelAlbum}
                                            className="text-[10px] font-bold px-2 py-1 rounded-lg bg-red-500/10 text-red-600 hover:bg-red-500 hover:text-slate-950 transition-colors">Cancel</button>)}
                                </div>
                            </div>
                            <div className="w-full h-1.5 rounded-full bg-black/[0.06] dark:bg-white/10 overflow-hidden mb-1">
                                <div className="h-full rounded-full bg-gradient-to-r from-teal-400 to-fuchsia-500 transition-all duration-500" style={{ width: `${Math.max(3, albumRun.progress)}%` }} />
                            </div>
                            <p className="text-[10px] text-slate-500 dark:text-slate-400 truncate">
                                {albumRun.status} · {albumRun.progress}% · {albumRun.message}
                            </p>
                        </div>
                    )}
                <div className="flex items-center justify-between gap-2">
                    <input value={newReleaseTitle} onChange={e => setNewReleaseTitle(e.target.value)}
                        placeholder="New release title…" className="apple-input text-xs flex-1"
                        aria-label="New release title"
                        onKeyDown={async e => {
                            if (e.key === 'Enter' && newReleaseTitle.trim()) {
                                const r = await profilesApi.createRelease({ profile_id: detail.profile.id, title: newReleaseTitle.trim() }).catch((e) => { toast(String(e?.response?.data?.detail?.error?.message || e?.message || "Request failed"), "error"); return null; });
                                if (r) { setDetail(d => d ? ({ ...d, releases: [r, ...d.releases] }) : d); setNewReleaseTitle(''); }
                            }
                        }} />
                    {newReleaseTitle.trim() && (
                        <button onClick={async () => {
                            const r = await profilesApi.createRelease({ profile_id: detail.profile.id, title: newReleaseTitle.trim() }).catch((e) => { toast(String(e?.response?.data?.detail?.error?.message || e?.message || "Request failed"), "error"); return null; });
                            if (r) { setDetail(d => d ? ({ ...d, releases: [r, ...d.releases] }) : d); setNewReleaseTitle(''); }
                        }} className="p-2 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500 hover:text-slate-950 transition-colors" aria-label="Create release"><Plus size={13} /></button>
                    )}
                </div>
                <label className="flex items-center gap-2 text-[10px] font-mono text-slate-500 dark:text-slate-400 select-none">
                    <input type="checkbox" checked={autopilot} onChange={e => setAutopilot(e.target.checked)}
                        className="accent-teal-500 w-3 h-3" aria-label="Autopilot mode" />
                    Autopilot — produce every track without approval pauses (burns budget faster on failure)
                </label>
                <label className="flex items-center gap-2 text-[10px] font-mono text-slate-500 dark:text-slate-400 select-none">
                    Budget cap
                    <select value={budgetMin} onChange={e => setBudgetMin(e.target.value as 'off' | '15' | '30' | '60')}
                        className="apple-input !py-1 !px-2 text-[10px] font-mono w-auto" aria-label="Budget cap">
                        <option value="off">off</option>
                        <option value="15">15 min</option>
                        <option value="30">30 min</option>
                        <option value="60">60 min</option>
                    </select>
                </label>
                <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500 dark:text-slate-400 select-none">
                    <span className="uppercase tracking-wider">Crew (extra LLM calls/track):</span>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" checked={crewStylist} onChange={e => setCrewStylist(e.target.checked)}
                            className="accent-teal-500 w-3 h-3" aria-label="Stylist crew agent" />
                        Stylist
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" checked={crewCritic} onChange={e => setCrewCritic(e.target.checked)}
                            className="accent-teal-500 w-3 h-3" aria-label="Critic crew agent" />
                        Critic
                    </label>
                </div>
                {detail.releases.length === 0 ? (
                    <p className="text-xs text-slate-500 italic py-1">No releases yet.</p>
                ) : detail.releases.map(r => (
                    <div key={r.id} className="flex items-center justify-between gap-2 p-2.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.04] dark:border-white/5">
                        {editReleaseId === r.id ? (
                            <div className="flex items-center gap-1.5 flex-1 min-w-0">
                                <input value={editReleaseTitle} onChange={e => setEditReleaseTitle(e.target.value)}
                                    onKeyDown={e => { if (e.key === 'Enter') handleRenameRelease(r.id); }}
                                    className="apple-input !py-1 text-xs flex-1" autoFocus aria-label="Release title" />
                                <input value={editReleaseDesc} onChange={e => setEditReleaseDesc(e.target.value)}
                                    onKeyDown={e => { if (e.key === 'Enter') handleRenameRelease(r.id); }}
                                    placeholder="description (optional)" aria-label="Release description"
                                    className="apple-input !py-1 text-[10px] flex-1" />
                                <button onClick={() => handleRenameRelease(r.id)} disabled={!editReleaseTitle.trim()}
                                    className="text-[10px] font-bold px-2 py-1 rounded-lg bg-emerald-500 text-slate-950 hover:bg-emerald-400 disabled:opacity-40 transition-colors">Save</button>
                                <button onClick={() => setEditReleaseId(null)}
                                    className="text-[10px] font-bold px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 text-slate-500 hover:text-slate-700 transition-colors">Cancel</button>
                            </div>
                        ) : (
                            <>
                                <div className="flex items-center gap-2 min-w-0">
                                    {r.cover_image_path && (
                                        <img src={`${API_BASE_URL}${r.cover_image_path}`} alt="" className="w-8 h-8 rounded-lg object-cover border border-black/10 dark:border-white/10" />
                                    )}
                                    <div className="min-w-0">
                                        <span className="text-xs font-bold text-slate-800 dark:text-slate-100 truncate">{r.title}</span>
                                        {r.status !== 'planned' && (
                                            <span className={`ml-2 text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-full ${r.status === 'completed' ? 'bg-emerald-500/15 text-emerald-600' : 'bg-sky-500/15 text-sky-600 dark:text-sky-300'}`}>{r.status}</span>
                                        )}
                                    </div>
                                </div>
                                <div className="flex items-center gap-1.5 shrink-0">
                                    <button onClick={() => generateCover(r.id)} disabled={coverGenBusy !== null}
                                        className="text-[10px] font-bold px-2 py-1 rounded-lg bg-fuchsia-500/10 text-fuchsia-600 dark:text-fuchsia-400 hover:bg-fuchsia-500 hover:text-slate-950 disabled:opacity-50 transition-colors"
                                        title="Generate release artwork">{coverGenBusy === r.id ? 'Imagining…' : 'Art'}</button>
                                    <button onClick={() => { setEditReleaseId(r.id); setEditReleaseTitle(r.title); setEditReleaseDesc(r.description || ''); }}
                                        className="text-[10px] font-bold px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                                        title="Rename release" aria-label={`Rename ${r.title}`}>Edit</button>
                                    <button onClick={() => handleDeleteRelease(r.id, r.title)}
                                        className="text-[10px] font-bold px-2 py-1 rounded-lg bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-slate-950 transition-colors"
                                        title="Delete release" aria-label={`Delete ${r.title}`}>Delete</button>
                                    <button onClick={() => toggleTracks(r.id)} className="text-[10px] font-bold px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-300 hover:bg-black/[0.08] dark:hover:bg-white/10 transition-colors">{openTracks === r.id ? 'Hide tracks' : 'Tracks'}</button>
                                    <button onClick={() => startAlbum(r.id, r.title)} disabled={albumActive}
                                        className="text-[10px] font-bold px-2 py-1 rounded-lg bg-fuchsia-500/10 text-fuchsia-600 dark:text-fuchsia-400 hover:bg-fuchsia-500 hover:text-slate-950 transition-colors disabled:opacity-40"
                                        title={autopilot ? 'Produce this album (autopilot: no pauses)' : 'Produce this album (gated: pauses after each track)'}>Produce</button>
                                </div>
                            </>
                        )}
                    </div>
                ))}
                {openTracks && (
                    <div className="mt-2 p-3 rounded-xl border border-black/[0.06] dark:border-white/[0.08] bg-black/[0.02] dark:bg-white/[0.03] space-y-2">
                        {!tracks ? (
                            <p className="text-[10px] text-slate-400 font-mono">Loading tracks…</p>
                        ) : (
                            <>
                                <div className="flex items-center justify-between">
                                    <span className="text-[11px] font-bold text-slate-700 dark:text-slate-200">{tracks.title} · {tracks.succeeded}/{tracks.total} complete</span>
                                    <div className="flex items-center gap-1">
                                        {tracks.status !== 'planned' && (
                                            <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-full bg-sky-500/15 text-sky-600 dark:text-sky-300">{tracks.status}</span>
                                        )}
                                        <span className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-full ${tracks.rollup === 'completed' ? 'bg-emerald-500/15 text-emerald-600' : tracks.rollup === 'partial' ? 'bg-amber-500/15 text-amber-600' : 'bg-black/[0.04] dark:bg-white/5 text-slate-500'}`}>{tracks.rollup}</span>
                                        {openReleaseMeta?.active_run && (
                                            <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-full bg-sky-500/15 text-sky-600 dark:text-sky-300 animate-pulse" title="An album run is currently producing on this release">● run live</span>
                                        )}
                                    </div>
                                </div>
                                {tracks.tracks.map((tr, idx) => (
                                    <div key={tr.id} className="flex items-center justify-between gap-2 p-2 rounded-lg bg-white/60 dark:bg-white/[0.04] border border-black/[0.04] dark:border-white/5">
                                        <div className="min-w-0">
                                            <p className="text-[11px] font-bold text-slate-800 dark:text-slate-100 truncate">
                                                {tr.title || '(untitled)'}
                                                {tr.review && tr.review.verdict !== 'unavailable' && (
                                                    <span className={`ml-1.5 text-[9px] font-mono uppercase px-1 py-0.5 rounded ${
                                                        tr.review.verdict === 'pass' ? 'bg-emerald-500/15 text-emerald-600'
                                                            : tr.review.verdict === 'concern' ? 'bg-amber-500/15 text-amber-600'
                                                                : 'bg-slate-500/15 text-slate-500'}`}
                                                        title={tr.review.notes || ''}>
                                                        {tr.review.verdict}{typeof tr.review.score === 'number' ? ` ${Math.round(tr.review.score * 100)}%` : ''}
                                                    </span>
                                                )}
                                            </p>
                                            <div className="flex items-center gap-2 mt-0.5">
                                                <span className="text-[9px] text-slate-500 font-mono">
                                                    {Math.round(tr.duration_ms / 1000)}s · {tr.seed_slot != null ? `slot ${tr.seed_slot + 1}` : `seed ${tr.seed ?? '—'}`}
                                                </span>
                                                {tr.used_real_inference ? (
                                                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[8px] font-bold uppercase tracking-wider bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                                                        ● MiniMax Music 3 (Neural)
                                                    </span>
                                                ) : (
                                                    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[8px] font-bold uppercase tracking-wider bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/20" title="Procedural waveform generator used">
                                                        ▲ Fallback Synth
                                                    </span>
                                                )}
                                            </div>
                                            {tr.status === 'completed' && tr.artifacts?.audio && (
                                                <div className="mt-1 max-w-[240px]" title="Click to play from any position">
                                                    <StaticWaveform
                                                        jobId={tr.id}
                                                        buckets={120}
                                                        heightClass="h-6"
                                                        progressFraction={currentTrack?.id === tr.id && duration > 0 ? currentTime / duration : null}
                                                        onSeekFraction={(f) => handleWaveSeek(tr, f)}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                        <div className="flex items-center gap-1 shrink-0">
                                            {tracks.tracks.length > 1 && (
                                                <span className="flex flex-col -space-y-1 mr-0.5">
                                                    <button onClick={() => moveTrack(idx, -1)} disabled={idx === 0}
                                                        className="text-slate-400 hover:text-teal-600 disabled:opacity-30 transition-colors"
                                                        title="Move up" aria-label={`Move ${tr.title || 'track'} up`}>▴</button>
                                                    <button onClick={() => moveTrack(idx, 1)} disabled={idx === tracks.tracks.length - 1}
                                                        className="text-slate-400 hover:text-teal-600 disabled:opacity-30 transition-colors"
                                                        title="Move down" aria-label={`Move ${tr.title || 'track'} down`}>▾</button>
                                                </span>
                                            )}
                                            {(['audio', 'midi', 'musicxml', 'mastered'] as const).map(k =>
                                                tr.artifacts?.[k] ? (
                                                    <a key={k} href={`${API_BASE_URL}${tr.artifacts[k]}`} target="_blank" rel="noreferrer"
                                                        className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500 hover:text-slate-950 transition-colors">{k.slice(0, 4)}</a>
                                                ) : null
                                            )}
                                            {(() => {
                                                if (!tr.artifacts?.stems) return null;
                                                try {
                                                    const parsed = typeof tr.artifacts.stems === 'string' ? JSON.parse(tr.artifacts.stems) : tr.artifacts.stems;
                                                    if (parsed && typeof parsed === 'object') {
                                                        return Object.entries(parsed).map(([stemName, stemPath]) => (
                                                            typeof stemPath === 'string' && stemPath ? (
                                                                <a key={stemName} href={`${API_BASE_URL}${stemPath}`} target="_blank" rel="noreferrer"
                                                                    title={`Download ${stemName} stem`}
                                                                    className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 hover:bg-cyan-500 hover:text-slate-950 transition-colors">
                                                                    {stemName.slice(0, 3)}
                                                                </a>
                                                            ) : null
                                                        ));
                                                    }
                                                } catch {
                                                    return null;
                                                }
                                                return null;
                                            })()}
                                            {tr.status === 'failed' && openTracks && (
                                                <button onClick={() => retryTrack(openTracks, tr.id, tr.title || 'Track')}
                                                    className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600 hover:bg-amber-500 hover:text-slate-950 transition-colors"
                                                    title="Reproduce this track from its seed">Retry</button>
                                            )}
                                            {tr.status === 'completed' && tr.artifacts?.audio && (
                                                <button onClick={() => playFromTracklist(tr.id, tracks.tracks)} disabled={playFetchingId === tr.id}
                                                    className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 hover:bg-emerald-500 hover:text-slate-950 disabled:opacity-50 transition-colors"
                                                    title="Play through the studio player">
                                                    {playFetchingId === tr.id ? '…' : '▶ Play'}
                                                </button>
                                            )}
                                            {tr.status === 'completed' && !tr.artifacts?.audio && (
                                                <span
                                                    className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-slate-500/15 text-slate-500 cursor-help"
                                                    title="Marked completed but no audio file is attached (render may have failed silently, or the file was moved/deleted). Use Retry to reproduce, or Detach to remove.">
                                                    ⚠ no audio
                                                </span>
                                            )}
                                            {tr.status === 'missing' && (
                                                <span
                                                    className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600 dark:text-amber-400 cursor-help"
                                                    title="The winning track for this slot was deleted or detached. The slot is kept visible so nothing vanishes silently.">
                                                    ⚠ track missing{typeof tr.seed_slot === 'number' ? ` (slot ${tr.seed_slot + 1})` : ''}
                                                </span>
                                            )}
                                            {tr.status === 'completed' && (
                                                <button onClick={() => openInStudio(tr.id)}
                                                    className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-600 dark:text-sky-300 hover:bg-sky-500 hover:text-slate-950 transition-colors"
                                                    title="Open this track in the studio (full playback, DAW, transcription)">Studio</button>
                                            )}
                                            <span className={`w-1.5 h-1.5 rounded-full ${tr.status === 'completed' ? 'bg-emerald-500' : tr.status === 'failed' ? 'bg-red-500' : 'bg-amber-400'}`} title={tr.status} />
                                            {openTracks && tr.status !== 'missing' && (
                                                <button onClick={() => detachTrack(openTracks, tr.id, tr.title || 'Track')}
                                                    className="text-slate-400 hover:text-red-500 transition-colors p-0.5 ml-0.5"
                                                    title="Detach track from this release">
                                                    <X size={11} />
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </>
                        )}
                    </div>
                )}
            </section>

            {/* Run history (C5): this artist's agent ledger — newest first. */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 sm:p-6 mb-6">
                <details className="group">
                    <summary className="flex items-center justify-between cursor-pointer select-none">
                        <div className="flex items-center gap-2.5">
                            <div className="w-8 h-8 rounded-xl bg-teal-500/10 dark:bg-teal-500/20 border border-teal-500/20 flex items-center justify-center text-teal-600 dark:text-teal-400 shrink-0">
                                <History size={16} />
                            </div>
                            <div>
                                <h2 className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                                    Agent Run History
                                    {runHistory.length > 0 && <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-500 dark:text-slate-400 font-normal">({runHistory.length})</span>}
                                </h2>
                                <p className="text-xs text-slate-500 dark:text-slate-400">Autonomous execution log and telemetry for this artist</p>
                            </div>
                        </div>
                        <ChevronDown size={15} className="text-slate-400 transition-transform group-open:rotate-180" />
                    </summary>
                    {runHistory.length === 0 ? (
                        <p className="text-xs text-slate-500 italic py-2">No agent runs yet — visions and album production will appear here.</p>
                    ) : (
                        <>
                        <div className="mt-3 space-y-1.5" role="list" aria-label="Agent run history">
                            {runHistory.map(r => (
                                <div key={r.id} role="listitem" className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.04] dark:border-white/5">
                                    <div className="flex items-center gap-2 min-w-0">
                                        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${r.status === 'succeeded' ? 'bg-emerald-500' : r.status === 'failed' ? 'bg-red-500' : ['queued', 'running', 'awaiting_approval'].includes(r.status) ? 'bg-amber-400 animate-pulse' : 'bg-slate-400'}`} title={r.status} />
                                        <span className="text-[11px] font-bold text-slate-700 dark:text-slate-200 capitalize truncate">{ROLE_LABELS[r.agent_name] || r.agent_name}</span>
                                        {r.error_message && (
                                            <span className="text-[9px] font-mono text-rose-500 truncate" title={r.error_message}>{r.error_message.slice(0, 60)}</span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2 shrink-0 font-mono text-[9px] text-slate-400">
                                        {(r.tokens_out ?? 0) > 0 && <span>{(r.tokens_out ?? 0).toLocaleString()} out</span>}
                                        {!!r.latency_ms && <span>{(r.latency_ms / 1000).toFixed(1)}s</span>}
                                        <span>{fmtUtcDate(r.created_at)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                        {runStats && runStats.total > 0 && (
                            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] font-mono text-slate-400" aria-label="Run aggregates">
                                <span>{runStats.total} runs</span>
                                {runStats.success_rate != null && (
                                    <span>· {Math.round(runStats.success_rate * 100)}% success</span>
                                )}
                                {runStats.latency_ms.p50 != null && (
                                    <span>· p50 {(runStats.latency_ms.p50 / 1000).toFixed(1)}s</span>
                                )}
                                {runStats.latency_ms.p95 != null && (
                                    <span>· p95 {(runStats.latency_ms.p95 / 1000).toFixed(1)}s</span>
                                )}
                                {runStats.tokens_out > 0 && (
                                    <span>· {runStats.tokens_out.toLocaleString()} out-tokens</span>
                                )}
                            </div>
                        )}
                        </>
                    )}
                </details>
            </section>
        </div>
    );
};
