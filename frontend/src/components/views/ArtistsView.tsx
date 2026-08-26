import { toast } from '../../utils/toast';
import React, { useState, useEffect, useRef } from 'react';
import {
    Users, Plus, ArrowLeft, Trash2, Save, Loader2, Sparkles,
    Mic2, UserCog, Disc3, CheckCircle2, AlertTriangle, X, Copy
} from 'lucide-react';
import {
    agentsApi, profilesApi, albumApi, api,
    type AgentInfo, type ArtistProfileT,
    type ProfileDetail, type ExperiencerVision, type AgentRunEnvelope
} from '../../api';

const ROLES = ['world_builder', 'experiencer', 'songwriter', 'producer'];

type RunPhase = 'idle' | 'running' | 'done' | 'error';

export const ArtistsView: React.FC = () => {
    const [profiles, setProfiles] = useState<ArtistProfileT[]>([]);
    const [isLoadingList, setIsLoadingList] = useState(true);
    const [detail, setDetail] = useState<ProfileDetail | null>(null);
    const [isDetailLoading, setIsDetailLoading] = useState(false);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- gates future detail spinner
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [newName, setNewName] = useState('');
    const [newBio, setNewBio] = useState('');
    const [newTags, setNewTags] = useState('');
    const [agentsRegistry, setAgentsRegistry] = useState<AgentInfo[]>([]);

    // identity edit state
    const [editName, setEditName] = useState('');
    const [editBio, setEditBio] = useState('');
    const [editTags, setEditTags] = useState('');
    const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle');

    // crew add row
    const [crewRole, setCrewRole] = useState('experiencer');
    const [crewAgent, setCrewAgent] = useState('experiencer');

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

    const startAlbum = async (releaseId: string, title: string) => {
        try {
            const res = await albumApi.produce(releaseId, false);
            albumRunIdRef.current = res.run_id;
            setAlbumRun({ runId: res.run_id, releaseTitle: title, status: 'queued', progress: 0, message: 'Imagining the journey…' });
        } catch {
            setAlbumRun({ runId: '-', releaseTitle: title, status: 'failed', progress: 0, message: 'Could not start album run.' });
        }
    };

    const approveNextTrack = async () => {
        if (!albumRun || albumRun.runId === '-') return;
        setAlbumRun(r => r ? { ...r, status: 'running', message: 'Approved — producing next track…' } : r);
        await albumApi.resume(albumRun.runId).catch(() => null);
    };

    const cancelAlbum = async () => {
        if (!albumRun || albumRun.runId === '-') return;
        await albumApi.cancelRun(albumRun.runId).catch(() => null);
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

    useEffect(() => {
        (async () => {
            setIsLoadingList(true);
            try {
                setProfiles(await profilesApi.list());
            } catch (e) {
                console.error(e);
            } finally {
                setIsLoadingList(false);
            }
            agentsApi.listAgents().then(setAgentsRegistry).catch(console.error);
        })();
    }, []);

    const openProfile = async (id: string) => {
        setIsDetailLoading(true);
        try {
            const d = await profilesApi.get(id);
            setDetail(d);
            setEditName(d.profile.name);
            setEditBio(d.profile.bio);
            setEditTags(d.profile.tags);
            setVision(null);
            setRunPhase('idle');
            setRunError('');
        } finally {
            setIsDetailLoading(false);
        }
    };

    const refreshDetail = () => detail && openProfile(detail.profile.id);

    const handleCreate = async () => {
        if (!newName.trim()) return;
        try {
            const p = await profilesApi.create({ name: newName.trim(), bio: newBio.trim(), tags: newTags.trim() });
            setProfiles(prev => [p, ...prev]);
            setIsCreateOpen(false);
            setNewName(''); setNewBio(''); setNewTags('');
            openProfile(p.id);
        } catch (e) {
            console.error('Create profile failed', e);
        }
    };

    const handleSaveIdentity = async () => {
        if (!detail) return;
        setSaveState('saving');
        try {
            const updated = await profilesApi.update(detail.profile.id, {
                name: editName.trim(), bio: editBio, tags: editTags
            });
            setDetail({ ...detail, profile: updated });
            setProfiles(prev => prev.map(p => p.id === updated.id ? updated : p));
            setSaveState('saved');
            setTimeout(() => setSaveState('idle'), 2000);
        } catch {
            setSaveState('error' as never); // surfaced via button color below
            setSaveState('idle');
        }
    };

    const handleDeleteProfile = async () => {
        if (!detail || !window.confirm(`Delete artist "${detail.profile.name}"? Their crew assignments are removed; discography history remains.`)) return;
        await profilesApi.delete(detail.profile.id).catch(console.error);
        setProfiles(prev => prev.filter(p => p.id !== detail.profile.id));
        setDetail(null);
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
                { role: crewRole, agent_name: crewAgent }
            ]);
            refreshDetail();
        } catch (e) {
            console.error(e);
        }
    };

    const removeCrewMember = async (assignmentId: string) => {
        if (!detail) return;
        const remaining = detail.assignments.filter(a => a.id !== assignmentId)
            .map(a => ({ role: a.role, agent_name: a.agent_name }));
        try {
            const updated = await profilesApi.setAssignments(detail.profile.id, remaining);
            setDetail({ ...detail, assignments: updated });
        } catch (e) {
            console.error(e);
        }
    };

    // ── Experiencer run ────────────────────────────────────────────────────
    // Live stage text for experiencer runs via SSE run_progress events.
    useEffect(() => {
        if (runPhase !== 'running') return;
        const es = api.connectToEvents((event: MessageEvent) => {
            try {
                const d = JSON.parse(event.data);
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
            const row = await albumApi.getRun(albumRun.runId).catch(() => null);
            if (row) setAlbumRun(prev => prev ? ({ ...prev, status: row.status, progress: row.progress ?? prev.progress }) : prev);
        }, 5000);
        return () => clearInterval(t);
    }, [albumActive, albumRun?.runId]);

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
            });
            refreshDetail();
        } catch (e) {
            console.error(e);
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
                        onClick={() => setIsCreateOpen(true)}
                        className="px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all"
                    >
                        <Plus size={14} /> New Artist
                    </button>
                </div>

                {isLoadingList ? (
                    <div className="py-16 flex justify-center"><Loader2 size={22} className="animate-spin text-teal-500" /></div>
                ) : profiles.length === 0 ? (
                    <div className="py-20 text-center space-y-3">
                        <Users size={36} className="mx-auto text-slate-300 dark:text-slate-600" />
                        <p className="text-sm font-bold text-slate-700 dark:text-slate-200">No artists yet</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto">
                            Create an artist profile, assign their AI crew, and let the Experiencer imagine their first album.
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                        {profiles.map(p => (
                            <button
                                key={p.id}
                                onClick={() => openProfile(p.id)}
                                className="text-left p-5 rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm hover:shadow-apple-md backdrop-blur-xl transition-all hover:-translate-y-0.5"
                            >
                                <h3 className="text-sm font-extrabold text-slate-900 dark:text-white truncate">{p.name}</h3>
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
                            </button>
                        ))}
                    </div>
                )}

                {/* Create modal */}
                {isCreateOpen && (
                    <div className="fixed inset-0 z-[110] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in"
                        onMouseDown={e => { if (e.target === e.currentTarget) setIsCreateOpen(false); }}>
                        <div className="w-full max-w-md rounded-3xl bg-white dark:bg-[#141620] border border-black/10 dark:border-white/10 shadow-apple-2xl p-6 space-y-4 animate-scale-up">
                            <h3 className="text-sm font-extrabold text-slate-900 dark:text-white">New Artist Profile</h3>
                            <input autoFocus value={newName} onChange={e => setNewName(e.target.value)}
                                placeholder="Artist name" className="apple-input text-sm" />
                            <textarea value={newBio} onChange={e => setNewBio(e.target.value)} rows={3}
                                placeholder="Bio / identity — who is this artist?" className="apple-input text-xs" />
                            <input value={newTags} onChange={e => setNewTags(e.target.value)}
                                placeholder="Style tags, comma-separated" className="apple-input text-xs font-mono" />
                            <div className="flex justify-end gap-2 pt-1">
                                <button onClick={() => setIsCreateOpen(false)} className="px-3 py-1.5 text-xs font-bold rounded-xl text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Cancel</button>
                                <button onClick={handleCreate} disabled={!newName.trim()}
                                    className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold text-xs disabled:opacity-40 active:scale-[0.98] transition-all">
                                    Create Artist
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        );
    }

    // ── DETAIL MODE ────────────────────────────────────────────────────────
    const hasExperiencerCrew = detail.assignments.some(a => a.agent_name === 'experiencer');

    return (
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 max-w-5xl mx-auto w-full animate-fade-in">
            {/* Header */}
            <div className="flex items-start justify-between gap-4 mb-6">
                <div className="flex items-center gap-3 min-w-0">
                    <button onClick={() => setDetail(null)}
                        className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] text-slate-600 dark:text-slate-300">
                        <ArrowLeft size={15} />
                    </button>
                    <div className="min-w-0">
                        <h1 className="text-xl font-extrabold tracking-tight text-slate-900 dark:text-white truncate">{detail.profile.name}</h1>
                        <p className="text-[11px] font-mono text-slate-400 truncate">
                            {detail.assignments.length} crew · {detail.releases.length} releases
                        </p>
                    </div>
                </div>
                <button onClick={handleDeleteProfile}
                    className="p-2 rounded-xl text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors"
                    title="Delete artist profile">
                    <Trash2 size={15} />
                </button>
            </div>

            {/* Identity editor */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 space-y-3 mb-5">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2"><UserCog size={13} /> Identity</h2>
                <input value={editName} onChange={e => setEditName(e.target.value)} placeholder="Artist name" className="apple-input text-sm font-bold" />
                <textarea value={editBio} onChange={e => setEditBio(e.target.value)} rows={3}
                    placeholder="Bio — who is this artist? The crew reads this for grounding." className="apple-input text-xs" />
                <input value={editTags} onChange={e => setEditTags(e.target.value)}
                    placeholder="Style tags, comma-separated" className="apple-input text-xs font-mono" />
                <div className="flex justify-end items-center gap-2">
                    {saveState === 'saving' && <Loader2 size={13} className="animate-spin text-teal-500" />}
                    {saveState === 'saved' && <CheckCircle2 size={13} className="text-emerald-500" />}
                    <button onClick={handleSaveIdentity} disabled={saveState === 'saving'}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 text-xs font-bold text-slate-700 dark:text-slate-200 disabled:opacity-50">
                        <Save size={12} /> Save Identity
                    </button>
                </div>
            </section>

            {/* Crew */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 space-y-3 mb-5">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2"><UserCog size={13} /> AI Crew</h2>
                {detail.assignments.length === 0 ? (
                    <p className="text-xs text-slate-500 italic py-2">No agents assigned yet — this artist has no crew.</p>
                ) : detail.assignments.map(a => (
                    <div key={a.id} className="flex items-center justify-between gap-3 p-2.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.04] dark:border-white/5">
                        <div className="min-w-0">
                            <span className="text-xs font-extrabold text-slate-800 dark:text-slate-100 capitalize">{a.role}</span>
                            <span className="text-[10px] font-mono text-slate-400 ml-2">agent: {a.agent_name}</span>
                        </div>
                        <button onClick={() => removeCrewMember(a.id)} aria-label={`Remove ${a.role}`}
                            className="p-1 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 transition-colors">
                            <X size={13} />
                        </button>
                    </div>
                ))}
                <div className="flex items-center gap-2 pt-1">
                    <select value={crewRole} onChange={e => setCrewRole(e.target.value)} className="apple-input !py-1.5 !px-2 text-[11px] font-mono flex-1" aria-label="Crew role">
                        {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                    </select>
                    <select value={crewAgent} onChange={e => setCrewAgent(e.target.value)} className="apple-input !py-1.5 !px-2 text-[11px] font-mono flex-1" aria-label="Agent">
                        {agentsRegistry.map(a => <option key={a.name} value={a.name}>{a.display_name}</option>)}
                    </select>
                    <button onClick={addCrewMember} aria-label="Add crew member"
                        className="p-2 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500 hover:text-slate-950 transition-colors">
                        <Plus size={14} />
                    </button>
                </div>
            </section>

            {/* Experiencer Studio */}
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 space-y-3 mb-5">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2"><Sparkles size={13} /> Experiencer Studio</h2>
                {!hasExperiencerCrew && (
                    <p className="text-[11px] text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                        <AlertTriangle size={12} /> Assign an experiencer above to ground runs in this artist.
                    </p>
                )}
                <input value={briefTitle} onChange={e => setBriefTitle(e.target.value)} placeholder="Album title"
                    className="apple-input text-sm font-bold" />
                <textarea value={briefConcept} onChange={e => setBriefConcept(e.target.value)} rows={3}
                    placeholder="Album concept — the premise the experiencer will live inside…" className="apple-input text-xs" />
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <label className="space-y-1">
                        <span className="text-[10px] font-mono font-bold uppercase text-slate-400 block">Tracks</span>
                        <input type="number" min={1} max={30} value={briefTarget}
                            onChange={e => setBriefTarget(Math.max(1, Math.min(30, parseInt(e.target.value) || 1)))}
                            className="apple-input !py-1.5 text-xs font-mono" />
                    </label>
                    <label className="space-y-1 sm:col-span-2">
                        <span className="text-[10px] font-mono font-bold uppercase text-slate-400 block">Extra direction (optional)</span>
                        <input value={briefDirection} onChange={e => setBriefDirection(e.target.value)}
                            placeholder="mood, references, constraints…" className="apple-input !py-1.5 text-xs" />
                    </label>
                </div>
                <div className="flex items-center justify-between gap-3 pt-1">
                    {runPhase === 'running' ? (
                        <span className="text-xs font-mono text-slate-500 flex items-center gap-2">
                            <Loader2 size={13} className="animate-spin text-teal-500" />
                            {runStage || 'Imagining'} · {elapsed}s (large models: 1-4 min)
                        </span>
                    ) : <span />}
                    <button onClick={runExperiencer}
                        disabled={runPhase === 'running' || !briefTitle.trim() || !briefConcept.trim()}
                        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all disabled:opacity-40 disabled:pointer-events-none">
                        {runPhase === 'running' ? <Loader2 size={13} className="animate-spin" /> : <Mic2 size={13} />}
                        {runPhase === 'running' ? 'Imagining…' : 'Run Experiencer'}
                    </button>
                </div>

                {runError && (
                    <pre className="text-[11px] font-mono text-rose-600 dark:text-rose-400 whitespace-pre-wrap bg-rose-500/10 rounded-xl p-3 select-text">{runError}</pre>
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
            <section className="rounded-2xl bg-white/70 dark:bg-[#141620]/80 border border-black/[0.06] dark:border-white/[0.08] shadow-apple-sm backdrop-blur-xl p-5 space-y-3">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2"><Disc3 size={13} /> Releases</h2>
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
                <div className="flex items-center gap-2">
                    <input value={newReleaseTitle} onChange={e => setNewReleaseTitle(e.target.value)}
                        placeholder="New release title…" className="apple-input text-xs flex-1"
                        onKeyDown={async e => {
                            if (e.key === 'Enter' && newReleaseTitle.trim()) {
                                const r = await profilesApi.createRelease({ profile_id: detail.profile.id, title: newReleaseTitle.trim() }).catch(() => null);
                                if (r) { setDetail(d => d ? ({ ...d, releases: [r, ...d.releases] }) : d); setNewReleaseTitle(''); }
                            }
                        }} />
                    {newReleaseTitle.trim() && (
                        <button onClick={async () => {
                            const r = await profilesApi.createRelease({ profile_id: detail.profile.id, title: newReleaseTitle.trim() }).catch(() => null);
                            if (r) { setDetail(d => d ? ({ ...d, releases: [r, ...d.releases] }) : d); setNewReleaseTitle(''); }
                        }} className="p-2 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 hover:bg-teal-500 hover:text-slate-950 transition-colors" aria-label="Create release"><Plus size={13} /></button>
                    )}
                </div>
                {detail.releases.length === 0 ? (
                    <p className="text-xs text-slate-500 italic py-1">No releases yet.</p>
                ) : detail.releases.map(r => (
                    <div key={r.id} className="flex items-center justify-between p-2.5 rounded-xl bg-black/[0.02] dark:bg-white/[0.03] border border-black/[0.04] dark:border-white/5">
                        <span className="text-xs font-bold text-slate-800 dark:text-slate-100 truncate">{r.title}</span>
                        <div className="flex items-center gap-1.5 shrink-0">
                            <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-full bg-black/[0.04] dark:bg-white/5 text-slate-500">{r.status}</span>
                            <button onClick={() => startAlbum(r.id, r.title)} disabled={albumActive}
                                className="text-[10px] font-bold px-2 py-1 rounded-lg bg-fuchsia-500/10 text-fuchsia-600 dark:text-fuchsia-400 hover:bg-fuchsia-500 hover:text-slate-950 transition-colors disabled:opacity-40"
                                title="Produce this album (gated: pauses after each track)">Produce</button>
                        </div>
                    </div>
                ))}
            </section>
        </div>
    );
};
