import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  api,
  trainingApi,
  workspaceApi,
  sessionApi,
  coverApi,
  type Job,
  type Project,
  type StudioSession,
  type SessionMessage
} from './api';
import { ComposerSidebar, type CompositionData } from './components/ComposerSidebar';
import { HistoryFeed } from './components/HistoryFeed';
import { TrainingStudio } from './components/TrainingStudio';
import { VoiceStudioModal } from './components/voice/VoiceStudioModal';
import { ModelsManagerModal } from './components/models/ModelsManagerModal';
import { LLMSettingsModal } from './components/LLMSettingsModal';
import { SessionWorkspace } from './components/workspace/SessionWorkspace';
import { FloatingStatusWidget } from './components/ui/FloatingStatusWidget';
import { MilimoLogo } from './components/ui/MilimoLogo';
import { useTheme } from './context/ThemeContext';
import { useAudioEngine } from './context/AudioEngineContext';

// Dedicated Reference IA Views
import { SongsView } from './components/views/SongsView';
import { PlaylistsView } from './components/views/PlaylistsView';
import { ProjectsView } from './components/views/ProjectsView';
import { MusicVideosView } from './components/views/MusicVideosView';
import { ProfileView } from './components/views/ProfileView';
import { TrackDetailView } from './components/views/TrackDetailView';
import { GlobalAudioPlayer } from './components/ui/GlobalAudioPlayer';
import { AppFooter } from './components/ui/AppFooter';

import {
  Plus,
  Compass,
  Sliders,
  Music,
  ListMusic,
  FolderKanban,
  Video,
  User,
  Mic,
  GraduationCap,
  Cpu,
  Settings,
  Sparkles,
  Upload,
  Sun,
  Moon,
  Laptop,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  Menu,
  X,
  MessageSquare,
  ArrowUp,
  Paperclip,
  Trash2,
  FileAudio,
  Square
} from 'lucide-react';

export type NavView =
  | 'explore'
  | 'songs'
  | 'projects'
  | 'playlists'
  | 'videos'
  | 'profile'
  | 'workspace'
  | 'sessions'
  | 'track-detail';

function App() {
  const { theme, setTheme } = useTheme();
  const [currentNav, setCurrentNav] = useState<NavView>('explore');
  const [previousNav, setPreviousNav] = useState<NavView>('songs');
  const [selectedTrack, setSelectedTrack] = useState<Job | null>(null);
  const [lyricsModels, setLyricsModels] = useState<string[]>([]);
  const [history, setHistory] = useState<Job[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [parentJob, setParentJob] = useState<Job | undefined>(undefined);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGeneratingLyrics, setIsGeneratingLyrics] = useState(false);
  const [activeWorkspaceJob, setActiveWorkspaceJob] = useState<Job | null>(null);

  // Studio Sessions & Multi-Turn Chat State
  const [sessions, setSessions] = useState<StudioSession[]>([]);
  const [activeSession, setActiveSession] = useState<StudioSession | null>(null);
  const [isChatSubmitting, setIsChatSubmitting] = useState(false);
  const [attachmentPath, setAttachmentPath] = useState<string | null>(null);
  const attachmentInputRef = useRef<HTMLInputElement>(null);

  // Centralized Audio Playback Engine
  const {
    currentTrack: engineTrack,
    isPlaying: engineIsPlaying,
    playTrack: enginePlayTrack,
    togglePlay: engineTogglePlay,
    pause: enginePause,
    stop: engineStop
  } = useAudioEngine();

  const playingSong = engineTrack;
  const isPlayingAudio = engineIsPlaying;
  const [activeProject, setActiveProject] = useState<Project | null>(null);

  // Responsive Layout States
  const [isLeftRailCollapsed, setIsLeftRailCollapsed] = useState<boolean>(() => {
    return typeof window !== 'undefined' ? window.innerWidth < 1200 : false;
  });
  const [isComposerOpen, setIsComposerOpen] = useState<boolean>(() => {
    return typeof window !== 'undefined' ? window.innerWidth >= 1024 : true;
  });
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  // Modals
  const [isTrainingOpen, setIsTrainingOpen] = useState(false);
  const [isVoiceStudioOpen, setIsVoiceStudioOpen] = useState(false);
  const [isModelsManagerOpen, setIsModelsManagerOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeCheckpoint, setActiveCheckpoint] = useState<{ name: string; id: string } | null>(null);

  // Chat-first Producer landing input
  const [producerInput, setProducerInput] = useState('');

  // Pagination & Search
  const [historyOffset, setHistoryOffset] = useState(0);
  const [historyFilter, setHistoryFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [hasMoreHistory, setHasMoreHistory] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const HISTORY_LIMIT = 20;

  // Window Resize Listener for Responsive Layout
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      if (width < 768) {
        setIsLeftRailCollapsed(true);
        setIsComposerOpen(false);
      } else if (width < 1200) {
        setIsLeftRailCollapsed(true);
        if (width < 1024) {
          setIsComposerOpen(false);
        }
      } else {
        setIsLeftRailCollapsed(false);
        setIsComposerOpen(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const refreshActiveCheckpoint = async () => {
    try {
      const checkpoints = await trainingApi.listCheckpoints();
      const active = checkpoints.find((c: { is_active: boolean }) => c.is_active);
      setActiveCheckpoint(active ? { name: active.name, id: active.id } : null);
    } catch (e) {
      console.error("Failed to load active checkpoint", e);
    }
  };

  const loadHistory = async (offset: number, filter: string, search: string, replace: boolean = false) => {
    if (isLoadingHistory && offset !== 0) return;
    setIsLoadingHistory(true);
    try {
      const jobs = await api.getHistory(HISTORY_LIMIT, offset, filter, search);
      setHasMoreHistory(jobs.length >= HISTORY_LIMIT);

      if (replace) {
        setHistory(jobs);
        if (jobs.length > 0 && !activeWorkspaceJob) {
          const recentCompleted = jobs.find(j => j.status === 'completed');
          if (recentCompleted) setActiveWorkspaceJob(recentCompleted);
        }
      } else {
        setHistory(prev => {
          const newJobs = jobs.filter(j => !prev.find(p => p.id === j.id));
          return [...prev, ...newJobs];
        });
      }
    } catch (e) {
      console.error("Failed to fetch history", e);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const loadSessions = async () => {
    try {
      const list = await sessionApi.listSessions();
      setSessions(list);
    } catch (e) {
      console.error("Failed to load sessions", e);
    }
  };

  const handleCreateNewSession = async () => {
    try {
      const newSession = await sessionApi.createSession({
        title: 'New Session'
      });
      setSessions(prev => [newSession, ...prev]);
      setActiveSession(newSession);
      setCurrentNav('explore');
      setProducerInput('');
    } catch (e) {
      console.error("Failed to create new session", e);
    }
  };

  const handleSelectSession = async (session: StudioSession) => {
    try {
      const full = await sessionApi.getSession(session.id);
      setActiveSession(full);
      setCurrentNav('explore');
    } catch (e) {
      console.error("Failed to select session", e);
    }
  };

  const handleDeleteSession = async (sessionId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    try {
      await sessionApi.deleteSession(sessionId);
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      if (activeSession?.id === sessionId) {
        setActiveSession(null);
      }
    } catch (err) {
      console.error("Failed to delete session", err);
    }
  };

  const [producerStatusStage, setProducerStatusStage] = useState<string>('Analyzing musical direction & style tags...');
  const activeChatAbortController = useRef<AbortController | null>(null);

  const handleCancelChatSubmission = () => {
    if (activeChatAbortController.current) {
      activeChatAbortController.current.abort();
      activeChatAbortController.current = null;
    }
    setIsChatSubmitting(false);
    window.dispatchEvent(new CustomEvent('milimo_progress', {
      detail: {
        job_id: 'producer-planning',
        stage: 'Cancelled',
        progress: 100,
        message: 'Prompt planning terminated by user.'
      }
    }));
  };

  const handleSendChatMessage = async (content: string) => {
    const trimmed = content.trim();
    if (!trimmed && !attachmentPath) return;

    const currentAttachment = attachmentPath;

    // 1. Instant User Feedback: Clear prompt input immediately (0ms)
    setProducerInput('');
    setAttachmentPath(null);

    // 2. Optimistic user message insertion
    const tempUserId = 'temp-msg-' + Date.now();
    let targetSession = activeSession;
    const optimisticUserMsg: SessionMessage = {
      id: tempUserId,
      session_id: targetSession?.id || 'temp',
      role: 'user',
      content: trimmed,
      audio_attachment_path: currentAttachment || undefined,
      created_at: new Date().toISOString()
    };

    if (!targetSession) {
      targetSession = {
        id: 'temp-session-' + Date.now(),
        title: trimmed.slice(0, 30) || 'New Session',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        messages: [optimisticUserMsg]
      };
      setActiveSession(targetSession);
      setSessions(prev => [targetSession!, ...prev]);
    } else {
      const updated = {
        ...targetSession,
        messages: [...(targetSession.messages || []), optimisticUserMsg]
      };
      setActiveSession(updated);
      setSessions(prev => prev.map(s => s.id === updated.id ? updated : s));
    }

    setIsChatSubmitting(true);
    setProducerStatusStage('Analyzing musical direction & style tags...');

    const controller = new AbortController();
    activeChatAbortController.current = controller;

    // 3. Immediate Floating Status HUD notification
    window.dispatchEvent(new CustomEvent('milimo_progress', {
      detail: {
        job_id: 'producer-planning',
        stage: 'AI Producer',
        progress: 25,
        message: 'Analyzing musical direction & style tags...'
      }
    }));

    // Stage timer transitions for live Apple-grade feedback during LLM execution
    const t1 = setTimeout(() => {
      setProducerStatusStage('Writing song lyrics with AI Co-Writer...');
      window.dispatchEvent(new CustomEvent('milimo_progress', {
        detail: {
          job_id: 'producer-planning',
          stage: 'AI Co-Writer',
          progress: 55,
          message: 'Writing structured song lyrics...'
        }
      }));
    }, 2500);

    const t2 = setTimeout(() => {
      setProducerStatusStage('Structuring arrangement & production captions...');
      window.dispatchEvent(new CustomEvent('milimo_progress', {
        detail: {
          job_id: 'producer-planning',
          stage: 'MiniMax Music 3',
          progress: 85,
          message: 'Structuring arrangement & MiniMax captions...'
        }
      }));
    }, 6000);

    try {
      // If session was freshly created optimistically, real create on server first
      let realSessionId = targetSession.id;
      if (targetSession.id.startsWith('temp-session-')) {
        const created = await sessionApi.createSession({
          title: trimmed.slice(0, 30) || 'New Session'
        });
        realSessionId = created.id;
      }

      const res = await sessionApi.sendChatMessage(realSessionId, {
        content: trimmed,
        role: 'user',
        audio_attachment_path: currentAttachment || undefined
      }, controller.signal);

      clearTimeout(t1);
      clearTimeout(t2);

      window.dispatchEvent(new CustomEvent('milimo_progress', {
        detail: {
          job_id: 'producer-planning',
          stage: 'AI Producer Ready',
          progress: 100,
          message: 'Arrangement & lyrics complete.'
        }
      }));

      setActiveSession(res.session);
      setSessions(prev => prev.map(s => (s.id === res.session.id || s.id === targetSession!.id ? res.session : s)));

      if (res.preset) {
        setProducerPreset(res.preset);
        if (!isComposerOpen) setIsComposerOpen(true);
      }
    } catch (e: any) {
      clearTimeout(t1);
      clearTimeout(t2);
      if (axios.isCancel(e) || e.name === 'CanceledError' || e.name === 'AbortError') {
        console.log("Chat submission cancelled by user.");
      } else {
        alert("Producer chat error: " + (e.response?.data?.detail || e.message));
      }
    } finally {
      setIsChatSubmitting(false);
      activeChatAbortController.current = null;
    }
  };

  useEffect(() => {
    refreshActiveCheckpoint();
    api.getLyricsModels().then(setLyricsModels).catch(console.error);
    loadHistory(0, 'all', '', true);
    loadSessions();
  }, []);

  const handleRefresh = () => {
    loadHistory(0, historyFilter, searchQuery, true);
    loadSessions();
    setHistoryOffset(0);
  };

  const handleLoadMore = () => {
    const newOffset = historyOffset + HISTORY_LIMIT;
    setHistoryOffset(newOffset);
    loadHistory(newOffset, historyFilter, searchQuery, false);
  };

  const handleFilterChange = (newStatus: string) => {
    setHistoryFilter(newStatus);
    setHistoryOffset(0);
    setHistory([]);
    loadHistory(0, newStatus, searchQuery, true);
  };

  const handleToggleFavorite = (jobId: string) => {
    setHistory(prev =>
      prev.map(j => (j.id === jobId ? { ...j, is_favorite: !j.is_favorite } : j))
    );
    if (selectedTrack?.id === jobId) {
      setSelectedTrack(prev => prev ? { ...prev, is_favorite: !prev.is_favorite } : null);
    }
    api.toggleFavorite(jobId).catch(console.error);
  };

  const handleSelectTrack = (track: Job) => {
    if (currentNav !== 'track-detail') {
      setPreviousNav(currentNav);
    }
    setSelectedTrack(track);
    setCurrentNav('track-detail');
    try {
      const url = new URL(window.location.href);
      url.searchParams.set('view', 'track-detail');
      url.searchParams.set('track', track.id);
      window.history.pushState({ view: 'track-detail', trackId: track.id }, '', url.toString());
    } catch (e) {}
  };

  const handleNavigate = (view: NavView) => {
    setCurrentNav(view);
    try {
      const url = new URL(window.location.href);
      url.searchParams.set('view', view);
      url.searchParams.delete('track');
      window.history.pushState({ view }, '', url.toString());
    } catch (e) {}
  };

  useEffect(() => {
    const handlePopState = async () => {
      const params = new URLSearchParams(window.location.search);
      const viewParam = params.get('view') as NavView | null;
      const trackId = params.get('track');

      if (viewParam === 'track-detail' && trackId) {
        try {
          const track = await api.getJobStatus(trackId);
          setSelectedTrack(track);
          setCurrentNav('track-detail');
        } catch (e) {
          console.error(e);
        }
      } else if (viewParam) {
        setCurrentNav(viewParam);
      }
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const handleDeleteJob = async (jobId: string) => {
    if (!confirm("Are you sure you want to delete this track? This action cannot be undone.")) return;

    // 1. Optimistic removal from state
    setHistory(prev => prev.filter(j => j.id !== jobId));
    if (engineTrack?.id === jobId) {
      engineStop();
    }
    if (activeWorkspaceJob?.id === jobId) {
      setActiveWorkspaceJob(null);
    }

    // 2. Clean up any playlist references in localStorage
    try {
      const saved = localStorage.getItem('milimo_playlists');
      if (saved) {
        const plList = JSON.parse(saved);
        const cleaned = plList.map((pl: any) => ({
          ...pl,
          songIds: (pl.songIds || []).filter((id: string) => id !== jobId)
        }));
        localStorage.setItem('milimo_playlists', JSON.stringify(cleaned));
      }
    } catch (e) {
      console.error("Failed to clean playlist references", e);
    }

    // 3. API Delete Call & Re-sync
    try {
      await api.deleteJob(jobId);
      handleRefresh();
    } catch (e) {
      console.error("Delete failed on server", e);
      handleRefresh();
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setHistoryOffset(0);
    setHistory([]);
    loadHistory(0, historyFilter, query, true);
  };

  const [generationProgress, setGenerationProgress] = useState<{
    step: number;
    total_steps: number;
    phase: string;
    progress: number;
    message: string;
  } | null>(null);

  // Throttle progress HUD re-renders (progress ticks are frequent; App re-renders are costly).
  const lastProgressRef = useRef(0);
  const PROGRESS_THROTTLE_MS = 400;

  // SSE Subscription
  useEffect(() => {
    const evtSource = api.connectToEvents((e) => {
      try {
        const type = e.type;
        const data = JSON.parse(e.data);

        if (type === 'job_progress') {
          const now = Date.now();
          if (now - lastProgressRef.current >= PROGRESS_THROTTLE_MS && (data.progress !== undefined || data.message)) {
            lastProgressRef.current = now;
            setGenerationProgress({
              step: data.step || 1,
              total_steps: data.total_steps || 4,
              phase: data.phase || 'generation',
              progress: data.progress || 25,
              message: data.message || 'Synthesizing master...'
            });
          }
          window.dispatchEvent(new CustomEvent('milimo_progress', { detail: data }));
        }

        if (type === 'job_update') {
          if (data.status === 'completed') {
            setIsGenerating(false);
            setGenerationProgress(null);
            setCurrentJobId(null);
            handleRefresh();
          } else if (data.status === 'failed') {
            setIsGenerating(false);
            setGenerationProgress(null);
            setCurrentJobId(null);
            handleRefresh();
          } else if (data.status === 'processing' || data.status === 'queued') {
            // Update local state only — avoid pulling/ re-rendering the full history on
            // every non-terminal transition (the terminal statuses above handle refresh).
            setCurrentJobId(data.job_id);
            setIsGenerating(true);
            if (data.status === 'processing' && currentJobId !== data.job_id) {
              handleRefresh();
            }
          }
        }
      } catch (err) {
        console.error("SSE Parse Error", err);
      }
    });

    return () => evtSource.close();
  }, []);

  // Polling Fallback during active generation
  useEffect(() => {
    if (!isGenerating || !currentJobId) return;

    const interval = setInterval(async () => {
      try {
        const job = await api.getJobStatus(currentJobId);
        if (job.status === 'completed') {
          setIsGenerating(false);
          setGenerationProgress(null);
          setCurrentJobId(null);
          handleRefresh();
        } else if (job.status === 'failed') {
          setIsGenerating(false);
          setGenerationProgress(null);
          setCurrentJobId(null);
          handleRefresh();
        }
      } catch (e) {
        console.error("Status check polling error", e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isGenerating, currentJobId]);

  const handleGenerateMusic = async (data: CompositionData) => {
    setIsGenerating(true);
    setGenerationProgress({
      step: 1,
      total_steps: 4,
      phase: 'generation',
      progress: 10,
      message: `Synthesizing with ${data.modelProvider || 'MiniMax Music 3'}...`
    });

    try {
      const res = await api.generateJob(
        data.topic,
        data.durationMs,
        data.lyrics,
        data.tags,
        data.cfgScale,
        data.temperature,
        data.topk,
        data.llmModel,
        parentJob?.id,
        data.seed,
        data.modelProvider || 'minimax_music3',
        data.voiceProfileId,
        data.structuredCaption,
        data.projectId || activeProject?.id,
        data.title,
        data.isInstrumental,
        data.coverImagePath,
        data.imagePrompt,
        activeSession?.id
      );
      setCurrentJobId(res.job_id);
      setParentJob(undefined);
      handleRefresh(); // Refresh immediately so the new job appears in history
    } catch (e: any) {
      alert("Generation failed: " + (e.response?.data?.detail || e.message));
      setIsGenerating(false);
      setGenerationProgress(null);
    }
  };

  // Preset populated by the "Ask Producer" flow so the composer reflects the producer's choices.
  const [producerPreset, setProducerPreset] = useState<Partial<CompositionData> | null>(null);

  // "Ask Producer" flow: the LLM actually writes lyrics + derives title/style, and we use
  // those real inputs to generate the final track AND prefill the composer panel.
  const handleProducerGenerate = async (prompt: string) => {
    if (!prompt) return;
    setProducerInput('');
    setIsGenerating(true);
    setGenerationProgress({
      step: 1,
      total_steps: 10,
      phase: 'AI Producer',
      progress: 15,
      message: 'AI Producer: Analyzing prompt & structuring musical brief...'
    });
    window.dispatchEvent(new CustomEvent('milimo_progress', {
      detail: {
        job_id: 'producer-direct',
        stage: 'AI Producer',
        progress: 15,
        message: 'Composing lyrics, arrangement & style tags...'
      }
    }));

    try {
      const composed = await api.producerCompose(prompt, undefined);
      const data: CompositionData = {
        title: composed.title || undefined,
        topic: composed.topic || prompt,
        lyrics: composed.lyrics || `[Verse 1]\n${prompt}\n[Chorus]\n${prompt}`,
        tags: composed.tags || 'Pop, Electronic, Synthwave',
        structuredCaption: composed.structured_caption,
        durationMs: 60000,
        temperature: 1.0,
        cfgScale: 1.5,
        topk: 50,
        llmModel: '',
        modelProvider: 'minimax_music3'
      };
      setProducerPreset(data);   // populate composer
      await handleGenerateMusic(data);
    } catch (e: any) {
      setIsGenerating(false);
      alert("Producer error: " + (e.response?.data?.detail || e.message));
    }
  };

  const handleGenerateLyrics = async (topic: string, model: string, currentLyrics?: string, tags?: string) => {
    setIsGeneratingLyrics(true);
    try {
      return await api.generateLyrics(topic, model, currentLyrics, tags);
    } finally {
      setIsGeneratingLyrics(false);
    }
  };

  const handleCancelJob = async (jobId?: string | null) => {
    try {
      if (jobId && jobId !== 'producer-direct') {
        await api.cancelJob(jobId);
      }
    } catch (e) {
      console.error("Failed to cancel job", e);
    } finally {
      setIsGenerating(false);
      setGenerationProgress(null);
      setCurrentJobId(null);
      handleRefresh();
      window.dispatchEvent(new CustomEvent('milimo_progress', {
        detail: {
          job_id: jobId || 'active',
          stage: 'Cancelled',
          progress: 100,
          message: 'Operation terminated by user.'
        }
      }));
    }
  };

  const handleOpenWorkspace = (job: Job) => {
    enginePause();
    setActiveWorkspaceJob(job);
    setCurrentNav('workspace');
  };

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await workspaceApi.uploadAndTranscribe(file);
      handleOpenWorkspace(res.job);
      handleRefresh();
    } catch (e) {
      alert("Audio upload and transcription failed");
    }
  };

  const handlePlaySong = (job: Job) => {
    if (!job.audio_path) return;
    const completed = history.filter(s => s.status === 'completed' && s.audio_path);
    if (engineTrack?.id === job.id) {
      engineTogglePlay(job);
    } else {
      enginePlayTrack(job, completed);
    }
  };

  return (
    <div className="h-screen w-full bg-[#f5f5f7] dark:bg-[#000000] flex overflow-hidden font-sans text-slate-900 dark:text-slate-100 select-none transition-colors duration-200">

      {/* Mobile Drawer Overlay Backdrop */}
      {isMobileMenuOpen && (
        <div
          onClick={() => setIsMobileMenuOpen(false)}
          className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm md:hidden animate-fade-in"
        />
      )}

      {/* 1. Apple-Style Persistent Reference Left Navigation Rail */}
      <nav
        className={`bg-white/85 dark:bg-[#12141c]/90 backdrop-blur-2xl border-r border-black/[0.06] dark:border-white/[0.08] flex flex-col justify-between flex-shrink-0 z-40 shadow-apple-sm dark:shadow-2xl transition-all duration-300 ease-in-out ${
          isMobileMenuOpen
            ? 'fixed inset-y-0 left-0 w-64 translate-x-0'
            : isLeftRailCollapsed
            ? 'w-18 md:w-20 hidden md:flex'
            : 'w-64 hidden md:flex'
        }`}
      >
        <div className="p-3.5 space-y-4 overflow-y-auto flex-1">
          {/* Header & Logo with Collapse Toggle */}
          <div className="flex items-center justify-between px-1 pt-1">
            {isLeftRailCollapsed ? (
              <div className="mx-auto">
                <MilimoLogo size="sm" showText={false} />
              </div>
            ) : (
              <>
                <MilimoLogo size="md" />
                <button
                  onClick={() => setIsLeftRailCollapsed(true)}
                  className="hidden md:block p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                  title="Collapse Sidebar"
                >
                  <PanelLeftClose size={15} />
                </button>
              </>
            )}
          </div>

          {/* Primary CTA: New Session + */}
          <button
            onClick={handleCreateNewSession}
            className={`w-full py-2.5 bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs rounded-xl flex items-center justify-center space-x-2 transition-all shadow-md shadow-teal-500/20 active:scale-[0.98] ${
              isLeftRailCollapsed ? 'px-0' : 'px-4'
            }`}
            title="New Session"
          >
            <Plus size={16} />
            {!isLeftRailCollapsed && <span>New session +</span>}
          </button>

          {/* Flat Reference Navigation (Songs, Projects, Playlists, Videos, Profile, DAW Workspace) */}
          <div className="space-y-1">
            {[
              { id: 'explore', label: 'Explore & Create', icon: Compass },
              { id: 'songs', label: 'Songs', icon: Music },
              { id: 'projects', label: 'Projects', icon: FolderKanban },
              { id: 'playlists', label: 'Playlists', icon: ListMusic },
              { id: 'videos', label: 'Music videos', icon: Video, badge: 'In Dev' },
              { id: 'profile', label: 'Profile', icon: User },
              { id: 'workspace', label: 'DAW Workspace', icon: Sliders }
            ].map(item => {
              const Icon = item.icon;
              const isActive = currentNav === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    if (item.id === 'workspace') {
                      if (activeWorkspaceJob) handleNavigate('workspace');
                      else if (history.length > 0) {
                        setActiveWorkspaceJob(history[0]);
                        handleNavigate('workspace');
                      }
                    } else {
                      handleNavigate(item.id as NavView);
                    }
                    setIsMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center justify-between rounded-xl text-xs font-semibold transition-all ${
                    isLeftRailCollapsed
                      ? 'justify-center py-2.5 px-0'
                      : 'px-3.5 py-2.5'
                  } ${
                    isActive
                      ? 'bg-black/[0.06] dark:bg-white/10 text-teal-600 dark:text-teal-300 font-bold shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5'
                  }`}
                  title={item.label}
                >
                  <div className="flex items-center space-x-3 truncate">
                    <Icon size={16} className="text-teal-500 dark:text-teal-400 flex-shrink-0" />
                    {!isLeftRailCollapsed && <span className="truncate">{item.label}</span>}
                  </div>
                  {!isLeftRailCollapsed && (item as any).badge && (
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded-md bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 font-bold">
                      {(item as any).badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Persistent Sessions Section (Conversation Threads) */}
          {!isLeftRailCollapsed && sessions.length > 0 && (
            <div className="pt-3 border-t border-black/[0.06] dark:border-white/5 space-y-1">
              <div className="flex items-center justify-between px-3 mb-1">
                <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                  Sessions
                </span>
                <button
                  onClick={handleCreateNewSession}
                  className="text-slate-400 hover:text-teal-500 p-0.5 rounded"
                  title="New Session"
                >
                  <Plus size={13} />
                </button>
              </div>
              <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
                {sessions.map(s => {
                  const isActive = activeSession?.id === s.id && currentNav === 'explore';
                  return (
                    <div
                      key={s.id}
                      onClick={() => handleSelectSession(s)}
                      className={`w-full text-left p-2 rounded-xl text-[11px] transition-colors flex items-center justify-between cursor-pointer group ${
                        isActive
                          ? 'bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20'
                          : 'hover:bg-black/[0.03] dark:hover:bg-white/5 text-slate-700 dark:text-slate-300'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0 flex-1">
                        <MessageSquare size={13} className="text-teal-500 flex-shrink-0" />
                        <span className="truncate">{s.title || 'Studio Session'}</span>
                      </div>
                      <button
                        onClick={(e) => handleDeleteSession(s.id, e)}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-500 transition-opacity ml-1 flex-shrink-0"
                        title="Delete session"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Production Engines Sub-menu */}
          <div className="pt-3 border-t border-black/[0.06] dark:border-white/5 space-y-1">
            {!isLeftRailCollapsed && (
              <span className="px-3 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider block mb-1">
                Production Engines
              </span>
            )}

            {[
              { label: 'Voice Studio', icon: Mic, color: 'text-teal-500 dark:text-teal-400', onClick: () => setIsVoiceStudioOpen(true) },
              { label: 'LoRA Studio', icon: GraduationCap, color: 'text-amber-500 dark:text-amber-400', badge: 'In Dev', onClick: () => setIsTrainingOpen(true) },
              { label: 'Models & HW', icon: Cpu, color: 'text-cyan-500 dark:text-cyan-400', onClick: () => setIsModelsManagerOpen(true) }
            ].map((engine, idx) => {
              const Icon = engine.icon;
              return (
                <button
                  key={idx}
                  onClick={() => {
                    engine.onClick();
                    setIsMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center rounded-xl text-xs font-medium text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/[0.03] dark:hover:bg-white/5 transition-colors ${
                    isLeftRailCollapsed
                      ? 'justify-center py-2 px-0'
                      : 'space-x-3 px-3.5 py-2'
                  }`}
                  title={engine.label}
                >
                  <Icon size={15} className={`${engine.color} flex-shrink-0`} />
                  {!isLeftRailCollapsed && <span className="truncate">{engine.label}</span>}
                  {!isLeftRailCollapsed && (engine as any).badge && (
                    <span className="text-[9px] font-mono px-1.5 py-0.2 rounded-full bg-amber-500/15 text-amber-600 dark:text-amber-400 font-bold border border-amber-500/20 ml-auto">
                      {(engine as any).badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Scrollable Un-grouped Sessions List (with waveform & MIDI badges) */}
          {!isLeftRailCollapsed && history.length > 0 && (
            <div className="pt-3 border-t border-black/[0.06] dark:border-white/5 space-y-1">
              <div className="flex items-center justify-between px-3 mb-1">
                <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                  Recent Sessions
                </span>
                <span className="text-[9px] font-mono text-teal-600 dark:text-teal-400">
                  {history.length} tracks
                </span>
              </div>
              <div className="space-y-1 max-h-44 overflow-y-auto pr-1">
                {history.slice(0, 8).map(session => {
                  const isReadyForDAW = session.status === 'completed';
                  return (
                    <button
                      key={session.id}
                      onClick={() => {
                        setActiveWorkspaceJob(session);
                        setCurrentNav('workspace');
                      }}
                      className="w-full text-left p-2 rounded-xl text-[11px] hover:bg-black/[0.03] dark:hover:bg-white/5 transition-colors flex items-center justify-between group"
                    >
                      <div className="truncate pr-2">
                        <div className="font-semibold text-slate-800 dark:text-slate-200 truncate">
                          {session.title || session.prompt.slice(0, 22)}
                        </div>
                        <div className="text-[9px] text-slate-400 font-mono">
                          {session.id.slice(0, 6)} · {session.tags?.split(',')[0] || 'Pop'}
                        </div>
                      </div>
                      <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded-full flex-shrink-0 ${
                        isReadyForDAW
                          ? 'bg-teal-500/10 text-teal-600 dark:text-teal-400 font-bold'
                          : 'bg-amber-500/10 text-amber-600'
                      }`}>
                        {isReadyForDAW ? '🎼 MIDI' : '⏳ Gen'}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Left Rail Footer: User Profile Card, Import, Theme & Settings */}
        <div className="p-3 border-t border-black/[0.06] dark:border-white/[0.08] space-y-2.5 bg-black/[0.01] dark:bg-white/[0.01]">
          {/* User Account / Profile Card */}
          {!isLeftRailCollapsed && (
            <button
              onClick={() => setCurrentNav('profile')}
              className="w-full p-2 rounded-xl bg-black/[0.02] dark:bg-white/[0.03] hover:bg-black/[0.05] dark:hover:bg-white/[0.07] border border-black/[0.04] dark:border-white/5 flex items-center space-x-2.5 transition-all text-left group"
            >
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-teal-500 to-cyan-500 flex items-center justify-center text-slate-950 font-bold text-xs shadow-sm flex-shrink-0">
                MK
              </div>
              <div className="truncate flex-1">
                <div className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                  Mainza Kangombe
                </div>
                <div className="text-[10px] text-slate-400 font-mono truncate">
                  Studio Master · Local GPU
                </div>
              </div>
            </button>
          )}

          {/* Audio Upload */}
          <label
            className={`w-full py-2 bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.07] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 rounded-xl text-xs font-medium text-slate-700 dark:text-slate-300 flex items-center justify-center cursor-pointer transition-colors shadow-sm ${
              isLeftRailCollapsed ? 'px-0' : 'px-3 space-x-2'
            }`}
            title="Import & Transcribe Audio"
          >
            <Upload size={14} className="text-teal-500 dark:text-teal-400 flex-shrink-0" />
            {!isLeftRailCollapsed && <span>Import & Transcribe</span>}
            <input type="file" accept="audio/*" onChange={handleAudioUpload} className="hidden" />
          </label>

          {/* Theme Switcher */}
          {!isLeftRailCollapsed ? (
            <div className="flex items-center justify-between bg-black/[0.04] dark:bg-white/5 p-1 rounded-xl border border-black/[0.06] dark:border-white/10">
              <button
                onClick={() => setTheme('system')}
                className={`flex-1 flex items-center justify-center py-1 rounded-lg text-[10px] font-semibold transition-all ${
                  theme === 'system'
                    ? 'bg-white dark:bg-white/20 text-teal-600 dark:text-teal-300 shadow-sm'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                }`}
                title="Follow System Theme"
              >
                <Laptop size={12} className="mr-1" />
                Auto
              </button>

              <button
                onClick={() => setTheme('light')}
                className={`flex-1 flex items-center justify-center py-1 rounded-lg text-[10px] font-semibold transition-all ${
                  theme === 'light'
                    ? 'bg-white text-teal-600 shadow-sm'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                }`}
                title="Light Mode"
              >
                <Sun size={12} className="mr-1" />
                Light
              </button>

              <button
                onClick={() => setTheme('dark')}
                className={`flex-1 flex items-center justify-center py-1 rounded-lg text-[10px] font-semibold transition-all ${
                  theme === 'dark'
                    ? 'bg-white/20 text-teal-300 shadow-sm'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                }`}
                title="Dark Mode"
              >
                <Moon size={12} className="mr-1" />
                Dark
              </button>
            </div>
          ) : (
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark')}
              className="w-full flex items-center justify-center py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-300"
              title={`Theme: ${theme}`}
            >
              {theme === 'dark' ? <Moon size={14} /> : theme === 'light' ? <Sun size={14} /> : <Laptop size={14} />}
            </button>
          )}

          {/* Expand / Collapse bottom toggle */}
          {isLeftRailCollapsed && (
            <button
              onClick={() => setIsLeftRailCollapsed(false)}
              className="w-full flex items-center justify-center py-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              title="Expand Sidebar"
            >
              <PanelLeftOpen size={15} />
            </button>
          )}

          {!isLeftRailCollapsed && (
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 px-1 pt-1">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
                <span className="text-[11px] font-medium truncate">MiniMax Engine Ready</span>
              </div>
              <button
                onClick={() => setIsSettingsOpen(true)}
                className="p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
                title="LLM Settings"
              >
                <Settings size={15} />
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* 2. Main Content Center Stage */}
      <main className="flex-1 h-full overflow-hidden flex flex-col relative z-10 bg-[#fbfbfd] dark:bg-[#0c0e14] transition-colors duration-200 min-w-0">
        {/* Top Responsive Mobile / Toolbar Strip */}
        <header className="flex md:hidden items-center justify-between px-4 py-2.5 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/70 dark:bg-[#12141c]/80 backdrop-blur-xl z-20">
          <button
            onClick={() => setIsMobileMenuOpen(true)}
            className="p-1.5 rounded-lg text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10"
          >
            <Menu size={18} />
          </button>
          <MilimoLogo size="sm" />
          <button
            onClick={() => setIsComposerOpen(!isComposerOpen)}
            className="p-1.5 rounded-lg text-teal-600 dark:text-teal-400 hover:bg-black/5 dark:hover:bg-white/10"
            title="Toggle Composer"
          >
            <Sparkles size={18} />
          </button>
        </header>

        {/* Top Responsive Desktop / Tablet Header */}
        {currentNav !== 'workspace' && (
          <header className="hidden md:flex items-center justify-between px-6 py-2.5 border-b border-black/[0.04] dark:border-white/[0.06] bg-white/40 dark:bg-[#12141c]/50 backdrop-blur-xl flex-shrink-0 z-20 transition-all">
            <div className="flex items-center space-x-3">
              {isLeftRailCollapsed && (
                <button
                  onClick={() => setIsLeftRailCollapsed(false)}
                  className="px-2.5 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/10 text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-black/[0.08] dark:hover:bg-white/10 shadow-sm transition-all flex items-center gap-1.5 text-xs font-semibold"
                  title="Expand Navigation Rail"
                  aria-label="Expand Navigation Rail"
                >
                  <PanelLeftOpen size={14} />
                  <span>Show Sidebar</span>
                </button>
              )}
            </div>

            <div className="flex items-center space-x-2 ml-auto">
              <button
                onClick={() => setIsComposerOpen(!isComposerOpen)}
                className={`px-3 py-1.5 rounded-xl border text-xs font-bold flex items-center space-x-1.5 transition-all shadow-sm ${
                  isComposerOpen
                    ? 'bg-black/[0.04] dark:bg-white/5 border-black/[0.06] dark:border-white/10 text-slate-600 dark:text-slate-300 hover:bg-black/[0.08] dark:hover:bg-white/10'
                    : 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 border-teal-400 shadow-md shadow-teal-500/20 active:scale-95'
                }`}
                title={isComposerOpen ? 'Hide Composer Sidebar' : 'Open Studio Composer Sidebar'}
                aria-label={isComposerOpen ? 'Hide Composer' : 'Open Composer'}
              >
                {isComposerOpen ? <PanelRightClose size={14} /> : <Sparkles size={14} />}
                <span>{isComposerOpen ? 'Hide Composer' : 'Composer'}</span>
              </button>
            </div>
          </header>
        )}

        {/* Active Studio Generation HUD / Real-Time Progress Notification Banner */}
        {isGenerating && (
          <div className="mx-4 sm:mx-6 mt-3 p-3.5 sm:p-4 rounded-2xl bg-white/95 dark:bg-[#151824]/95 border border-teal-500/40 shadow-apple-lg backdrop-blur-2xl animate-slide-down flex flex-col gap-2.5 z-30">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center space-x-3 min-w-0">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-teal-500 to-cyan-400 p-0.5 flex items-center justify-center flex-shrink-0 animate-pulse">
                  <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                    <Sparkles size={16} className="text-teal-400 animate-spin-slow" />
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                      Synthesizing Master & Multi-Stem Transcription
                    </h4>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/20">
                      Step {generationProgress?.step || 1}/4
                    </span>
                  </div>
                  <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium truncate mt-0.5">
                    {generationProgress?.message || 'Synthesizing track audio and neural stems...'}
                  </p>
                </div>
              </div>

              <button
                onClick={() => handleCancelJob(currentJobId)}
                className="px-3 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 text-xs font-bold transition-all flex items-center gap-1.5 border border-rose-500/20 shadow-sm active:scale-95 flex-shrink-0"
                title="Cancel generation"
                aria-label="Cancel Generation"
              >
                <Square size={11} className="fill-current" />
                <span>Cancel</span>
              </button>
            </div>

            {/* Glowing Studio Progress Bar */}
            <div className="w-full h-2 bg-black/[0.04] dark:bg-white/5 rounded-full overflow-hidden relative">
              <div
                className="h-full rounded-full bg-gradient-to-r from-teal-500 via-cyan-400 to-sky-500 transition-all duration-300 shadow-[0_0_12px_rgba(20,184,166,0.6)]"
                style={{ width: `${Math.max(8, generationProgress?.progress || 15)}%` }}
              />
            </div>
          </div>
        )}

        {/* View Switching Logic */}
        {currentNav === 'workspace' && activeWorkspaceJob ? (
          <div className="flex-1 overflow-hidden flex flex-col min-w-0">
            <SessionWorkspace
              job={activeWorkspaceJob}
              onClose={() => setCurrentNav(previousNav || 'explore')}
            />
          </div>
        ) : currentNav === 'track-detail' && selectedTrack ? (
          <TrackDetailView
            track={selectedTrack}
            allJobs={history}
            isPlaying={isPlayingAudio && playingSong?.id === selectedTrack.id}
            playingSongId={playingSong?.id}
            onBack={() => setCurrentNav(previousNav || 'songs')}
            onPlay={handlePlaySong}
            onOpenWorkspace={handleOpenWorkspace}
            onExtend={(job) => {
              setParentJob(job);
              setCurrentNav('explore');
              if (!isComposerOpen) setIsComposerOpen(true);
            }}
            onReroll={(preset) => {
              setProducerPreset(preset);
              setCurrentNav('explore');
              if (!isComposerOpen) setIsComposerOpen(true);
            }}
            onToggleFavorite={handleToggleFavorite}
            onTrackUpdated={(updated) => {
              setSelectedTrack(updated);
              setHistory(prev => prev.map(j => j.id === updated.id ? updated : j));
            }}
            onSelectTrack={handleSelectTrack}
          />
        ) : currentNav === 'songs' ? (
          <SongsView
            songs={history}
            currentJobId={playingSong?.id}
            onPlay={handlePlaySong}
            onOpenWorkspace={handleOpenWorkspace}
            onToggleFavorite={handleToggleFavorite}
            onExtend={(job) => {
              setParentJob(job);
              setCurrentNav('explore');
              if (!isComposerOpen) setIsComposerOpen(true);
            }}
            onDelete={handleDeleteJob}
            onSelectTrack={handleSelectTrack}
          />
        ) : currentNav === 'projects' ? (
          <ProjectsView
            allJobs={history}
            onOpenWorkspace={handleOpenWorkspace}
            onPlay={handlePlaySong}
            playingSongId={playingSong?.id}
            isPlayingAudio={isPlayingAudio}
            onGenerateInProject={(project) => {
              setActiveProject(project);
              setProducerInput(project.description || `${project.name} session`);
              setCurrentNav('explore');
              if (!isComposerOpen) setIsComposerOpen(true);
            }}
            onSelectTrack={handleSelectTrack}
          />
        ) : currentNav === 'playlists' ? (
          <PlaylistsView
            songs={history}
            onPlaySong={handlePlaySong}
            onOpenWorkspace={handleOpenWorkspace}
            onSelectTrack={handleSelectTrack}
          />
        ) : currentNav === 'videos' ? (
          <MusicVideosView
            songs={history}
            onPlay={handlePlaySong}
          />
        ) : currentNav === 'profile' ? (
          <ProfileView
            songs={history}
            onPlaySong={handlePlaySong}
            onOpenWorkspace={handleOpenWorkspace}
            onSelectTrack={handleSelectTrack}
          />
        ) : currentNav === 'sessions' ? (
          <div className="flex-1 overflow-hidden flex flex-col min-w-0">
            <HistoryFeed
              history={history}
              currentJobId={currentJobId}
              onRefresh={handleRefresh}
              onExtend={(job) => {
                setParentJob(job);
                setCurrentNav('explore');
                if (!isComposerOpen) setIsComposerOpen(true);
              }}
              onOpenWorkspace={handleOpenWorkspace}
              onLoadMore={handleLoadMore}
              hasMore={hasMoreHistory}
              onFilterChange={handleFilterChange}
              currentFilter={historyFilter}
              onSearch={handleSearch}
              searchQuery={searchQuery}
              isLoadingMore={isLoadingHistory}
              onToggleFavorite={handleToggleFavorite}
              onDelete={handleDeleteJob}
              onSelectTrack={handleSelectTrack}
            />
          </div>
        ) : (
          /* Explore & New Session Stage */
          <div className="flex-1 overflow-y-auto flex flex-col justify-between p-4 sm:p-6 md:p-8 min-w-0 relative">
            {/* Hidden Attachment Input */}
            <input
              type="file"
              ref={attachmentInputRef}
              accept="audio/*,image/*"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (file) {
                  if (file.type.startsWith('image/')) {
                    const res = await coverApi.uploadCoverImage(file);
                    setAttachmentPath(res.url);
                  } else {
                    const res = await workspaceApi.uploadAndTranscribe(file);
                    setAttachmentPath(res.job.audio_path || null);
                  }
                }
              }}
              className="hidden"
            />

            <div className="max-w-3xl w-full mx-auto space-y-6 md:space-y-8 pt-4 md:pt-6">
              {/* If Session Conversation Exists: Show Chat Stream */}
              {activeSession && activeSession.messages && activeSession.messages.length > 0 ? (
                <div className="space-y-4 pb-6 animate-fade-in">
                  <div className="flex items-center justify-between pb-3 border-b border-black/[0.06] dark:border-white/[0.08]">
                    <div className="flex items-center gap-2">
                      <MessageSquare size={16} className="text-teal-500" />
                      <h2 className="text-base font-bold text-slate-900 dark:text-white">
                        {activeSession.title || 'Studio Session'}
                      </h2>
                    </div>
                    <button
                      onClick={handleCreateNewSession}
                      className="px-3 py-1.5 rounded-xl bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 text-xs font-semibold text-slate-600 dark:text-slate-300 transition-colors"
                    >
                      + New Session
                    </button>
                  </div>

                  <div className="space-y-3">
                    {activeSession.messages.map((msg, idx) => {
                      const isUser = msg.role === 'user';
                      return (
                        <div
                          key={msg.id || idx}
                          className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
                        >
                          <div
                            className={`max-w-[85%] sm:max-w-[75%] p-4 rounded-2xl text-xs sm:text-sm leading-relaxed ${
                              isUser
                                ? 'bg-teal-500 text-slate-950 font-medium rounded-br-sm shadow-md'
                                : 'bg-white dark:bg-[#181a24] text-slate-800 dark:text-slate-200 border border-black/[0.08] dark:border-white/10 rounded-bl-sm shadow-apple-sm'
                            }`}
                          >
                            {!isUser && (
                              <div className="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-black/[0.04] dark:border-white/[0.06]">
                                <div className="flex items-center gap-1.5 text-[11px] font-bold text-teal-600 dark:text-teal-400">
                                  <Sparkles size={13} />
                                  <span>Producer AI</span>
                                </div>
                                {msg.preset_data_json && (
                                  <div className="flex items-center gap-1.5">
                                    <button
                                      onClick={() => {
                                        try {
                                          const p = JSON.parse(msg.preset_data_json!);
                                          setProducerPreset(p);
                                          if (!isComposerOpen) setIsComposerOpen(true);
                                        } catch (e) {
                                          console.error(e);
                                        }
                                      }}
                                      className="px-2 py-0.5 rounded-lg bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 text-[10px] font-bold transition-colors"
                                      title="Load into Composer"
                                    >
                                      Load in Composer
                                    </button>
                                  </div>
                                )}
                              </div>
                            )}
                            <p className="whitespace-pre-wrap font-sans text-xs leading-relaxed">{msg.content}</p>

                            {msg.audio_attachment_path && (
                              <div className="mt-2 p-2 rounded-xl bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 flex items-center gap-2 text-[11px] font-mono">
                                <FileAudio size={14} className="text-teal-500" />
                                <span className="truncate">{msg.audio_attachment_path.split('/').pop()}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}

                    {isChatSubmitting && (
                      <div className="flex justify-start animate-slide-up">
                        <div className="p-4 rounded-2xl bg-white/95 dark:bg-[#181a24]/95 border border-teal-500/30 dark:border-teal-500/20 rounded-bl-sm shadow-apple-md flex flex-col gap-2.5 max-w-sm w-full backdrop-blur-xl">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping" />
                              <span className="text-[11px] font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider">
                                AI Producer Composing
                              </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <Sparkles size={13} className="text-teal-500 animate-pulse" />
                              <button
                                onClick={handleCancelChatSubmission}
                                className="p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-rose-500 transition-colors"
                                title="Cancel Producer generation"
                              >
                                <X size={13} />
                              </button>
                            </div>
                          </div>
                          <p className="text-xs text-slate-700 dark:text-slate-200 font-sans flex items-center gap-2">
                            <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-teal-500 border-t-transparent animate-spin flex-shrink-0" />
                            <span className="truncate font-medium">{producerStatusStage}</span>
                          </p>
                          <div className="w-full bg-black/5 dark:bg-white/10 rounded-full h-1 overflow-hidden">
                            <div className="bg-gradient-to-r from-teal-500 to-cyan-400 h-full w-3/4 animate-pulse rounded-full" />
                          </div>
                          <div className="flex justify-end">
                            <button
                              onClick={handleCancelChatSubmission}
                              className="text-[11px] font-bold text-rose-500 hover:text-rose-600 hover:underline flex items-center gap-1 transition-colors"
                            >
                              <Square size={10} className="fill-current" />
                              <span>Stop generating</span>
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* New Session Hero & 3 Visual Action Cards (Apple Design) */
                <div className="text-center space-y-6 pt-2">
                  <div className="flex justify-center">
                    <MilimoLogo size="lg" showText={true} />
                  </div>

                  <div className="space-y-2">
                    <h2 className="text-2xl sm:text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white font-sans">
                      Give the silence something worth remembering.
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 max-w-lg mx-auto leading-relaxed">
                      Speak it into being. Shape it until it&apos;s yours.
                    </p>
                  </div>

                  {/* 3 Visual Starter Action Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 pt-2 text-left">
                    {/* Card 1: Brainstorm lyrics */}
                    <div
                      onClick={() => {
                        setProducerInput('Brainstorm lyrics for an emotional indie pop ballad');
                        if (!isComposerOpen) setIsComposerOpen(true);
                      }}
                      className="p-4 rounded-2xl bg-white/70 dark:bg-[#181a24]/80 border border-black/[0.08] dark:border-white/10 hover:border-teal-500/50 dark:hover:border-teal-500/50 shadow-apple-sm hover:shadow-apple-md cursor-pointer transition-all hover:scale-[1.02] group flex flex-col justify-between"
                    >
                      <div className="space-y-2">
                        <div className="w-9 h-9 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400 flex items-center justify-center font-bold">
                          ✍️
                        </div>
                        <h3 className="text-xs font-bold text-slate-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                          Brainstorm lyrics
                        </h3>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                          Generate verses, choruses, and rhymes with AI Co-Writer
                        </p>
                      </div>
                    </div>

                    {/* Card 2: Create a song together */}
                    <div
                      onClick={() => {
                        handleProducerGenerate('Create a vibrant synthwave pop track with driving bass and energetic drums');
                      }}
                      className="p-4 rounded-2xl bg-white/70 dark:bg-[#181a24]/80 border border-black/[0.08] dark:border-white/10 hover:border-teal-500/50 dark:hover:border-teal-500/50 shadow-apple-sm hover:shadow-apple-md cursor-pointer transition-all hover:scale-[1.02] group flex flex-col justify-between"
                    >
                      <div className="space-y-2">
                        <div className="w-9 h-9 rounded-xl bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 flex items-center justify-center font-bold">
                          🎹
                        </div>
                        <h3 className="text-xs font-bold text-slate-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                          Create a song together
                        </h3>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                          Collaborate with Producer on arrangement, style, and instrumentation
                        </p>
                      </div>
                    </div>

                    {/* Card 3: Remix my music */}
                    <div
                      onClick={() => {
                        const fileInput = document.querySelector('input[type="file"][accept="audio/*"]') as HTMLInputElement;
                        fileInput?.click();
                      }}
                      className="p-4 rounded-2xl bg-white/70 dark:bg-[#181a24]/80 border border-black/[0.08] dark:border-white/10 hover:border-teal-500/50 dark:hover:border-teal-500/50 shadow-apple-sm hover:shadow-apple-md cursor-pointer transition-all hover:scale-[1.02] group flex flex-col justify-between"
                    >
                      <div className="space-y-2">
                        <div className="w-9 h-9 rounded-xl bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center font-bold">
                          🎛️
                        </div>
                        <h3 className="text-xs font-bold text-slate-900 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors">
                          Remix my music
                        </h3>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                          Upload audio, separate stems, change vocals, and rearrange
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Sticky Bottom Conversational Producer Prompt Bar (Apple Design) */}
            <div className="max-w-3xl w-full mx-auto pt-6 pb-2 sticky bottom-0 z-20">
              {attachmentPath && (
                <div className="mb-2 p-2 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-between text-xs text-teal-700 dark:text-teal-300 font-mono animate-slide-up">
                  <div className="flex items-center gap-2 truncate">
                    <Paperclip size={13} />
                    <span className="truncate">Attached: {attachmentPath.split('/').pop()}</span>
                  </div>
                  <button onClick={() => setAttachmentPath(null)} className="p-1 hover:text-rose-500 text-xs">✕</button>
                </div>
              )}

              <div className="bg-white/80 dark:bg-[#181a24]/90 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] shadow-2xl backdrop-blur-2xl p-2 sm:p-2.5 flex items-center gap-2 transition-all focus-within:border-teal-500/60 focus-within:shadow-[0_0_20px_rgba(20,184,166,0.2)]">
                {/* Left Action Buttons */}
                <button
                  onClick={() => attachmentInputRef.current?.click()}
                  className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                  title="Attach Audio or Image Reference"
                >
                  <Plus size={18} />
                </button>

                <button
                  onClick={() => setIsComposerOpen(!isComposerOpen)}
                  className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                  title="Open Sound & Lyrics Controls"
                >
                  <Sliders size={18} />
                </button>

                {/* Center Text Input */}
                <input
                  type="text"
                  placeholder="Describe a song, artist, style, or upload audio..."
                  value={producerInput}
                  onChange={(e) => setProducerInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (producerInput || attachmentPath) && !isChatSubmitting) {
                      handleSendChatMessage(producerInput);
                    }
                  }}
                  className="flex-1 bg-transparent px-2 py-2 text-xs sm:text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none min-w-0"
                />

                {/* Right Buttons: Mic & Send Arrow */}
                <button
                  onClick={() => setProducerInput('Vocal lead with acoustic guitar and ambient reverb')}
                  className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                  title="Voice Input"
                >
                  <Mic size={18} />
                </button>

                {isChatSubmitting ? (
                  <button
                    type="button"
                    onClick={handleCancelChatSubmission}
                    className="w-9 h-9 rounded-full bg-rose-500 hover:bg-rose-600 text-white font-bold flex items-center justify-center transition-all shadow-md shadow-rose-500/25 active:scale-95 flex-shrink-0 animate-pulse"
                    title="Stop / Terminate Producer thinking"
                    aria-label="Stop generation"
                  >
                    <Square size={12} className="fill-current" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      if ((producerInput || attachmentPath) && !isChatSubmitting) {
                        handleSendChatMessage(producerInput);
                      }
                    }}
                    disabled={!producerInput.trim() && !attachmentPath}
                    className="w-9 h-9 rounded-full bg-teal-500 hover:bg-teal-400 disabled:opacity-40 text-slate-950 font-bold flex items-center justify-center transition-all shadow-md shadow-teal-500/20 active:scale-95 flex-shrink-0"
                    title="Send message to Producer"
                  >
                    <ArrowUp size={18} />
                  </button>
                )}
              </div>
            </div>

            {/* Explore Feed */}
            <div className="pt-6">
              <HistoryFeed
                history={history}
                currentJobId={currentJobId}
                onRefresh={handleRefresh}
                onExtend={(job) => {
                  setParentJob(job);
                  if (!isComposerOpen) setIsComposerOpen(true);
                }}
                onOpenWorkspace={(job) => {
                  setActiveWorkspaceJob(job);
                  setCurrentNav('workspace');
                }}
                onLoadMore={handleLoadMore}
                hasMore={hasMoreHistory}
                onFilterChange={handleFilterChange}
                currentFilter={historyFilter}
                onSearch={handleSearch}
                searchQuery={searchQuery}
                isLoadingMore={isLoadingHistory}
                onToggleFavorite={handleToggleFavorite}
                onDelete={handleDeleteJob}
                onSelectTrack={handleSelectTrack}
              />

              {/* Global Creator Footer */}
              <AppFooter />
            </div>
          </div>
        )}
      </main>

      {/* 3. Right Slide-Over Composer Panel (Adaptive & Collapsible) */}
      <aside
        className={`h-full z-30 shadow-apple-lg border-l border-black/[0.06] dark:border-white/[0.08] bg-white/95 dark:bg-[#12141c]/95 backdrop-blur-2xl transition-all duration-300 ease-in-out flex-shrink-0 ${
          isComposerOpen
            ? 'w-full sm:w-[380px] md:w-[400px] xl:w-[420px] fixed sm:static inset-y-0 right-0 translate-x-0'
            : 'w-0 translate-x-full sm:translate-x-0 sm:w-0 overflow-hidden border-l-0 pointer-events-none'
        }`}
      >
        <div className="h-full w-full sm:w-[380px] md:w-[400px] xl:w-[420px] relative">
          {/* Close button for responsive mobile / overlay mode */}
          <button
            onClick={() => setIsComposerOpen(false)}
            className="sm:hidden absolute top-4 right-4 z-40 p-1.5 rounded-full bg-black/5 dark:bg-white/10 text-slate-500"
          >
            <X size={16} />
          </button>

          <ComposerSidebar
            onGenerate={handleGenerateMusic}
            isGenerating={isGenerating}
            lyricsModels={lyricsModels}
            onRefreshModels={() => api.getLyricsModels().then(setLyricsModels)}
            onGenerateLyrics={handleGenerateLyrics}
            isGeneratingLyrics={isGeneratingLyrics}
            currentJobId={currentJobId || undefined}
            onCancel={handleCancelJob}
            parentJob={parentJob}
            onClearParentJob={() => setParentJob(undefined)}
            onOpenTraining={() => setIsTrainingOpen(true)}
            activeCheckpoint={activeCheckpoint}
            activeProject={activeProject}
            onClearActiveProject={() => setActiveProject(null)}
            producerPreset={producerPreset}
          />
        </div>
      </aside>

      {/* Global Modals & Monitor */}
      <TrainingStudio
        isOpen={isTrainingOpen}
        onClose={() => setIsTrainingOpen(false)}
        onCheckpointsChange={refreshActiveCheckpoint}
      />

      <VoiceStudioModal
        isOpen={isVoiceStudioOpen}
        onClose={() => setIsVoiceStudioOpen(false)}
      />

      <ModelsManagerModal
        isOpen={isModelsManagerOpen}
        onClose={() => setIsModelsManagerOpen(false)}
      />

      <LLMSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        currentConfig={{}}
        onConfigUpdate={() => {
          api.getLyricsModels().then(setLyricsModels).catch(console.error);
        }}
      />

      {/* Global Apple Studio Dock Player: Contextually hidden on Track Studio & DAW Workspace views */}
      {currentNav !== 'track-detail' && currentNav !== 'workspace' && engineTrack && (
        <GlobalAudioPlayer
          onOpenWorkspace={(job) => {
            enginePause();
            setActiveWorkspaceJob(job);
            setCurrentNav('workspace');
          }}
          onSelectTrack={handleSelectTrack}
        />
      )}

      <FloatingStatusWidget />
    </div>
  );
}

export default App;
