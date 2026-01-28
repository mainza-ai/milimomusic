import { useState, useEffect } from 'react';
import { api, trainingApi } from './api';
import type { Job } from './api';
import { ComposerSidebar } from './components/ComposerSidebar';
import type { CompositionData } from './components/ComposerSidebar';
import { HistoryFeed } from './components/HistoryFeed';

import { TrainingStudio } from './components/TrainingStudio';
import { useTrainingMonitor } from './hooks/useTrainingMonitor'; // Imported in Step 1956

function App() {
  const [lyricsModels, setLyricsModels] = useState<string[]>([]);
  const [history, setHistory] = useState<Job[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [parentJob, setParentJob] = useState<Job | undefined>(undefined);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGeneratingLyrics, setIsGeneratingLyrics] = useState(false);

  // Training Global State
  const [isTrainingOpen, setIsTrainingOpen] = useState(false);
  const [activeCheckpoint, setActiveCheckpoint] = useState<{ name: string, id: string } | null>(null);
  const { activeJob } = useTrainingMonitor();

  // Refresh active checkpoint
  const refreshActiveCheckpoint = async () => {
    try {
      const checkpoints = await trainingApi.listCheckpoints();
      const active = checkpoints.find((c: { is_active: boolean }) => c.is_active);
      setActiveCheckpoint(active ? { name: active.name, id: active.id } : null);
    } catch (e) {
      console.error("Failed to load active checkpoint", e);
    }
  };

  useEffect(() => {
    refreshActiveCheckpoint();
  }, []);

  // Pagination State
  const [historyOffset, setHistoryOffset] = useState(0);
  const [historyFilter, setHistoryFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [hasMoreHistory, setHasMoreHistory] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const HISTORY_LIMIT = 20;

  // Initial Load
  useEffect(() => {
    api.getLyricsModels().then(setLyricsModels).catch(console.error);
    loadHistory(0, 'all', '', true);
  }, []);

  // loadHistory function
  const loadHistory = async (offset: number, filter: string, search: string, replace: boolean = false) => {
    if (isLoadingHistory && offset !== 0) return; // Prevent double loading for pagination, but allow rapid filter switching (handled by replace)
    setIsLoadingHistory(true);
    try {
      const jobs = await api.getHistory(HISTORY_LIMIT, offset, filter, search);
      if (jobs.length < HISTORY_LIMIT) {
        setHasMoreHistory(false);
      } else {
        setHasMoreHistory(true);
      }

      if (replace) {
        setHistory(jobs);
      } else {
        setHistory(prev => {
          // Deduplicate just in case
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

  const handleRefresh = () => {
    // Reload current view
    loadHistory(0, historyFilter, searchQuery, true);
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
    setHistory([]); // Clear immediately for feedback
    loadHistory(0, newStatus, searchQuery, true);
  };

  const handleToggleFavorite = (jobId: string) => {
    // 1. Optimistic Update
    setHistory(prevHistory =>
      prevHistory.map(job =>
        job.id === jobId ? { ...job, is_favorite: !job.is_favorite } : job
      )
    );

    // 2. Background API Call
    api.toggleFavorite(jobId).catch(err => {
      console.error("Failed to toggle favorite", err);
      // Revert on failure
      setHistory(prevHistory =>
        prevHistory.map(job =>
          job.id === jobId ? { ...job, is_favorite: !job.is_favorite } : job
        )
      );
    });
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setHistoryOffset(0);
    setHistory([]);
    loadHistory(0, historyFilter, query, true);
  };

  // SSE Connection
  useEffect(() => {
    console.log("Connecting to SSE...");
    const evtSource = api.connectToEvents((e) => {
      try {
        const type = e.type;
        const data = JSON.parse(e.data);
        console.log("SSE Event:", type, data);

        if (type === 'job_update') {
          // Status change (processing -> completed, etc)
          if (data.status === 'completed') {
            setIsGenerating(false);
            setCurrentJobId(null);
            handleRefresh();
          } else if (data.status === 'failed') {
            setIsGenerating(false);
            setCurrentJobId(null);
            alert(`Generation Failed: ${data.error || "Unknown error"}`);
            handleRefresh();
          } else if (data.status === 'processing') {
            // ensure we track it
            setCurrentJobId(data.job_id);
            setIsGenerating(true);
            handleRefresh();
          }
        }

        if (type === 'job_progress') {
          // Broadcast this to children via a custom event
          window.dispatchEvent(new CustomEvent('milimo_progress', { detail: data }));
        }

      } catch (err) {
        console.error("SSE Parse Error", err);
      }
    });

    return () => {
      evtSource.close();
    };
  }, []);


  const handleGenerateMusic = async (data: CompositionData) => {
    setIsGenerating(true);
    try {
      // optimistic update logic could go here, but for now we rely on polling
      console.log("Starting generation with data:", data);
      const { job_id } = await api.generateJob(
        data.topic,
        data.durationMs,
        data.lyrics,
        data.tags,
        data.cfgScale,
        data.temperature,
        data.topk,
        data.llmModel,
        parentJob?.id,
        parentJob?.seed
      );
      console.log("Generation started, Job ID:", job_id);
      setCurrentJobId(job_id);
      setParentJob(undefined); // Clear extension mode after starting
    } catch (e: any) {
      console.error("Generation failed", e);
      let errorMsg = e.message;
      if (e.response && e.response.data && e.response.data.detail) {
        // If detail is array (validation errors), stringify it
        if (Array.isArray(e.response.data.detail)) {
          errorMsg = "Validation Error: " + e.response.data.detail.map((err: any) => `${err.loc.join('.')} - ${err.msg}`).join(', ');
        } else {
          errorMsg = e.response.data.detail;
        }
      }
      alert("Generation failed: " + errorMsg);
      setIsGenerating(false);
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

  const handleCancelJob = async (jobId: string) => {
    if (!confirm("Are you sure you want to stop generation?")) return;
    try {
      await api.cancelJob(jobId);
      // Optimistic update
      setIsGenerating(false);
      setCurrentJobId(null);
      handleRefresh(); // Refresh history after cancellation
    } catch (e) {
      console.error("Failed to cancel", e);
      alert("Failed to cancel job");
    }
  };

  const handleExtendJob = (job: Job) => {
    // Enter extension mode
    // 1. Set parent Job
    setParentJob(job);
    // 2. Pre-fill happens in ComposerSidebar via useEffect

    // Let's scroll to top?
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleClearParentJob = () => {
    setParentJob(undefined);
  };

  const refreshModels = () => {
    api.getLyricsModels()
      .then(setLyricsModels)
      .catch(e => {
        console.error("Failed to refresh models", e);
        setLyricsModels([]); // Clear to avoid showing stale models
      });
  };

  return (
    <div className="h-screen w-full bg-slate-50 flex flex-col md:flex-row overflow-hidden font-sans text-slate-900 selection:bg-indigo-100 selection:text-indigo-900">

      {/* Left Column: History Feed (Main Content) */}
      <main className="flex-1 h-full relative z-0 order-2 md:order-1 overflow-hidden">
        <div className="absolute inset-0 mesh-bg opacity-30 -z-10" />
        <HistoryFeed
          history={history}
          currentJobId={currentJobId}
          onRefresh={handleRefresh}
          onExtend={handleExtendJob} // Phase 9
          onLoadMore={handleLoadMore}
          hasMore={hasMoreHistory}
          onFilterChange={handleFilterChange}
          currentFilter={historyFilter}
          onSearch={handleSearch}
          searchQuery={searchQuery}
          isLoadingMore={isLoadingHistory}
          onToggleFavorite={handleToggleFavorite}
        />
      </main>

      {/* Right Column: Composer Sidebar (Fixed Width on Desktop, Full on Mobile) */}
      <aside className="w-full md:w-[400px] h-auto md:h-full relative z-10 flex-shrink-0 order-1 md:order-2">
        <ComposerSidebar
          onGenerate={handleGenerateMusic}
          isGenerating={isGenerating}
          lyricsModels={lyricsModels}
          onRefreshModels={refreshModels}
          onGenerateLyrics={handleGenerateLyrics}
          isGeneratingLyrics={isGeneratingLyrics}
          currentJobId={currentJobId || undefined}
          onCancel={handleCancelJob}
          parentJob={parentJob}
          onClearParentJob={handleClearParentJob}
          onOpenTraining={() => setIsTrainingOpen(true)}
          activeCheckpoint={activeCheckpoint}
        />
      </aside>

      {/* Global Training Modal */}
      <TrainingStudio
        isOpen={isTrainingOpen}
        onClose={() => setIsTrainingOpen(false)}
        onCheckpointsChange={refreshActiveCheckpoint}
      />

      {/* Persistent Training Status Widget (When Modal Closed) */}
      {!isTrainingOpen && activeJob && (() => {
        // Time calculations for widget
        const getElapsed = () => {
          if (!activeJob.started_at) return '';
          const start = new Date(activeJob.started_at).getTime();
          const elapsed = Math.floor((Date.now() - start) / 1000);
          const h = Math.floor(elapsed / 3600);
          const m = Math.floor((elapsed % 3600) / 60);
          return h > 0 ? `${h}h ${m}m` : `${m}m`;
        };
        const getETA = () => {
          if (!activeJob.started_at || activeJob.progress <= 0) return '';
          const start = new Date(activeJob.started_at).getTime();
          const elapsed = Date.now() - start;
          const total = elapsed / (activeJob.progress / 100);
          const remaining = Math.floor((total - elapsed) / 1000);
          const h = Math.floor(remaining / 3600);
          const m = Math.floor((remaining % 3600) / 60);
          return h > 0 ? `~${h}h ${m}m` : m > 0 ? `~${m}m` : '<1m';
        };
        const elapsed = getElapsed();
        const eta = getETA();

        return (
          <div
            onClick={() => setIsTrainingOpen(true)}
            className="fixed bottom-6 right-6 z-50 bg-white/90 backdrop-blur-md shadow-lg border border-cyan-200 rounded-2xl px-4 py-3 flex items-center gap-3 cursor-pointer hover:scale-105 transition-transform animate-in slide-in-from-bottom-4 group"
          >
            <div className="relative">
              <div className="w-2.5 h-2.5 bg-cyan-500 rounded-full animate-pulse" />
              <div className="absolute inset-0 bg-cyan-400 rounded-full animate-ping opacity-50" />
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-600">Training Active</span>
              <span className="text-xs font-mono text-slate-700">
                {activeJob.status === 'preprocessing'
                  ? 'Preprocessing...'
                  : `Epoch ${activeJob.current_epoch}/${activeJob.total_epochs}${activeJob.current_loss ? ` • Loss: ${activeJob.current_loss.toFixed(4)}` : ''}`
                }
              </span>
              {/* Time metrics row */}
              {elapsed && (
                <span className="text-[10px] text-slate-400 mt-0.5">
                  ⏱ {elapsed} {eta && activeJob.progress > 0 && activeJob.progress < 100 && <span className="text-cyan-600 font-medium ml-1">ETA: {eta}</span>}
                </span>
              )}
            </div>
            <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden ml-2">
              <div className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-500" style={{ width: `${activeJob.progress}%` }} />
            </div>
          </div>
        );
      })()}

    </div>
  );
}

export default App;
