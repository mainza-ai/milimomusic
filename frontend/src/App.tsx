import { useState, useEffect } from 'react';
import { api } from './api';
import type { Job } from './api';
import { ComposerSidebar } from './components/ComposerSidebar';
import type { CompositionData } from './components/ComposerSidebar';
import { HistoryFeed } from './components/HistoryFeed';

function App() {
  const [lyricsModels, setLyricsModels] = useState<string[]>([]);
  const [history, setHistory] = useState<Job[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [parentJob, setParentJob] = useState<Job | undefined>(undefined);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGeneratingLyrics, setIsGeneratingLyrics] = useState(false);

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
        data.topic, // Fixed: CompositionData uses 'topic', not 'prompt'
        data.durationMs,
        data.lyrics,
        data.tags,
        data.cfgScale,
        parentJob?.id, // Pass extension context
        parentJob?.seed // Pass seed for consistency
      );
      console.log("Generation started, Job ID:", job_id);
      setCurrentJobId(job_id);
      setParentJob(undefined); // Clear extension mode after starting
    } catch (e: any) {
      alert("Generation failed: " + e.message);
      setIsGenerating(false);
    }
  };

  const handleGenerateLyrics = async (topic: string, model: string, currentLyrics?: string) => {
    setIsGeneratingLyrics(true);
    try {
      return await api.generateLyrics(topic, model, currentLyrics);
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
        />
      </aside>

    </div>
  );
}

export default App;
