import React, { useState, useEffect, useRef } from 'react';
import { type Job, type TimedLine, API_BASE_URL } from '../../api';
import { useAudioEngine } from '../../context/AudioEngineContext';
import { DEFAULT_COVER_ART } from '../../constants/assets';
import {
  Play,
  Pause,
  RotateCcw,
  RotateCw,
  SkipBack,
  SkipForward,
  Repeat,
  Repeat1,
  Shuffle,
  Volume2,
  VolumeX,
  Sliders,
  Download,
  Disc,
  X,
  Gauge,
  Mic2,
  Copy,
  Check,
  ListMusic,
  Trash2,
  FileText
} from 'lucide-react';

interface GlobalAudioPlayerProps {
  onOpenWorkspace: (job: Job) => void;
  onSelectTrack?: (job: Job) => void;
}

export const GlobalAudioPlayer: React.FC<GlobalAudioPlayerProps> = ({
  onOpenWorkspace,
  onSelectTrack
}) => {
  const {
    currentTrack: currentSong,
    isPlaying,
    currentTime,
    duration,
    volume,
    isMuted,
    playbackRate,
    repeatMode,
    isShuffle,
    playlist,
    playTrack,
    togglePlay,
    nextTrack,
    prevTrackOrRestart,
    seek,
    setVolume,
    toggleMute,
    setPlaybackRate,
    setRepeatMode,
    toggleShuffle,
    removeFromQueue,
    clearQueue,
    stop
  } = useAudioEngine();

  const [timeMode, setTimeMode] = useState<'elapsed' | 'remaining'>('elapsed');
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [isQueueOpen, setIsQueueOpen] = useState(false);

  // Synchronized Lyrics State
  const [isLyricsOpen, setIsLyricsOpen] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const lyricsScrollRef = useRef<HTMLDivElement | null>(null);

  const timedLyrics: TimedLine[] = currentSong?.timed_lyrics_json
    ? typeof currentSong.timed_lyrics_json === 'string'
      ? JSON.parse(currentSong.timed_lyrics_json)
      : currentSong.timed_lyrics_json
    : [];

  const rawLyrics = currentSong?.lyrics || '';

  // Continuous Active Line finder with proximity smoothing
  const activeLineIndex = (() => {
    if (timedLyrics.length === 0) return -1;
    for (let i = 0; i < timedLyrics.length; i++) {
      const line = timedLyrics[i];
      const nextLine = timedLyrics[i + 1];
      const lineStart = line.start;
      const lineEnd = nextLine ? nextLine.start : (line.end || line.start + 6);
      if (currentTime >= lineStart && currentTime < lineEnd) {
        return i;
      }
    }
    if (currentTime >= timedLyrics[timedLyrics.length - 1].start) {
      return timedLyrics.length - 1;
    }
    return -1;
  })();

  // Auto-scroll active lyric line into center view
  useEffect(() => {
    if (isLyricsOpen && activeLineIndex !== -1 && lyricsScrollRef.current) {
      const activeEl = lyricsScrollRef.current.querySelector(`[data-line-idx="${activeLineIndex}"]`);
      if (activeEl) {
        activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [activeLineIndex, isLyricsOpen]);

  const handleCopyLyrics = () => {
    if (!rawLyrics) return;
    navigator.clipboard.writeText(rawLyrics);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleSeekToTime = (timeInSec: number) => {
    seek(timeInSec);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    seek(parseFloat(e.target.value));
  };

  const handleRewind10 = () => {
    seek(Math.max(0, currentTime - 10));
  };

  const handleAdvance10 = () => {
    seek(Math.min(duration, currentTime + 10));
  };

  const toggleRepeat = () => {
    if (repeatMode === 'off') setRepeatMode('all');
    else if (repeatMode === 'all') setRepeatMode('one');
    else setRepeatMode('off');
  };

  const formatTime = (seconds: number): string => {
    if (!seconds || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const downloadAudio = () => {
    if (!currentSong?.audio_path) return;
    const url = currentSong.audio_path.startsWith('http')
      ? currentSong.audio_path
      : `${API_BASE_URL}${currentSong.audio_path}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentSong.title || 'milimo_track'}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!currentSong) return null;

  const artworkUrl = currentSong.cover_image_path
    ? currentSong.cover_image_path.startsWith('http')
      ? currentSong.cover_image_path
      : `${API_BASE_URL}${currentSong.cover_image_path}`
    : DEFAULT_COVER_ART;

  const upcomingQueue = playlist.filter((p) => p.id !== currentSong.id);

  return (
    <div className="fixed bottom-6 left-0 right-0 z-50 flex flex-col items-center pointer-events-none px-3 sm:px-6 animate-slide-up">
      {/* Up Next Queue Drawer */}
      {isQueueOpen && (
        <div className="w-full max-w-5xl bg-white/95 dark:bg-[#12141c]/95 border border-black/[0.08] dark:border-white/10 shadow-apple-2xl backdrop-blur-2xl rounded-3xl p-5 mb-3 pointer-events-auto flex flex-col max-h-[380px] animate-fade-in">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.06] dark:border-white/10">
            <div className="flex items-center space-x-2">
              <ListMusic size={18} className="text-teal-600 dark:text-teal-400" />
              <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Playing Queue ({playlist.length} {playlist.length === 1 ? 'track' : 'tracks'})
              </h3>
            </div>
            <div className="flex items-center space-x-2">
              {upcomingQueue.length > 0 && (
                <button
                  onClick={clearQueue}
                  className="px-2.5 py-1 rounded-xl text-xs font-mono text-rose-600 dark:text-rose-400 hover:bg-rose-500/10 flex items-center gap-1 transition-colors"
                >
                  <Trash2 size={12} />
                  <span>Clear Queue</span>
                </button>
              )}
              <button
                onClick={() => setIsQueueOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto pt-3 space-y-1.5 custom-scrollbar pr-1">
            {/* Now Playing Row */}
            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-teal-600 dark:text-teal-400 mb-1 px-2">
              Now Playing
            </div>
            <div className="p-2.5 rounded-2xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-between">
              <div className="flex items-center space-x-3 min-w-0 flex-1">
                <div className="w-9 h-9 rounded-xl overflow-hidden bg-black/10 flex-shrink-0 flex items-center justify-center relative">
                  <img
                    src={artworkUrl}
                    alt="Now playing cover"
                    className="w-full h-full object-cover"
                  />
                  {isPlaying && (
                    <div className="absolute inset-0 bg-teal-900/40 backdrop-blur-[1px] flex items-center justify-center">
                      <div className="w-3 h-3 flex items-end justify-center space-x-0.5">
                        <div className="w-0.5 h-full bg-teal-300 animate-pulse" />
                        <div className="w-0.5 h-2/3 bg-teal-300 animate-pulse" />
                        <div className="w-0.5 h-4/5 bg-teal-300 animate-pulse" />
                      </div>
                    </div>
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                    {currentSong.title || currentSong.prompt || 'Untitled Track'}
                  </div>
                  <div className="text-[10px] text-slate-400 truncate">
                    {currentSong.model_provider || 'MiniMax Music 3'} • {formatTime(duration)}
                  </div>
                </div>
              </div>
              <span className="text-[10px] font-mono font-bold text-teal-600 dark:text-teal-400 bg-teal-500/20 px-2 py-0.5 rounded-full">
                Active
              </span>
            </div>

            {/* Up Next List */}
            {upcomingQueue.length > 0 ? (
              <>
                <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-400 mt-3 mb-1 px-2">
                  Up Next ({upcomingQueue.length})
                </div>
                {upcomingQueue.map((track) => {
                  const trackArt = track.cover_image_path
                    ? track.cover_image_path.startsWith('http')
                      ? track.cover_image_path
                      : `${API_BASE_URL}${track.cover_image_path}`
                    : DEFAULT_COVER_ART;
                  return (
                    <div
                      key={track.id}
                      className="p-2 rounded-xl hover:bg-black/[0.04] dark:hover:bg-white/5 border border-transparent hover:border-black/[0.06] dark:hover:border-white/5 flex items-center justify-between transition-colors group"
                    >
                      <div
                        onClick={() => playTrack(track)}
                        className="flex items-center space-x-3 min-w-0 flex-1 cursor-pointer"
                      >
                        <div className="w-8 h-8 rounded-lg overflow-hidden bg-black/10 flex-shrink-0">
                          <img src={trackArt} alt="Track cover" className="w-full h-full object-cover" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-xs font-semibold text-slate-800 dark:text-slate-200 truncate group-hover:text-teal-600 dark:group-hover:text-teal-400">
                            {track.title || track.prompt || 'Untitled Track'}
                          </div>
                          <div className="text-[10px] text-slate-400 truncate">
                            {track.tags || track.model_provider || 'Studio Track'}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-1 pl-2">
                        <button
                          onClick={() => playTrack(track)}
                          className="p-1 rounded-lg hover:bg-teal-500/10 text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                          title="Play Now"
                        >
                          <Play size={13} />
                        </button>
                        <button
                          onClick={() => removeFromQueue(track.id)}
                          className="p-1 rounded-lg hover:bg-rose-500/10 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 transition-colors"
                          title="Remove from Queue"
                        >
                          <X size={13} />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </>
            ) : (
              <div className="text-center py-4 text-slate-400 text-xs">
                No more tracks in queue. Add songs from your library!
              </div>
            )}
          </div>
        </div>
      )}

      {/* Synchronized Lyrics Drawer */}
      {isLyricsOpen && (
        <div className="w-full max-w-5xl bg-white/95 dark:bg-[#12141c]/95 border border-black/[0.08] dark:border-white/10 shadow-apple-2xl backdrop-blur-2xl rounded-3xl p-5 mb-3 pointer-events-auto flex flex-col max-h-[380px] animate-fade-in">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.06] dark:border-white/10">
            <div className="flex items-center space-x-2">
              <Mic2 size={18} className="text-teal-600 dark:text-teal-400" />
              <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                Synchronized Lyrics Sheet
              </h3>
              {timedLyrics.length > 0 && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20">
                  LRC Timed
                </span>
              )}
            </div>

            <div className="flex items-center space-x-2">
              {currentSong && (
                <a
                  href={`${API_BASE_URL}/tracks/${currentSong.id}/lrc`}
                  download={`${currentSong.title || 'lyrics'}.lrc`}
                  className="px-2.5 py-1 rounded-xl text-xs font-mono text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-1 transition-colors"
                  title="Download Synchronized LRC File"
                >
                  <FileText size={12} className="text-teal-500" />
                  <span>.LRC</span>
                </a>
              )}
              {rawLyrics && (
                <button
                  onClick={handleCopyLyrics}
                  className="px-2.5 py-1 rounded-xl text-xs font-mono text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-1 transition-colors"
                  title="Copy lyrics to clipboard"
                >
                  {isCopied ? <Check size={12} className="text-teal-500" /> : <Copy size={12} />}
                  <span>{isCopied ? 'Copied' : 'Copy'}</span>
                </button>
              )}
              <button
                onClick={() => setIsLyricsOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                title="Close Lyrics"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          <div
            ref={lyricsScrollRef}
            className="flex-1 overflow-y-auto py-4 space-y-3 custom-scrollbar text-center px-4"
          >
            {timedLyrics.length > 0 ? (
              timedLyrics.map((line, idx) => {
                const isActive = idx === activeLineIndex;
                const isSection = (line as any).is_section || (line.text.startsWith('[') && line.text.endsWith(']'));

                if (isSection) {
                  return (
                    <div key={idx} className="py-2">
                      <span className="text-[11px] font-mono font-bold uppercase tracking-widest text-teal-600 dark:text-teal-400 bg-teal-500/10 px-3 py-1 rounded-full border border-teal-500/20">
                        {line.text}
                      </span>
                    </div>
                  );
                }

                return (
                  <div
                    key={idx}
                    data-line-idx={idx}
                    onClick={() => handleSeekToTime(line.start)}
                    className={`cursor-pointer transition-all duration-300 px-4 py-2 rounded-2xl ${
                      isActive
                        ? 'text-teal-600 dark:text-teal-300 font-extrabold text-base sm:text-lg scale-105 bg-teal-500/10 shadow-sm'
                        : 'text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 text-sm'
                    }`}
                  >
                    {isActive && line.words && line.words.length > 0 ? (
                      <span className="inline-flex flex-wrap justify-center gap-1.5">
                        {line.words.map((w, wIdx) => {
                          const isWordSung = currentTime >= w.start;
                          return (
                            <span
                              key={wIdx}
                              className={`transition-colors duration-150 ${
                                isWordSung
                                  ? 'text-teal-600 dark:text-teal-300 font-black drop-shadow-sm'
                                  : 'text-slate-400 dark:text-slate-500 opacity-60'
                              }`}
                            >
                              {w.word}
                            </span>
                          );
                        })}
                      </span>
                    ) : (
                      <span>{line.text}</span>
                    )}
                    {isActive && (
                      <span className="text-[10px] font-mono block opacity-60 mt-0.5">
                        {formatTime(line.start)}
                      </span>
                    )}
                  </div>
                );
              })
            ) : rawLyrics ? (
              <div className="text-sm sm:text-base leading-relaxed text-slate-700 dark:text-slate-300 whitespace-pre-wrap font-sans text-left px-4">
                {rawLyrics}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 py-8">
                <Mic2 size={32} className="opacity-30 mb-2" />
                <p className="text-xs">No lyrics found for this track.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Floating Apple Player Bar */}
      <div className="w-full max-w-5xl bg-white/90 dark:bg-[#12141c]/90 border border-black/[0.08] dark:border-white/10 shadow-apple-2xl backdrop-blur-2xl rounded-3xl p-3 sm:p-4 pointer-events-auto flex flex-col space-y-2.5 transition-all">
        {/* Scrubber Progress Bar */}
        <div className="flex items-center space-x-2 sm:space-x-3 w-full px-1">
          <button
            onClick={() => setTimeMode(timeMode === 'elapsed' ? 'remaining' : 'elapsed')}
            className="text-[10px] font-mono font-medium text-slate-500 dark:text-slate-400 w-10 text-right select-none hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
            title="Toggle time elapsed / remaining"
          >
            {timeMode === 'elapsed'
              ? formatTime(currentTime)
              : `-${formatTime(Math.max(0, duration - currentTime))}`}
          </button>

          <div className="flex-1 relative flex items-center group/scrub">
            <input
              type="range"
              min="0"
              max={duration || 100}
              step="0.1"
              value={currentTime}
              onChange={handleSeek}
              title={`Playhead Position: ${formatTime(currentTime)} / ${formatTime(duration)}`}
              aria-label="Audio Playhead Scrubber"
              className="w-full accent-teal-500 h-1.5 bg-black/[0.06] dark:bg-white/10 rounded-lg appearance-none cursor-pointer group-hover/scrub:h-2 transition-all"
            />
          </div>

          <span className="text-[10px] font-mono font-medium text-slate-500 dark:text-slate-400 w-10 select-none">
            {formatTime(duration)}
          </span>
        </div>

        {/* 3-Zone Isolated Flex Layout: Left Track Info, Center Transport, Right Tools */}
        <div className="flex items-center justify-between gap-3 w-full">
          {/* Zone 1: Left Track Identity */}
          <div
            onClick={() => onSelectTrack?.(currentSong)}
            className="flex items-center space-x-2.5 min-w-0 max-w-[200px] sm:max-w-[240px] md:max-w-[280px] shrink cursor-pointer group/player-track hover:opacity-90 transition-opacity"
            title="Open Track Studio"
          >
            <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-2xl bg-gradient-to-tr from-teal-500/20 via-cyan-500/20 to-sky-500/20 border border-black/[0.08] dark:border-white/10 p-0.5 shrink-0 flex items-center justify-center shadow-sm relative overflow-hidden group">
              <img
                src={artworkUrl}
                alt="Track Cover"
                className="w-full h-full object-cover rounded-xl"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = DEFAULT_COVER_ART;
                }}
              />
              <Disc
                size={16}
                className={`absolute text-teal-300 drop-shadow-md transition-transform ${
                  isPlaying ? 'animate-spin-slow' : 'opacity-0 group-hover:opacity-100'
                }`}
              />
            </div>

            <div className="min-w-0 flex-1 overflow-hidden">
              <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-slate-100 truncate group-hover/player-track:text-teal-600 dark:group-hover/player-track:text-teal-400 transition-colors">
                {currentSong.title || currentSong.prompt || 'Untitled Track'}
              </h3>
              <div className="flex items-center space-x-1.5 mt-0.5 truncate">
                <span className="text-[9px] sm:text-[10px] font-mono px-1.5 py-0.2 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20 truncate">
                  {currentSong.model_provider || 'MiniMax Music 3'}
                </span>
              </div>
            </div>
          </div>

          {/* Zone 2: Center Master Transport Controls (Never wraps, never shrinks, always centered) */}
          <div className="flex items-center justify-center space-x-1 sm:space-x-1.5 shrink-0 mx-auto">
            <button
              onClick={toggleShuffle}
              className={`p-1.5 rounded-xl transition-colors shrink-0 hidden sm:block ${
                isShuffle
                  ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                  : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
              title={`Shuffle: ${isShuffle ? 'On' : 'Off'}`}
            >
              <Shuffle size={14} />
            </button>

            {/* Return to Start / Previous Track Button */}
            <button
              onClick={prevTrackOrRestart}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
              title="Return to Start / Previous Track (|<<) (Bracket Left / Home)"
              aria-label="Return to Start or Previous Track"
            >
              <SkipBack size={15} />
            </button>

            {/* Rewind 10s */}
            <button
              onClick={handleRewind10}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
              title="Rewind 10 seconds (J / Left Arrow)"
              aria-label="Rewind 10 seconds"
            >
              <RotateCcw size={15} />
            </button>

            {/* Master Play/Pause Button */}
            <button
              onClick={() => togglePlay()}
              className="w-10 h-10 sm:w-11 sm:h-11 rounded-2xl bg-gradient-to-tr from-teal-500 via-cyan-400 to-sky-500 hover:from-teal-400 hover:to-cyan-300 text-slate-950 font-bold flex items-center justify-center shadow-lg shadow-teal-500/30 hover:scale-105 active:scale-95 transition-transform shrink-0"
              title={isPlaying ? 'Pause (K / Space)' : 'Play (K / Space)'}
              aria-label={isPlaying ? 'Pause Audio' : 'Play Audio'}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
            </button>

            {/* Advance 10s */}
            <button
              onClick={handleAdvance10}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
              title="Advance 10 seconds (L / Right Arrow)"
              aria-label="Advance 10 seconds"
            >
              <RotateCw size={15} />
            </button>

            {/* Next Track */}
            <button
              onClick={nextTrack}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
              title="Next Track (>>|) (Bracket Right)"
              aria-label="Next Track"
            >
              <SkipForward size={15} />
            </button>

            {/* Repeat / Loop Toggle */}
            <button
              onClick={toggleRepeat}
              className={`p-1.5 rounded-xl transition-colors shrink-0 hidden sm:block ${
                repeatMode !== 'off'
                  ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                  : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
              title={`Repeat Mode: ${repeatMode}`}
            >
              {repeatMode === 'one' ? <Repeat1 size={14} /> : <Repeat size={14} />}
            </button>
          </div>

          {/* Zone 3: Right Studio Tools (Flex items shrink and hide responsively, zero overlap) */}
          <div className="flex items-center justify-end space-x-1 sm:space-x-1.5 shrink min-w-0">
            {/* Up Next Queue Toggle Button */}
            <button
              onClick={() => {
                setIsQueueOpen(!isQueueOpen);
                if (isLyricsOpen) setIsLyricsOpen(false);
              }}
              className={`p-1.5 sm:p-2 rounded-xl transition-all flex items-center gap-1 shadow-sm shrink-0 ${
                isQueueOpen
                  ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold'
                  : 'bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300'
              }`}
              title="Toggle Playing Queue"
              aria-label="Toggle Playing Queue"
            >
              <ListMusic size={14} />
              <span className="text-[10px] font-bold hidden 2xl:inline">Queue</span>
            </button>

            {/* Lyrics Sheet Toggle Button */}
            <button
              onClick={() => {
                setIsLyricsOpen(!isLyricsOpen);
                if (isQueueOpen) setIsQueueOpen(false);
              }}
              className={`p-1.5 sm:p-2 rounded-xl transition-all flex items-center gap-1 shadow-sm shrink-0 ${
                isLyricsOpen
                  ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold'
                  : 'bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300'
              }`}
              title="Toggle Synchronized Lyrics"
              aria-label="Toggle Lyrics Sheet"
            >
              <Mic2 size={14} />
              <span className="text-[10px] font-bold hidden 2xl:inline">Lyrics</span>
            </button>

            {/* Playback Speed Menu */}
            <div className="relative shrink-0">
              <button
                onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
                className="px-1.5 sm:px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[10px] font-mono font-bold text-slate-700 dark:text-slate-300 transition-colors flex items-center gap-0.5"
                title="Playback Speed"
              >
                <Gauge size={11} className="text-teal-500" />
                <span>{playbackRate}x</span>
              </button>

              {isSpeedMenuOpen && (
                <div className="absolute bottom-full mb-2 right-0 bg-white dark:bg-[#181a24] border border-black/[0.08] dark:border-white/10 rounded-xl shadow-apple-lg p-1 space-y-1 z-50 animate-fade-in">
                  {[0.75, 1.0, 1.25, 1.5, 2.0].map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setPlaybackRate(s);
                        setIsSpeedMenuOpen(false);
                      }}
                      className={`w-full px-3 py-1 text-left text-xs font-mono rounded-lg transition-colors ${
                        playbackRate === s
                          ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 font-bold'
                          : 'text-slate-600 dark:text-slate-400 hover:bg-black/5 dark:hover:bg-white/5'
                      }`}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Volume Slider Control */}
            <div className="flex items-center space-x-1 group/vol hidden md:flex shrink-0">
              <button
                onClick={toggleMute}
                title={isMuted || volume === 0 ? "Unmute Audio (M)" : "Mute Audio (M)"}
                aria-label={isMuted || volume === 0 ? "Unmute Audio" : "Mute Audio"}
                className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
              >
                {isMuted || volume === 0 ? <VolumeX size={14} /> : <Volume2 size={14} />}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={isMuted ? 0 : volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                title={`Playback Volume: ${Math.round((isMuted ? 0 : volume) * 100)}% (Arrow Up/Down)`}
                aria-label="Playback Volume Slider"
                className="w-12 sm:w-16 accent-teal-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* DAW Button */}
            <button
              onClick={() => onOpenWorkspace(currentSong)}
              className="p-1.5 sm:p-2 rounded-xl bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 font-bold text-xs transition-all shadow-sm flex items-center gap-1 shrink-0"
              title="Open Track in DAW Workspace"
            >
              <Sliders size={13} />
              <span className="hidden sm:inline">DAW</span>
            </button>

            {/* Download Master Audio */}
            <button
              onClick={downloadAudio}
              className="p-1.5 sm:p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors shrink-0"
              title="Download Master Audio (.wav)"
            >
              <Download size={13} />
            </button>

            {/* Dismiss Player */}
            <button
              onClick={stop}
              className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors shrink-0"
              title="Dismiss Player"
            >
              <X size={13} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
