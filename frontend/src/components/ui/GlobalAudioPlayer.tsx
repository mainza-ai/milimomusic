import React, { useState, useEffect, useRef } from 'react';
import { type Job, type TimedLine, API_BASE_URL } from '../../api';
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
  FileText
} from 'lucide-react';
import { AudioVisualizer } from './AudioVisualizer';

interface GlobalAudioPlayerProps {
  currentSong: Job | null;
  playlist: Job[];
  isPlaying: boolean;
  onTogglePlay: () => void;
  onNext: () => void;
  onPrev: () => void;
  onClose: () => void;
  onOpenWorkspace: (job: Job) => void;
}

export const GlobalAudioPlayer: React.FC<GlobalAudioPlayerProps> = ({
  currentSong,
  playlist,
  isPlaying,
  onTogglePlay,
  onNext,
  onPrev,
  onClose,
  onOpenWorkspace
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [timeMode, setTimeMode] = useState<'elapsed' | 'remaining'>('elapsed');

  // Playback Features
  const [repeatMode, setRepeatMode] = useState<'off' | 'all' | 'one'>('off');
  const [isShuffle, setIsShuffle] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [volume, setVolume] = useState(() => {
    const saved = localStorage.getItem('milimo_volume');
    return saved ? parseFloat(saved) : 0.8;
  });
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);

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

  const activeLineIndex = timedLyrics.findIndex(
    (l) => currentTime >= l.start && currentTime <= l.end
  );

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
    if (audioRef.current) {
      audioRef.current.currentTime = timeInSec;
      setCurrentTime(timeInSec);
    }
  };

  // Audio Sync
  useEffect(() => {
    if (!audioRef.current || !currentSong?.audio_path) return;

    const fullUrl = currentSong.audio_path.startsWith('http')
      ? currentSong.audio_path
      : `${API_BASE_URL}${currentSong.audio_path}`;

    if (audioRef.current.src !== fullUrl) {
      audioRef.current.src = fullUrl;
      audioRef.current.playbackRate = playbackSpeed;
      if (isPlaying) {
        audioRef.current.play().catch(() => {});
      }
    }
  }, [currentSong]);

  useEffect(() => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.play().catch(() => {});
    } else {
      audioRef.current.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackSpeed;
    }
  }, [playbackSpeed]);

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration || 60);
    }
  };

  const handleEnded = () => {
    if (repeatMode === 'one') {
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play();
      }
    } else if (repeatMode === 'all' || playlist.length > 1) {
      onNext();
    } else {
      onTogglePlay();
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const handleRewind10 = () => {
    if (audioRef.current) {
      const newTime = Math.max(0, audioRef.current.currentTime - 10);
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleAdvance10 = () => {
    if (audioRef.current) {
      const newTime = Math.min(duration, audioRef.current.currentTime + 10);
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const toggleRepeat = () => {
    if (repeatMode === 'off') setRepeatMode('all');
    else if (repeatMode === 'all') setRepeatMode('one');
    else setRepeatMode('off');
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const downloadAudio = () => {
    if (!currentSong?.audio_path) return;
    const a = document.createElement('a');
    a.href = currentSong.audio_path.startsWith('http')
      ? currentSong.audio_path
      : `${API_BASE_URL}${currentSong.audio_path}`;
    a.download = `${currentSong.title || 'milimo-track'}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!currentSong) return null;

  return (
    <div className="fixed bottom-0 inset-x-0 z-40 p-2 sm:p-4 pointer-events-none flex flex-col items-center">
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
      />

      {isLyricsOpen && (
        <div className="w-full max-w-4xl mb-3 p-5 sm:p-6 rounded-3xl bg-white/95 dark:bg-[#10121a]/95 border border-teal-500/30 shadow-apple-2xl backdrop-blur-3xl pointer-events-auto flex flex-col max-h-[60vh] sm:max-h-[420px] transition-all animate-slide-up z-50">
          <div className="flex items-center justify-between border-b border-black/[0.06] dark:border-white/[0.08] pb-3 mb-3">
            <div className="flex items-center space-x-2.5 min-w-0">
              <div className="p-2 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400">
                <Mic2 size={16} />
              </div>
              <div className="min-w-0">
                <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-slate-100 truncate">
                  {currentSong.title || currentSong.prompt || "Track Lyrics"}
                </h3>
                <p className="text-[11px] text-teal-600 dark:text-teal-400 font-mono">
                  {timedLyrics.length > 0 ? '✨ Real-time synchronized lyrics' : 'Full Song Lyrics'}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={handleCopyLyrics}
                className="px-2.5 py-1 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                title="Copy Lyrics to Clipboard"
              >
                {isCopied ? <Check size={13} className="text-teal-500" /> : <Copy size={13} />}
                <span className="text-[11px]">{isCopied ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                onClick={() => setIsLyricsOpen(false)}
                className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                title="Close Lyrics"
              >
                <X size={15} />
              </button>
            </div>
          </div>

          <div ref={lyricsScrollRef} className="flex-1 overflow-y-auto pr-2 space-y-4 py-2 select-text font-sans">
            {timedLyrics.length > 0 ? (
              timedLyrics.map((line, idx) => {
                const isCurrent = currentTime >= line.start && currentTime <= line.end;
                const isPast = currentTime > line.end;
                const isSectionHeader = line.text.startsWith('[') && line.text.endsWith(']');

                if (isSectionHeader) {
                  return (
                    <div key={idx} className="pt-3 pb-1">
                      <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2.5 py-1 rounded-full border border-teal-500/20">
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
                    className={`cursor-pointer transition-all duration-300 rounded-xl px-3 py-1.5 ${
                      isCurrent
                        ? 'bg-teal-500/15 dark:bg-teal-500/20 text-teal-900 dark:text-teal-200 font-extrabold text-base sm:text-lg scale-[1.01] shadow-sm'
                        : isPast
                        ? 'text-slate-500 dark:text-slate-400 font-medium text-sm hover:text-teal-600 dark:hover:text-teal-400'
                        : 'text-slate-400 dark:text-slate-500 font-normal text-sm hover:text-slate-700 dark:hover:text-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span>{line.text}</span>
                      <span className="text-[10px] font-mono text-slate-400 dark:text-slate-600 opacity-0 group-hover:opacity-100 flex-shrink-0">
                        {formatTime(line.start)}
                      </span>
                    </div>
                  </div>
                );
              })
            ) : rawLyrics ? (
              <pre className="text-xs sm:text-sm font-sans leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap">
                {rawLyrics}
              </pre>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-400 space-y-2">
                <FileText size={24} className="opacity-40" />
                <p className="text-xs">No lyrics recorded for this instrumental or custom master.</p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="w-full max-w-5xl bg-white/90 dark:bg-[#12141c]/90 border border-black/[0.08] dark:border-white/[0.08] shadow-apple-2xl rounded-3xl p-3 sm:p-4 backdrop-blur-2xl pointer-events-auto transition-all">
        <div className="flex items-center space-x-3 mb-2 px-1">
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

        <div className="flex items-center justify-between gap-2 sm:gap-4">
          <div className="flex items-center space-x-3 min-w-0 flex-1 max-w-xs sm:max-w-sm">
            <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-2xl bg-gradient-to-tr from-teal-500/20 via-cyan-500/20 to-sky-500/20 border border-black/[0.08] dark:border-white/10 p-1 flex-shrink-0 flex items-center justify-center shadow-sm relative overflow-hidden group">
              <img
                src="/milimo_logo.png"
                alt="Track Cover"
                className="w-full h-full object-cover rounded-xl"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
              <Disc
                size={18}
                className={`absolute text-teal-600 dark:text-teal-400 transition-transform ${
                  isPlaying ? 'animate-spin-slow' : 'opacity-0 group-hover:opacity-100'
                }`}
              />
            </div>

            <div className="min-w-0 pr-2">
              <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-slate-100 truncate">
                {currentSong.title || currentSong.prompt || 'Untitled Track'}
              </h3>
              <div className="flex items-center space-x-1.5 mt-0.5">
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-md bg-teal-500/10 text-teal-700 dark:text-teal-300 font-semibold border border-teal-500/20 truncate">
                  {currentSong.model_provider || 'MiniMax Music 3'}
                </span>
                <span className="text-[10px] text-slate-400 dark:text-slate-500 truncate hidden sm:inline">
                  {currentSong.tags || 'Studio Master'}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setIsShuffle(!isShuffle)}
              className={`p-1.5 rounded-xl transition-colors hidden xs:block ${
                isShuffle
                  ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                  : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
              title={`Shuffle: ${isShuffle ? 'On' : 'Off'}`}
            >
              <Shuffle size={15} />
            </button>

            <button
              onClick={onPrev}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              title="Previous Track"
            >
              <SkipBack size={16} />
            </button>

            <button
              onClick={handleRewind10}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              title="Rewind 10 seconds"
            >
              <RotateCcw size={15} />
            </button>

            <button
              onClick={onTogglePlay}
              className="w-10 h-10 sm:w-11 sm:h-11 rounded-2xl bg-gradient-to-tr from-teal-500 via-cyan-400 to-sky-500 hover:from-teal-400 hover:to-cyan-300 text-slate-950 font-bold flex items-center justify-center shadow-lg shadow-teal-500/30 hover:scale-105 active:scale-95 transition-transform"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
            </button>

            <button
              onClick={handleAdvance10}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              title="Advance 10 seconds"
            >
              <RotateCw size={15} />
            </button>

            <button
              onClick={onNext}
              className="p-1.5 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              title="Next Track"
            >
              <SkipForward size={16} />
            </button>

            <button
              onClick={toggleRepeat}
              className={`p-1.5 rounded-xl transition-colors hidden xs:block ${
                repeatMode !== 'off'
                  ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                  : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
              title={`Repeat: ${repeatMode}`}
            >
              {repeatMode === 'one' ? <Repeat1 size={15} /> : <Repeat size={15} />}
            </button>
          </div>

          <div className="flex items-center space-x-2 justify-end flex-1 max-w-xs">
            <div className="w-20 sm:w-28 h-8 hidden md:block rounded-xl overflow-hidden bg-black/[0.04] dark:bg-white/5 border border-black/[0.06] dark:border-white/5 p-1">
              <AudioVisualizer
                mediaElement={audioRef.current}
                isPlaying={isPlaying}
                mode="mirror"
                accentGradient="cyberpunk"
              />
            </div>

            <button
              onClick={() => setIsLyricsOpen(!isLyricsOpen)}
              className={`p-2 rounded-xl transition-all flex items-center gap-1 shadow-sm ${
                isLyricsOpen
                  ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-950 font-bold'
                  : 'bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-300'
              }`}
              title="Toggle Synchronized Lyrics"
              aria-label="Toggle Lyrics Sheet"
            >
              <Mic2 size={14} />
              <span className="text-[10px] font-bold hidden xl:inline">Lyrics</span>
            </button>

            <div className="relative">
              <button
                onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
                className="px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[10px] font-mono font-bold text-slate-700 dark:text-slate-300 transition-colors flex items-center gap-1"
                title="Playback Speed"
              >
                <Gauge size={12} className="text-teal-500" />
                <span>{playbackSpeed}x</span>
              </button>

              {isSpeedMenuOpen && (
                <div className="absolute bottom-full mb-2 right-0 bg-white dark:bg-[#181a24] border border-black/[0.08] dark:border-white/10 rounded-xl shadow-apple-lg p-1 space-y-1 z-50 animate-fade-in">
                  {[0.75, 1.0, 1.25, 1.5, 2.0].map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setPlaybackSpeed(s);
                        setIsSpeedMenuOpen(false);
                      }}
                      className={`w-full px-3 py-1 text-left text-xs font-mono rounded-lg transition-colors ${
                        playbackSpeed === s
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

            {/* Volume Control */}
            <div className="flex items-center space-x-1.5 group/vol hidden lg:flex">
              <button
                onClick={() => setIsMuted(!isMuted)}
                title={isMuted || volume === 0 ? "Unmute Audio" : "Mute Audio"}
                aria-label={isMuted || volume === 0 ? "Unmute Audio" : "Mute Audio"}
                className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
              >
                {isMuted || volume === 0 ? <VolumeX size={15} /> : <Volume2 size={15} />}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={isMuted ? 0 : volume}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setVolume(val);
                  setIsMuted(val === 0);
                  localStorage.setItem('milimo_volume', val.toString());
                }}
                title={`Playback Volume: ${Math.round((isMuted ? 0 : volume) * 100)}%`}
                aria-label="Playback Volume Slider"
                className="w-16 accent-teal-500 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <button
              onClick={() => onOpenWorkspace(currentSong)}
              className="p-2 rounded-xl bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 font-bold text-xs transition-all shadow-sm flex items-center gap-1"
              title="Open Track in DAW Workspace"
            >
              <Sliders size={14} />
              <span className="hidden sm:inline">DAW</span>
            </button>

            <button
              onClick={downloadAudio}
              className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              title="Download Master Audio (.wav)"
            >
              <Download size={14} />
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              title="Dismiss Player"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
