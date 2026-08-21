import React, { useState, useEffect, useRef } from 'react';
import { type Job, type TimedLine, API_BASE_URL } from '../../api';
import { useAudioEngine } from '../../context/AudioEngineContext';
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
  Check
} from 'lucide-react';
import { AudioVisualizer } from './AudioVisualizer';

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
    togglePlay,
    nextTrack,
    prevTrack,
    seek,
    setVolume,
    toggleMute,
    setPlaybackRate,
    setRepeatMode,
    toggleShuffle,
    stop
  } = useAudioEngine();

  const [timeMode, setTimeMode] = useState<'elapsed' | 'remaining'>('elapsed');
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

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds < 0) return '0:00';
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
                className="p-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 font-bold text-xs transition-colors flex items-center gap-1.5"
                title="Copy Lyrics to Clipboard"
              >
                {isCopied ? <Check size={14} className="text-teal-500" /> : <Copy size={14} />}
                <span className="hidden sm:inline">{isCopied ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                onClick={() => setIsLyricsOpen(false)}
                className="p-2 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                title="Close Lyrics"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          <div
            ref={lyricsScrollRef}
            className="flex-1 overflow-y-auto space-y-3.5 pr-2 custom-scrollbar text-center py-4"
          >
            {timedLyrics.length > 0 ? (
              timedLyrics.map((line, idx) => {
                const isActive = activeLineIndex === idx;
                const isPast = currentTime > line.end;
                return (
                  <div
                    key={idx}
                    data-line-idx={idx}
                    onClick={() => handleSeekToTime(line.start)}
                    className={`cursor-pointer px-4 py-2 rounded-2xl transition-all duration-300 ${
                      isActive
                        ? 'text-lg sm:text-xl font-extrabold text-teal-600 dark:text-teal-300 scale-105 bg-teal-500/10'
                        : isPast
                        ? 'text-sm sm:text-base text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                        : 'text-sm sm:text-base text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                    }`}
                  >
                    <p>{line.text}</p>
                    {isActive && (
                      <span className="text-[10px] font-mono text-teal-500/70 font-normal">
                        {formatTime(line.start)} - {formatTime(line.end)}
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

        <div className="flex items-center justify-between gap-2 sm:gap-4">
          <div
            onClick={() => onSelectTrack?.(currentSong)}
            className="flex items-center space-x-3 min-w-0 flex-1 max-w-xs sm:max-w-sm cursor-pointer group/player-track hover:opacity-90 transition-opacity"
            title="Open Track Studio"
          >
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
              <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-slate-100 truncate group-hover/player-track:text-teal-600 dark:group-hover/player-track:text-teal-400 transition-colors">
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
              onClick={toggleShuffle}
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
              onClick={prevTrack}
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
              onClick={() => togglePlay()}
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
              onClick={nextTrack}
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

            {/* Volume Control */}
            <div className="flex items-center space-x-1.5 group/vol hidden lg:flex">
              <button
                onClick={toggleMute}
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
                onChange={(e) => setVolume(parseFloat(e.target.value))}
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
              onClick={stop}
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
