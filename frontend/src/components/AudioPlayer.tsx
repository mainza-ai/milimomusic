import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { API_BASE_URL } from '../api';
import {
  Play,
  Pause,
  Download,
  SkipBack,
  SkipForward,
  RotateCcw,
  RotateCw,
  Repeat,
  Repeat1,
  Volume2,
  VolumeX,
  Wand2,
  Gauge
} from 'lucide-react';
import { GlassCard } from './ui/GlassCard';
import { InpaintModal } from './InpaintModal';

interface AudioPlayerProps {
  audioUrl: string;
  jobId: string;
  className?: string;
  onNext?: () => void;
  onPrev?: () => void;
  title?: string;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  audioUrl,
  jobId,
  className,
  onNext,
  onPrev,
  title
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState('0:00');
  const [currentTime, setCurrentTime] = useState('0:00');

  // Playback Features
  const [repeatMode, setRepeatMode] = useState<'off' | 'one'>('off');
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isSpeedOpen, setIsSpeedOpen] = useState(false);

  // Volume Persistence
  const [volume, setVolume] = useState(() => {
    const saved = localStorage.getItem('milimo_volume');
    return saved ? parseFloat(saved) : 0.7;
  });
  const [isMuted, setIsMuted] = useState(false);
  const [isInpaintOpen, setIsInpaintOpen] = useState(false);

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds < 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // Create Canvas Waveform Gradient
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    let progressGrad: string | CanvasGradient = '#14b8a6';
    let waveGrad: string | CanvasGradient = 'rgba(20, 184, 166, 0.3)';

    if (ctx) {
      const g = ctx.createLinearGradient(0, 0, 400, 0);
      g.addColorStop(0, '#00f2fe');
      g.addColorStop(0.5, '#14b8a6');
      g.addColorStop(1, '#0284c7');
      progressGrad = g;

      const wg = ctx.createLinearGradient(0, 0, 400, 0);
      wg.addColorStop(0, 'rgba(0, 242, 254, 0.3)');
      wg.addColorStop(1, 'rgba(14, 165, 233, 0.25)');
      waveGrad = wg;
    }

    wavesurfer.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: waveGrad,
      progressColor: progressGrad,
      cursorColor: '#00f2fe',
      cursorWidth: 2,
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      height: 56,
      normalize: true,
      backend: 'MediaElement'
    });

    wavesurfer.current.on('ready', () => {
      setDuration(formatTime(wavesurfer.current?.getDuration() || 0));
      wavesurfer.current?.setVolume(isMuted ? 0 : volume);
    });

    wavesurfer.current.on('audioprocess', () => {
      setCurrentTime(formatTime(wavesurfer.current?.getCurrentTime() || 0));
    });

    wavesurfer.current.on('finish', () => {
      if (repeatMode === 'one') {
        wavesurfer.current?.play();
      } else {
        setIsPlaying(false);
        if (onNext) onNext();
      }
    });

    return () => {
      const ws = wavesurfer.current;
      wavesurfer.current = null;
      if (ws) {
        try {
          ws.unAll();
          ws.stop();
          ws.destroy();
        } catch {}
      }
    };
  }, []);

  // Load Audio URL
  useEffect(() => {
    if (wavesurfer.current && audioUrl) {
      const fullUrl = audioUrl.startsWith('http') ? audioUrl : `${API_BASE_URL}${audioUrl}`;
      wavesurfer.current.load(fullUrl);
    }
  }, [audioUrl]);

  // Handle Play/Pause
  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(wavesurfer.current.isPlaying());
    }
  };

  const handleRewind10 = () => {
    if (wavesurfer.current) {
      const cur = wavesurfer.current.getCurrentTime();
      wavesurfer.current.setTime(Math.max(0, cur - 10));
    }
  };

  const handleAdvance10 = () => {
    if (wavesurfer.current) {
      const cur = wavesurfer.current.getCurrentTime();
      const dur = wavesurfer.current.getDuration() || 300;
      wavesurfer.current.setTime(Math.min(dur, cur + 10));
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    setIsSpeedOpen(false);
    if (wavesurfer.current) {
      const media = wavesurfer.current.getMediaElement();
      if (media) media.playbackRate = speed;
    }
  };

  const downloadAudio = () => {
    const fullUrl = audioUrl.startsWith('http') ? audioUrl : `${API_BASE_URL}${audioUrl}`;
    const a = document.createElement('a');
    a.href = fullUrl;
    a.download = `milimo-track-${jobId}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <GlassCard className={`p-4 md:p-5 ${className || ''} space-y-3`}>
      {title && (
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate max-w-xs">
            {title}
          </span>
          <span className="text-[10px] font-mono text-teal-600 dark:text-teal-400 font-bold">
            48kHz Stereo Master
          </span>
        </div>
      )}

      {/* Waveform Area */}
      <div className="relative w-full rounded-2xl overflow-hidden bg-black/[0.03] dark:bg-white/[0.03] p-3 border border-black/[0.06] dark:border-white/10 shadow-inner">
        <div ref={containerRef} className="w-full cursor-pointer" />
      </div>

      {/* Player Controls Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
        <span className="text-xs font-mono font-semibold text-slate-500 dark:text-slate-400 w-12 flex-shrink-0">
          {currentTime}
        </span>

        {/* Center Transport Buttons */}
        <div className="flex items-center gap-1 sm:gap-1.5 flex-shrink-0">
          {/* Return to Start / Zero */}
          <button
            onClick={() => {
              if (wavesurfer.current) {
                const cur = wavesurfer.current.getCurrentTime();
                if (cur > 3 || !onPrev) {
                  wavesurfer.current.setTime(0);
                } else {
                  onPrev();
                }
              } else if (onPrev) {
                onPrev();
              }
            }}
            className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-slate-600 dark:text-slate-400 flex-shrink-0"
            title="Return to Start / Previous (|<<)"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          {/* Rewind 10s */}
          <button
            onClick={handleRewind10}
            className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-slate-600 dark:text-slate-400 flex-shrink-0"
            title="Rewind 10 seconds"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Glowing Play/Pause */}
          <button
            onClick={togglePlay}
            className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-teal-500 via-cyan-400 to-sky-500 hover:from-teal-400 hover:to-cyan-300 text-slate-950 shadow-md shadow-teal-500/25 hover:scale-105 transition-transform active:scale-95 flex items-center justify-center font-bold flex-shrink-0"
            title={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4 ml-0.5" />
            )}
          </button>

          {/* Advance 10s */}
          <button
            onClick={handleAdvance10}
            className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-slate-600 dark:text-slate-400 flex-shrink-0"
            title="Advance 10 seconds"
          >
            <RotateCw className="w-4 h-4" />
          </button>

          {onNext && (
            <button
              onClick={onNext}
              className="p-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-slate-600 dark:text-slate-400 flex-shrink-0"
              title="Next Track"
            >
              <SkipForward className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Right Tools (Speed, Loop, Volume, Inpaint, Download) */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Speed Selector */}
          <div className="relative">
            <button
              onClick={() => setIsSpeedOpen(!isSpeedOpen)}
              className="px-2 py-1 rounded-lg bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-[10px] font-mono font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1 transition-colors"
              title="Playback Speed"
            >
              <Gauge size={12} className="text-teal-500" />
              <span>{playbackSpeed}x</span>
            </button>

            {isSpeedOpen && (
              <div className="absolute bottom-full mb-2 right-0 bg-white dark:bg-[#181a24] border border-black/[0.08] dark:border-white/10 rounded-xl shadow-apple-lg p-1 space-y-1 z-50 animate-fade-in">
                {[0.75, 1.0, 1.25, 1.5, 2.0].map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSpeedChange(s)}
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

          {/* Repeat Toggle */}
          <button
            onClick={() => setRepeatMode(repeatMode === 'off' ? 'one' : 'off')}
            className={`p-1.5 rounded-xl transition-colors ${
              repeatMode === 'one'
                ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
                : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
            }`}
            title={`Loop Track: ${repeatMode}`}
          >
            {repeatMode === 'one' ? <Repeat1 size={15} /> : <Repeat size={15} />}
          </button>

          {/* Volume Control */}
          <div className="flex items-center gap-1 group/volume">
            <button
              onClick={() => {
                const newMuted = !isMuted;
                setIsMuted(newMuted);
                if (wavesurfer.current) {
                  wavesurfer.current.setVolume(newMuted ? 0 : volume);
                }
              }}
              className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
            >
              {isMuted || volume === 0 ? (
                <VolumeX className="w-4 h-4" />
              ) : (
                <Volume2 className="w-4 h-4" />
              )}
            </button>
            <div className="w-0 overflow-hidden group-hover/volume:w-16 transition-all duration-300">
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
                  if (wavesurfer.current) wavesurfer.current.setVolume(val);
                  localStorage.setItem('milimo_volume', val.toString());
                }}
                className="w-16 h-1 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
              />
            </div>
          </div>

          {/* Inpaint Button */}
          <button
            onClick={() => setIsInpaintOpen(true)}
            className="p-1.5 rounded-lg bg-teal-500/10 hover:bg-teal-500 text-teal-700 dark:text-teal-300 hover:text-slate-950 transition-colors font-bold text-xs"
            title="Repair Audio Segment"
          >
            <Wand2 className="w-4 h-4" />
          </button>

          {/* Download Audio */}
          <button
            onClick={downloadAudio}
            className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
            title="Download Audio"
          >
            <Download className="w-4 h-4" />
          </button>

          <div className="w-12 text-right text-xs font-mono text-slate-500 dark:text-slate-400">
            {duration}
          </div>
        </div>
      </div>

      {/* Inpaint Modal */}
      <InpaintModal
        isOpen={isInpaintOpen}
        onClose={() => setIsInpaintOpen(false)}
        jobId={jobId}
        duration={wavesurfer.current?.getDuration() || 0}
        title={title}
      />
    </GlassCard>
  );
};
