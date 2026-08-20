import React, { useEffect, useRef } from 'react';
import { getAudioContext } from '../../utils/audioContext';

interface AudioVisualizerProps {
    mediaElement: HTMLMediaElement | null;
    isPlaying: boolean;
    className?: string;
    mode?: 'bars' | 'wave' | 'mirror';
    accentGradient?: 'neon' | 'sunset' | 'cyberpunk' | 'aurora';
}

const sourceCache = new WeakMap<HTMLMediaElement, MediaElementAudioSourceNode>();

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
    mediaElement,
    isPlaying,
    className,
    mode = 'mirror',
    accentGradient = 'neon'
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const contextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationRef = useRef<number | undefined>(undefined);
    const peakBarsRef = useRef<number[]>([]);

    useEffect(() => {
        if (!mediaElement) return;

        const ctx = getAudioContext();
        contextRef.current = ctx;

        let source: MediaElementAudioSourceNode;
        if (sourceCache.has(mediaElement)) {
            source = sourceCache.get(mediaElement)!;
        } else {
            try {
                source = ctx.createMediaElementSource(mediaElement);
                sourceCache.set(mediaElement, source);
            } catch (e) {
                return;
            }
        }

        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.82;
        analyserRef.current = analyser;

        try {
            source.connect(analyser);
            analyser.connect(ctx.destination);
        } catch {}

        return () => {
            try {
                if (source) source.disconnect(analyser);
                if (analyser) analyser.disconnect();
            } catch {}
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, [mediaElement]);

    const draw = () => {
        if (!canvasRef.current || !analyserRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d', { alpha: true });
        if (!ctx) return;

        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Resize peaks array if needed
        if (peakBarsRef.current.length !== bufferLength) {
            peakBarsRef.current = new Array(bufferLength).fill(0);
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const width = canvas.width;
        const height = canvas.height;
        const barCount = 48;
        const step = Math.floor(bufferLength / barCount);
        const barWidth = (width / barCount) * 0.7;
        const barSpacing = (width / barCount) * 0.3;

        // Create Vibrant Multi-Color Gradient
        const gradient = ctx.createLinearGradient(0, height, width, 0);
        if (accentGradient === 'sunset') {
            gradient.addColorStop(0, '#f59e0b');
            gradient.addColorStop(0.5, '#fb923c');
            gradient.addColorStop(1, '#06b6d4');
        } else if (accentGradient === 'cyberpunk') {
            gradient.addColorStop(0, '#00f2fe');
            gradient.addColorStop(0.4, '#0ea5e9');
            gradient.addColorStop(0.7, '#10b981');
            gradient.addColorStop(1, '#f59e0b');
        } else if (accentGradient === 'aurora') {
            gradient.addColorStop(0, '#00c6ff');
            gradient.addColorStop(0.5, '#0072ff');
            gradient.addColorStop(1, '#38ef7d');
        } else {
            // Neon Default: Electric Cyan -> Vivid Teal -> Amber Gold
            gradient.addColorStop(0, '#00f2fe');
            gradient.addColorStop(0.4, '#14b8a6');
            gradient.addColorStop(0.8, '#06b6d4');
            gradient.addColorStop(1, '#f59e0b');
        }

        if (mode === 'mirror') {
            // Dual Mirror Spectrum (Center Y)
            const centerY = height / 2;
            ctx.shadowBlur = 10;
            ctx.shadowColor = 'rgba(0, 242, 254, 0.4)';

            for (let i = 0; i < barCount; i++) {
                const value = dataArray[i * step] || 0;
                const percent = value / 255;
                const h = (percent * (centerY - 4)) * 0.95;

                // Update peak gravity
                if (h > peakBarsRef.current[i]) {
                    peakBarsRef.current[i] = h;
                } else {
                    peakBarsRef.current[i] = Math.max(0, peakBarsRef.current[i] - 0.8);
                }

                const x = i * (barWidth + barSpacing) + barSpacing / 2;

                ctx.fillStyle = gradient;

                // Top Half
                if (h > 1) {
                    ctx.beginPath();
                    ctx.roundRect(x, centerY - h, barWidth, h, [3, 3, 0, 0]);
                    ctx.fill();

                    // Bottom Mirror Half with subtle fade
                    ctx.globalAlpha = 0.55;
                    ctx.beginPath();
                    ctx.roundRect(x, centerY, barWidth, h * 0.7, [0, 0, 3, 3]);
                    ctx.fill();
                    ctx.globalAlpha = 1.0;
                }

                // Glowing Peak Dots
                const peak = peakBarsRef.current[i];
                if (peak > 2) {
                    ctx.fillStyle = '#ffffff';
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = '#00f2fe';
                    ctx.beginPath();
                    ctx.arc(x + barWidth / 2, centerY - peak - 2, 1.2, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        } else if (mode === 'wave') {
            // Smooth Waveform Ribbon
            ctx.beginPath();
            ctx.moveTo(0, height / 2);

            for (let i = 0; i < bufferLength; i += 2) {
                const x = (i / bufferLength) * width;
                const v = (dataArray[i] / 255) - 0.5;
                const y = (height / 2) + v * (height * 0.8);

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.strokeStyle = gradient;
            ctx.lineWidth = 2.5;
            ctx.shadowBlur = 12;
            ctx.shadowColor = 'rgba(20, 184, 166, 0.6)';
            ctx.stroke();
        } else {
            // Standard Upward Spectrum Bars
            ctx.shadowBlur = 8;
            ctx.shadowColor = 'rgba(0, 242, 254, 0.4)';

            for (let i = 0; i < barCount; i++) {
                const value = dataArray[i * step] || 0;
                const h = (value / 255) * (height - 6);
                const x = i * (barWidth + barSpacing) + barSpacing / 2;

                ctx.fillStyle = gradient;
                if (h > 1) {
                    ctx.beginPath();
                    ctx.roundRect(x, height - h, barWidth, h, [4, 4, 0, 0]);
                    ctx.fill();
                }
            }
        }

        if (isPlaying) {
            animationRef.current = requestAnimationFrame(draw);
        }
    };

    useEffect(() => {
        if (isPlaying) {
            if (contextRef.current?.state === 'suspended') {
                contextRef.current.resume();
            }
            draw();
        } else {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            // Gentle clear
            if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
        }
    }, [isPlaying]);

    return (
        <canvas
            ref={canvasRef}
            width={400}
            height={60}
            className={`${className || ''} w-full h-full block`}
        />
    );
};
