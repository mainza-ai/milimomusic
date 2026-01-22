import React, { useEffect, useRef } from 'react';
import { getAudioContext } from '../../utils/audioContext';

interface AudioVisualizerProps {
    mediaElement: HTMLMediaElement | null;
    isPlaying: boolean;
    className?: string;
    barColor?: string;
}

// Cache source nodes to prevent "can only be connected once" errors
const sourceCache = new WeakMap<HTMLMediaElement, MediaElementAudioSourceNode>();

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
    mediaElement,
    isPlaying,
    className,
    barColor = 'rgb(6, 182, 212)' // Cyan-500 default
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const contextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationRef = useRef<number | undefined>(undefined);

    useEffect(() => {
        if (!mediaElement) return;

        // Use Singleton Context
        const ctx = getAudioContext();
        contextRef.current = ctx;

        // Create or Retrieve Source Node
        let source: MediaElementAudioSourceNode;
        if (sourceCache.has(mediaElement)) {
            source = sourceCache.get(mediaElement)!;
        } else {
            source = ctx.createMediaElementSource(mediaElement);
            sourceCache.set(mediaElement, source);
        }

        // Create Analyser (one per visualizer instance is fine, but cleaner to share? No, instance is visualizer)
        // Analyser needs to be created on the SAME context
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;

        // Connect: Source -> Analyser -> Destination
        // We must be careful not to create multiple connections from the same source?
        // Source node fan-out is allowed. 
        try {
            source.connect(analyser);
            analyser.connect(ctx.destination);
        } catch (e) {
            console.warn("Visualizer connection error (ignoring already connected warning):", e);
        }

        return () => {
            // Cleanup: Disconnect this analyser from the source to prevent graph buildup
            // We CANNOT close the context (shared).
            // We CAN disconnect the analyser.
            try {
                if (source) source.disconnect(analyser);
                if (analyser) analyser.disconnect();
            } catch (e) {
                // Ignore disconnect errors
            }
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

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Config
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;

        // Gradient
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, barColor); // Base
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0.8)'); // Tip

        for (let i = 0; i < bufferLength; i++) {
            barHeight = (dataArray[i] / 255) * canvas.height;

            // Smooth curve damping
            // barHeight = barHeight * 0.8; 

            ctx.fillStyle = gradient;

            // Rounded bars look more modern
            // ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

            // Draw rounded rect equivalent
            if (barHeight > 0) {
                ctx.beginPath();
                ctx.roundRect(x, canvas.height - barHeight, barWidth, barHeight, [4, 4, 0, 0]);
                ctx.fill();
            }

            x += barWidth + 2; // Spacing
        }

        if (isPlaying) {
            animationRef.current = requestAnimationFrame(draw);
        } else {
            // Draw one last frame or clear? Keep last frame looks paused.
            // Maybe fade out?
            // animationRef.current = requestAnimationFrame(draw); // Keep updating even if paused? (Usually silent)
        }
    };

    useEffect(() => {
        if (isPlaying) {
            // Resume shared context if suspended
            if (contextRef.current?.state === 'suspended') {
                contextRef.current.resume();
            }
            draw();
        } else {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            // Optional: Clear canvas on stop? Or keep waveform frozen?
            // Let's keep it frozen or running 0s
            // draw(); // One frame of silence if paused?
        }
    }, [isPlaying]);

    return (
        <canvas
            ref={canvasRef}
            width={300}
            height={60}
            className={`${className} `}
        />
    );
};
