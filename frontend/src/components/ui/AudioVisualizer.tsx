import React, { useEffect, useRef } from 'react';

interface AudioVisualizerProps {
    mediaElement: HTMLMediaElement | null;
    isPlaying: boolean;
    className?: string;
    barColor?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
    mediaElement,
    isPlaying,
    className,
    barColor = 'rgb(6, 182, 212)' // Cyan-500 default
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const contextRef = useRef<AudioContext | null>(null);
    const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationRef = useRef<number | undefined>(undefined);

    useEffect(() => {
        if (!mediaElement) return;

        // Initialize Audio Context (singleton-ish behavior needed usually, but scoped here is okay if handled carefully)
        if (!contextRef.current) {
            const AudioContextClass = (window.AudioContext || (window as any).webkitAudioContext);
            contextRef.current = new AudioContextClass();
            analyserRef.current = contextRef.current.createAnalyser();
            analyserRef.current.fftSize = 256; // Tradeoff: resolution vs smooth bars. 256 = 128 bins.

            // Connect nodes
            try {
                // IMPORTANT: MediaElementSource can only be created once per element. 
                // In React strict mode or re-renders, this might fail if not checked.
                // However, since we re-use the same element from WaveSurfer, it should be persistent?
                // Actually WaveSurfer might re-create it. We need to be careful.
                // For now, we wrap in try-catch or check if we already have a source map (not easy).
                // A better approach is usually to let WaveSurfer handle the audio graph, but WaveSurfer 7 is simpler.

                // Hack: We only create source if we haven't linked this exact element before?
                // Or just try.
                if (contextRef.current) {
                    sourceRef.current = contextRef.current.createMediaElementSource(mediaElement);
                    if (analyserRef.current) {
                        sourceRef.current.connect(analyserRef.current);
                        analyserRef.current.connect(contextRef.current.destination);
                    }
                }
            } catch (e) {
                // If source already connected, we might need to skip source creation?
                // But we can't easily tap into an existing graph without the node reference.
                // Assuming AudioPlayer unmounts cleans this up?
                console.warn("Visualizer connection issue (likely strict mode re-mount):", e);
            }
        }

        return () => {
            // Cleanup context? 
            // If we close context, the media stops playing. AudioContext controls the hardware.
            // So we generally DON'T close the context if we want audio to persist, 
            // BUT this component is part of the player.
            // Let's just cancel animation.
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
            // Resume context if needed (browsers auto-suspend)
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
            className={`${className}`}
        />
    );
};
