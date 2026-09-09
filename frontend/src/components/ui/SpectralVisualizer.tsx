import React, { useEffect, useRef } from 'react';

interface SpectralVisualizerProps {
    analyser?: AnalyserNode | null;
    isPlaying?: boolean;
    height?: number;
    barCount?: number;
    className?: string;
    variant?: 'bars' | 'curve' | 'minimal';
}

export const SpectralVisualizer: React.FC<SpectralVisualizerProps> = ({
    analyser,
    isPlaying = false,
    height = 36,
    barCount = 32,
    className = '',
    variant = 'bars',
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animFrameRef = useRef<number | null>(null);
    const peaksRef = useRef<number[]>([]);
    const dimsRef = useRef<{ width: number; height: number; dpr: number }>({ width: 0, height: 0, dpr: 1 });

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Track dimensions via ResizeObserver — eliminates clientWidth/clientHeight layout reads in rAF
        const updateDimensions = () => {
            const width = canvas.clientWidth || 64;
            const h = canvas.clientHeight || height;
            const dpr = Math.min(window.devicePixelRatio || 1, 2); // Cap at 2x DPR to save GPU rasterization
            dimsRef.current = { width, height: h, dpr };
            if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(h * dpr)) {
                canvas.width = Math.round(width * dpr);
                canvas.height = Math.round(h * dpr);
            }
        };

        updateDimensions();
        const ro = new ResizeObserver(() => {
            updateDimensions();
            if (!isPlaying) {
                drawFrame(false, 0);
            }
        });
        ro.observe(canvas);

        // Initialize peaks array if needed
        if (peaksRef.current.length !== barCount) {
            peaksRef.current = new Array(barCount).fill(0);
        }

        // Buffer for FFT data
        const fftSize = analyser ? analyser.frequencyBinCount : 64;
        const dataArray = new Uint8Array(fftSize);
        let idlePhase = 0;

        const drawFrame = (active: boolean, phase: number) => {
            const { width, height: h, dpr } = dimsRef.current;
            if (width <= 0 || h <= 0) return;

            ctx.save();
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, width, h);

            let hasLiveSignal = false;
            if (analyser && active) {
                analyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for (let i = 0; i < 32 && i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                if (sum > 10) hasLiveSignal = true;
            }

            if (variant === 'curve') {
                ctx.beginPath();
                ctx.moveTo(0, h);

                const step = width / Math.max(1, barCount - 1);
                for (let i = 0; i < barCount; i++) {
                    let val = 0;
                    if (hasLiveSignal) {
                        const index = Math.min(
                            Math.floor(Math.pow(i / barCount, 1.8) * dataArray.length),
                            dataArray.length - 1
                        );
                        val = dataArray[index] / 255;
                    } else if (active) {
                        val = (Math.sin(phase + i * 0.3) * 0.15 + 0.15) * 0.5;
                    }

                    const x = i * step;
                    const y = h - val * (h - 4);
                    if (i === 0) ctx.lineTo(x, y);
                    else {
                        const prevX = (i - 1) * step;
                        const midX = (prevX + x) / 2;
                        ctx.quadraticCurveTo(prevX, y, midX, y);
                    }
                }

                ctx.lineTo(width, h);
                ctx.closePath();

                const grad = ctx.createLinearGradient(0, 0, width, h);
                grad.addColorStop(0, 'rgba(45, 212, 191, 0.4)');
                grad.addColorStop(0.5, 'rgba(6, 182, 212, 0.3)');
                grad.addColorStop(1, 'rgba(168, 85, 247, 0.15)');
                ctx.fillStyle = grad;
                ctx.fill();

                ctx.lineWidth = 1.5;
                ctx.strokeStyle = '#2dd4bf';
                ctx.stroke();
            } else {
                const barSpacing = 2;
                const totalSpacing = (barCount - 1) * barSpacing;
                const barWidth = Math.max(2, (width - totalSpacing) / barCount);

                for (let i = 0; i < barCount; i++) {
                    let val = 0;
                    if (hasLiveSignal) {
                        const binIndex = Math.min(
                            Math.floor(Math.pow(i / barCount, 1.6) * dataArray.length),
                            dataArray.length - 1
                        );
                        val = dataArray[binIndex] / 255;
                    } else if (active) {
                        const harmonic = Math.sin(phase * 2 + i * 0.4) * 0.5 + 0.5;
                        val = 0.08 + harmonic * 0.12;
                    } else {
                        val = 0.03;
                    }

                    if (val > peaksRef.current[i]) {
                        peaksRef.current[i] = val;
                    } else {
                        peaksRef.current[i] = Math.max(0, peaksRef.current[i] - 0.02);
                    }

                    const barHeight = Math.max(2, val * (h - 4));
                    const x = i * (barWidth + barSpacing);
                    const y = h - barHeight;

                    const barGrad = ctx.createLinearGradient(0, h, 0, 0);
                    barGrad.addColorStop(0, '#0d9488');
                    barGrad.addColorStop(0.6, '#2dd4bf');
                    barGrad.addColorStop(0.9, '#38bdf8');
                    barGrad.addColorStop(1, '#f59e0b');

                    ctx.fillStyle = hasLiveSignal || active ? barGrad : 'rgba(100, 116, 139, 0.2)';
                    
                    const radius = Math.min(1.5, barWidth / 2);
                    ctx.beginPath();
                    ctx.roundRect(x, y, barWidth, barHeight, [radius, radius, 0, 0]);
                    ctx.fill();

                    if ((hasLiveSignal || active) && peaksRef.current[i] > 0.05) {
                        const peakY = Math.max(1, h - peaksRef.current[i] * (h - 4));
                        ctx.fillStyle = '#f87171';
                        ctx.fillRect(x, peakY, barWidth, 1.5);
                    }
                }
            }

            ctx.restore();
        };

        // If not playing, render single resting frame and stop — NO requestAnimationFrame loop when idle!
        if (!isPlaying) {
            drawFrame(false, 0);
            return () => {
                ro.disconnect();
            };
        }

        // Active animation loop (only runs while playing, capped at 60fps to prevent 120Hz+ GPU thrashing)
        let isLoopRunning = true;
        let lastFrameTime = 0;
        const TARGET_FPS = 60;
        const FRAME_INTERVAL = 1000 / TARGET_FPS;

        const render = (timestamp: number) => {
            if (!isLoopRunning) return;
            if (document.hidden) {
                // Bail out when tab is hidden to save 100% GPU
                animFrameRef.current = requestAnimationFrame(render);
                return;
            }
            const delta = timestamp - lastFrameTime;
            if (delta >= FRAME_INTERVAL) {
                lastFrameTime = timestamp - (delta % FRAME_INTERVAL);
                drawFrame(true, idlePhase);
                idlePhase += 0.04;
            }
            animFrameRef.current = requestAnimationFrame(render);
        };

        animFrameRef.current = requestAnimationFrame(render);

        const handleVisibilityChange = () => {
            if (!document.hidden && isLoopRunning) {
                drawFrame(true, idlePhase);
            }
        };
        document.addEventListener('visibilitychange', handleVisibilityChange);

        return () => {
            isLoopRunning = false;
            ro.disconnect();
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            if (animFrameRef.current !== null) {
                cancelAnimationFrame(animFrameRef.current);
                animFrameRef.current = null;
            }
        };
    }, [analyser, isPlaying, barCount, variant, height]);

    return (
        <canvas
            ref={canvasRef}
            className={`w-full block rounded-lg overflow-hidden ${className}`}
            style={{ height }}
            role="img"
            aria-label="Audio Spectral Visualizer"
        />
    );
};
