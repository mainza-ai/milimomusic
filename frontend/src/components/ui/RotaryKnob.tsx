import React, { useState, useRef } from 'react';

interface RotaryKnobProps {
    label?: string;
    value: number;
    min?: number;
    max?: number;
    step?: number;
    defaultValue?: number;
    bipolar?: boolean;
    unit?: string;
    size?: number;
    onChange: (value: number) => void;
    ariaLabel?: string;
    className?: string;
}

export const RotaryKnob: React.FC<RotaryKnobProps> = ({
    label,
    value,
    min = -50,
    max = 50,
    step = 1,
    defaultValue = 0,
    bipolar = true,
    unit = '',
    size = 46,
    onChange,
    ariaLabel,
    className = '',
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const startYRef = useRef<number>(0);
    const startValRef = useRef<number>(value);
    const knobRef = useRef<HTMLDivElement>(null);

    // Angle mapping: -135deg (min) to +135deg (max), total range = 270deg
    const minAngle = -135;
    const maxAngle = 135;
    const angleRange = maxAngle - minAngle;

    const normalizedFraction = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
    const currentAngle = minAngle + normalizedFraction * angleRange;

    // Center angle for bipolar knobs
    const centerAngle = minAngle + ((defaultValue - min) / (max - min || 1)) * angleRange;

    const clamp = (val: number) => Math.min(max, Math.max(min, val));

    const handlePointerDown = (e: React.PointerEvent) => {
        e.preventDefault();
        setIsDragging(true);
        startYRef.current = e.clientY;
        startValRef.current = value;
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
    };

    const handlePointerMove = (e: React.PointerEvent) => {
        if (!isDragging) return;

        const deltaY = startYRef.current - e.clientY;
        const speed = e.shiftKey ? 0.2 : 1.0; // Fine-tuning with Shift
        const sensitivity = (max - min) / 120; // 120px full travel
        let newVal = startValRef.current + deltaY * sensitivity * speed;

        // Snap to center detent if bipolar and close
        if (bipolar && Math.abs(newVal - defaultValue) < (max - min) * 0.02) {
            newVal = defaultValue;
        }

        // Quantize by step
        if (step > 0) {
            newVal = Math.round(newVal / step) * step;
        }

        onChange(clamp(newVal));
    };

    const handlePointerUp = (e: React.PointerEvent) => {
        if (isDragging) {
            setIsDragging(false);
            try {
                (e.target as HTMLElement).releasePointerCapture(e.pointerId);
            } catch {
                // Ignore if capture already lost
            }
        }
    };

    const handleDoubleClick = () => {
        onChange(defaultValue);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        let delta = 0;
        if (e.key === 'ArrowUp' || e.key === 'ArrowRight') delta = step;
        else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') delta = -step;
        else if (e.key === 'PageUp') delta = step * 10;
        else if (e.key === 'PageDown') delta = -step * 10;
        else if (e.key === 'Home') {
            onChange(min);
            e.preventDefault();
            return;
        } else if (e.key === 'End') {
            onChange(max);
            e.preventDefault();
            return;
        }

        if (delta !== 0) {
            e.preventDefault();
            onChange(clamp(value + delta));
        }
    };

    // SVG Arc calculations
    const radius = size * 0.38;
    const center = size / 2;
    const strokeWidth = 3;

    const polarToCartesian = (centerX: number, centerY: number, r: number, angleInDegrees: number) => {
        const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;
        return {
            x: centerX + r * Math.cos(angleInRadians),
            y: centerY + r * Math.sin(angleInRadians),
        };
    };

    const describeArc = (x: number, y: number, r: number, startA: number, endA: number) => {
        const start = polarToCartesian(x, y, r, endA);
        const end = polarToCartesian(x, y, r, startA);
        const largeArcFlag = endA - startA <= 180 ? '0' : '1';
        return ['M', start.x, start.y, 'A', r, r, 0, largeArcFlag, 0, end.x, end.y].join(' ');
    };

    // Arc path for the active value
    let activeArcPath = '';
    if (bipolar) {
        if (currentAngle >= centerAngle) {
            activeArcPath = describeArc(center, center, radius, centerAngle, Math.max(centerAngle + 0.1, currentAngle));
        } else {
            activeArcPath = describeArc(center, center, radius, currentAngle, centerAngle);
        }
    } else {
        activeArcPath = describeArc(center, center, radius, minAngle, Math.max(minAngle + 0.1, currentAngle));
    }

    const backgroundArcPath = describeArc(center, center, radius, minAngle, maxAngle);

    // Indicator line
    const indicatorLength = radius - 4;
    const indicatorPos = polarToCartesian(center, center, indicatorLength, currentAngle);

    // Display formatted value
    const formatDisplayVal = () => {
        if (bipolar && min === -50 && max === 50) {
            if (value === 0) return 'C';
            return value < 0 ? `L${Math.abs(value)}` : `R${value}`;
        }
        return `${Math.round(value * 10) / 10}${unit}`;
    };

    return (
        <div className={`flex flex-col items-center select-none ${className}`}>
            {label && (
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 dark:text-slate-500 font-bold mb-1">
                    {label}
                </span>
            )}
            <div
                ref={knobRef}
                role="slider"
                tabIndex={0}
                aria-label={ariaLabel || label || 'Rotary control'}
                aria-valuenow={value}
                aria-valuemin={min}
                aria-valuemax={max}
                aria-valuetext={formatDisplayVal()}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onDoubleClick={handleDoubleClick}
                onKeyDown={handleKeyDown}
                className={`relative cursor-ns-resize focus:outline-none focus:ring-1 focus:ring-teal-500/50 rounded-full transition-transform active:scale-95 ${
                    isDragging ? 'cursor-grabbing' : 'cursor-grab'
                }`}
                style={{ width: size, height: size }}
                title={`${ariaLabel || label || 'Control'}: ${formatDisplayVal()} (Double click to reset, Shift+drag for fine tuning)`}
            >
                <svg width={size} height={size} className="overflow-visible pointer-events-none">
                    <defs>
                        <radialGradient id="knobBodyGrad" cx="40%" cy="35%" r="60%">
                            <stop offset="0%" stopColor="rgba(255,255,255,0.08)" />
                            <stop offset="100%" stopColor="rgba(0,0,0,0.4)" />
                        </radialGradient>
                        <linearGradient id="activeArcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#2dd4bf" />
                            <stop offset="100%" stopColor="#06b6d4" />
                        </linearGradient>
                    </defs>

                    {/* Track Background */}
                    <path
                        d={backgroundArcPath}
                        fill="none"
                        stroke="currentColor"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        className="text-slate-200 dark:text-slate-800"
                    />

                    {/* Active Illuminated Arc */}
                    <path
                        d={activeArcPath}
                        fill="none"
                        stroke="url(#activeArcGrad)"
                        strokeWidth={strokeWidth + 0.5}
                        strokeLinecap="round"
                        className="filter drop-shadow-[0_0_4px_rgba(45,212,191,0.5)]"
                    />

                    {/* Outer Knob Cap */}
                    <circle
                        cx={center}
                        cy={center}
                        r={radius - 4}
                        className="fill-slate-100 dark:fill-[#1b1e2a] stroke-slate-300 dark:stroke-slate-700/80"
                        strokeWidth={1}
                    />

                    {/* Inner 3D Dial Fill */}
                    <circle
                        cx={center}
                        cy={center}
                        r={radius - 5}
                        fill="url(#knobBodyGrad)"
                    />

                    {/* Center Zero Detent Dot (for bipolar knobs) */}
                    {bipolar && (
                        <circle
                            cx={center}
                            cy={center - radius}
                            r={1.2}
                            className="fill-slate-400 dark:fill-slate-600"
                        />
                    )}

                    {/* Needle Indicator */}
                    <line
                        x1={center}
                        y1={center}
                        x2={indicatorPos.x}
                        y2={indicatorPos.y}
                        stroke="#2dd4bf"
                        strokeWidth={2}
                        strokeLinecap="round"
                        className="filter drop-shadow-[0_0_2px_rgba(45,212,191,0.8)]"
                    />
                    <circle cx={center} cy={center} r={2} className="fill-teal-400" />
                </svg>
            </div>

            {/* Readout */}
            <span className="text-[10px] font-mono font-bold text-slate-700 dark:text-slate-300 mt-1 min-w-[28px] text-center">
                {formatDisplayVal()}
            </span>
        </div>
    );
};
