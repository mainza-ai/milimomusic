/**
 * Provenance badge (plan 1.2).
 *
 * Every artifact now carries a `quality` verdict computed from per-stage
 * provenance. `neural` renders nothing — a badge on every healthy track would be
 * noise. Anything degraded or synthetic is always visible, because the whole
 * point is that a user can never ship a placeholder believing it is model output.
 */
import React from 'react';
import { ShieldCheck, AlertTriangle, FlaskConical } from 'lucide-react';

export type JobQuality = 'neural' | 'degraded' | 'placeholder' | string;

export function qualityOf(job: { quality?: string; used_fallback_synth?: boolean }): JobQuality {
    if (job.quality) return job.quality;
    // Tracks produced before provenance existed cannot be asserted as neural —
    // several of them shipped fabricated notation. Report "unknown", which the
    // badge treats as "say nothing" rather than "verified good".
    if (job.used_fallback_synth) return 'placeholder';
    return 'unknown';
}

interface ProvenanceBadgeProps {
    job: { quality?: string; used_fallback_synth?: boolean; fallback_reason?: string | null };
    /** `compact` renders an icon-only pill for dense list rows. */
    variant?: 'full' | 'compact';
    className?: string;
}

export const ProvenanceBadge: React.FC<ProvenanceBadgeProps> = ({ job, variant = 'full', className = '' }) => {
    const quality = qualityOf(job);
    // Nothing to say for healthy or unprovenanced tracks.
    if (quality === 'neural' || quality === 'unknown') return null;

    const isPlaceholder = quality === 'placeholder';
    const Icon = isPlaceholder ? FlaskConical : AlertTriangle;
    const label = isPlaceholder ? 'Placeholder audio' : 'Degraded';
    const title = isPlaceholder
        ? `This track was produced by a fallback generator, not the model.${job.fallback_reason ? ` Reason: ${job.fallback_reason}` : ''}`
        : `One or more steps fell back to DSP/heuristic output.${job.fallback_reason ? ` Reason: ${job.fallback_reason}` : ''}`;
    const tone = isPlaceholder
        ? 'bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30'
        : 'bg-sky-500/10 text-sky-700 dark:text-sky-300 border-sky-500/30';

    return (
        <span
            role="status"
            title={title}
            aria-label={title}
            className={`inline-flex items-center gap-1 rounded-lg border font-bold ${tone} ${
                variant === 'compact' ? 'px-1.5 py-0.5 text-[9px]' : 'px-2 py-0.5 text-[10px]'
            } ${className}`}
        >
            <Icon size={variant === 'compact' ? 9 : 11} />
            {variant === 'full' && <span>{label}</span>}
        </span>
    );
};

/** Positive badge used only where the user explicitly asks "was this real?". */
export const NeuralBadge: React.FC<{ className?: string }> = ({ className = '' }) => (
    <span
        className={`inline-flex items-center gap-1 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-700 dark:text-emerald-300 ${className}`}
    >
        <ShieldCheck size={11} />
        <span>Neural</span>
    </span>
);
