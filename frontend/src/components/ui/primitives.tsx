import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { X, Loader2 } from 'lucide-react';

// ── Shared class helper ─────────────────────────────────────────────────────
export function cn(...inputs: ClassValue[]): string {
    return twMerge(clsx(inputs));
}

// ── Button ──────────────────────────────────────────────────────────────────
// Single canonical button primitive. Replaces the ≥4 divergent hand-rolled
// CTA recipes (one of which — the main Generate action — had no hover state).
type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
    loading?: boolean;
}

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
    // Brand gradient CTA. slate-950 text on teal/cyan ≈ 10:1 contrast.
    primary:
        'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 shadow-md shadow-teal-500/20',
    secondary:
        'bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 text-slate-700 dark:text-slate-200 border border-black/[0.06] dark:border-white/10',
    ghost:
        'bg-transparent hover:bg-black/[0.04] dark:hover:bg-white/5 text-slate-600 dark:text-slate-300',
    danger:
        'bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-rose-500/20',
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
    sm: 'px-3 py-1.5 text-xs rounded-xl gap-1.5',
    md: 'px-4 py-2 text-sm rounded-xl gap-2',
};

export const Button: React.FC<ButtonProps> = ({
    variant = 'secondary',
    size = 'sm',
    loading = false,
    disabled,
    className,
    children,
    ...rest
}) => (
    <button
        disabled={disabled || loading}
        className={cn(
            'inline-flex items-center justify-center font-bold transition-all active:scale-[0.98]',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50',
            'disabled:opacity-50 disabled:pointer-events-none cursor-pointer',
            BUTTON_VARIANTS[variant],
            BUTTON_SIZES[size],
            className
        )}
        {...rest}
    >
        {loading && <Loader2 size={14} className="animate-spin" />}
        {children}
    </button>
);

// ── Spinner ─────────────────────────────────────────────────────────────────
// The ONE busy indicator. Retires the ≥6 ad-hoc spinner idioms.
interface SpinnerProps {
    size?: number;
    className?: string;
    label?: string;
}

export const Spinner: React.FC<SpinnerProps> = ({ size = 16, className, label }) => (
    <span role="status" aria-label={label || 'Loading'} className={cn('inline-flex items-center', className)}>
        <Loader2 size={size} className="animate-spin text-teal-500" />
    </span>
);

// ── Toggle switch ───────────────────────────────────────────────────────────
// Accessible switch: real <button>, keyboard operable, aria-checked. Replaces
// the <div onClick> switches that keyboard users could not operate at all.
interface ToggleProps {
    checked: boolean;
    onChange: (next: boolean) => void;
    label: string;
    size?: 'sm' | 'md';
}

export const Toggle: React.FC<ToggleProps> = ({ checked, onChange, label, size = 'md' }) => {
    const dims = size === 'sm'
        ? { track: 'w-8 h-[18px]', knob: 'w-3.5 h-3.5', travel: 'translate-x-[14px]' }
        : { track: 'w-10 h-[22px]', knob: 'w-[18px] h-[18px]', travel: 'translate-x-[18px]' };
    return (
        <button
            type="button"
            role="switch"
            aria-checked={checked}
            aria-label={label}
            title={label}
            onClick={() => onChange(!checked)}
            className={cn(
                'relative inline-flex items-center rounded-full transition-colors duration-200 flex-shrink-0 cursor-pointer',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50',
                dims.track,
                checked ? 'bg-teal-500' : 'bg-slate-300 dark:bg-slate-600'
            )}
        >
            <span
                className={cn(
                    'inline-block transform rounded-full bg-white shadow-sm transition-transform duration-200',
                    dims.knob,
                    checked ? dims.travel : 'translate-x-0.5'
                )}
            />
        </button>
    );
};

// ── Badge ───────────────────────────────────────────────────────────────────
type BadgeTone = 'teal' | 'amber' | 'rose' | 'neutral';

const BADGE_TONES: Record<BadgeTone, string> = {
    teal: 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border-teal-500/20',
    amber: 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/20',
    rose: 'bg-rose-500/15 text-rose-600 dark:text-rose-300 border-rose-500/30',
    neutral: 'bg-black/[0.04] dark:bg-white/5 text-slate-600 dark:text-slate-400 border-black/[0.06] dark:border-white/10',
};

interface BadgeProps {
    tone?: BadgeTone;
    className?: string;
    children: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({ tone = 'neutral', className, children }) => (
    <span
        className={cn(
            'inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border whitespace-nowrap',
            BADGE_TONES[tone],
            className
        )}
    >
        {children}
    </span>
);

// ── Modal ───────────────────────────────────────────────────────────────────
// One modal shell for the app: portal, Escape + backdrop close, focus trap,
// focus restore, aria-modal. Retires 15 ad-hoc modal implementations that
// could not be closed with Escape and trapped no focus.
interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    title?: string;
    /** Tailwind max-width class, e.g. "max-w-lg". */
    widthClass?: string;
    children: React.ReactNode;
}

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, widthClass = 'max-w-lg', children }) => {
    const panelRef = useRef<HTMLDivElement | null>(null);
    const lastActiveRef = useRef<Element | null>(null);

    useEffect(() => {
        if (!isOpen) return;
        lastActiveRef.current = document.activeElement;

        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                e.stopPropagation();
                onClose();
                return;
            }
            if (e.key !== 'Tab' || !panelRef.current) return;
            // Minimal focus trap: cycle tab within the panel.
            const focusables = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE))
                .filter(el => !el.hasAttribute('disabled'));
            if (focusables.length === 0) return;
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            if (e.shiftKey && document.activeElement === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        };

        document.addEventListener('keydown', onKeyDown, true);
        // Move initial focus into the dialog.
        requestAnimationFrame(() => {
            const first = panelRef.current?.querySelector<HTMLElement>(FOCUSABLE);
            first?.focus();
        });
        return () => {
            document.removeEventListener('keydown', onKeyDown, true);
            // Restore focus to wherever the user opened the modal from.
            if (lastActiveRef.current instanceof HTMLElement) {
                lastActiveRef.current.focus();
            }
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return createPortal(
        <div
            className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in"
            onMouseDown={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div
                ref={panelRef}
                role="dialog"
                aria-modal="true"
                aria-label={title}
                className={cn(
                    'w-full bg-white dark:bg-surface-raised border border-black/10 dark:border-white/10 rounded-3xl shadow-apple-2xl animate-scale-up flex flex-col',
                    widthClass
                )}
            >
                {title && (
                    <div className="flex items-center justify-between border-b border-black/10 dark:border-white/10 px-6 py-4">
                        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">{title}</h3>
                        <button
                            onClick={onClose}
                            aria-label="Close dialog"
                            className="p-1.5 rounded-xl text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                        >
                            <X size={16} />
                        </button>
                    </div>
                )}
                {children}
            </div>
        </div>,
        document.body
    );
};

// ── useModalA11y ────────────────────────────────────────────────────────────
// The a11y half of <Modal> as a hook, for modals whose shells are custom
// (animations, max-height layouts, unusual widths): focus trap, Escape to
// close, and focus restore on close. Adopt it — do not hand-roll another trap.
export function useModalA11y(
    isOpen: boolean,
    onClose: () => void,
    panelRef: React.RefObject<HTMLElement | null>,
) {
    const lastActiveRef = useRef<Element | null>(null);

    useEffect(() => {
        if (!isOpen) return;
        lastActiveRef.current = document.activeElement;

        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                e.stopPropagation();
                onClose();
                return;
            }
            if (e.key !== 'Tab' || !panelRef.current) return;
            const focusables = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE))
                .filter(el => !el.hasAttribute('disabled'));
            if (focusables.length === 0) return;
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            if (e.shiftKey && document.activeElement === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        };

        document.addEventListener('keydown', onKeyDown, true);
        requestAnimationFrame(() => {
            const first = panelRef.current?.querySelector<HTMLElement>(FOCUSABLE);
            first?.focus();
        });
        return () => {
            document.removeEventListener('keydown', onKeyDown, true);
            if (lastActiveRef.current instanceof HTMLElement) {
                lastActiveRef.current.focus();
            }
        };
    }, [isOpen, onClose, panelRef]);
}
