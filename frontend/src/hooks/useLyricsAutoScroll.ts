import { useEffect, useRef, useState, useCallback } from 'react';

export interface UseLyricsAutoScrollOptions {
    activeLineIndex: number;
    enabled?: boolean;
    resumeDelayMs?: number;
}

export interface UseLyricsAutoScrollReturn {
    containerRef: React.RefObject<HTMLDivElement | null>;
    isAutoScrollPaused: boolean;
    resumeAutoScroll: () => void;
    handleScroll: () => void;
    handleWheel: () => void;
    scrollToActiveLine: (smooth?: boolean) => void;
}

/**
 * useLyricsAutoScroll
 *
 * Professional, container-scoped lyrics auto-scrolling hook modeled after Apple Music / Spotify.
 *
 * Features:
 * 1. Strictly container-scoped: Uses container.scrollTo() rather than window-scrolling scrollIntoView().
 * 2. Perfect vertical centering of the currently sung lyric line.
 * 3. Smart user scroll detection: Pauses auto-scrolling when user manually scrolls or wheels to browse lyrics.
 * 4. Automatic resumption after user scroll idle timeout (default 4s) or manual 1-click "Sync Lyrics" trigger.
 * 5. Instant scroll alignment on initial view mount / tab toggle.
 */
export function useLyricsAutoScroll({
    activeLineIndex,
    enabled = true,
    resumeDelayMs = 4000
}: UseLyricsAutoScrollOptions): UseLyricsAutoScrollReturn {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [isAutoScrollPaused, setIsAutoScrollPaused] = useState(false);
    const isProgrammaticScrollRef = useRef(false);
    const programmaticScrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const userScrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const scrollToActiveLine = useCallback((smooth: boolean = true) => {
        const container = containerRef.current;
        if (!container || activeLineIndex < 0 || container.clientHeight === 0) return;

        const activeEl = container.querySelector(`[data-line-idx="${activeLineIndex}"]`) as HTMLElement | null;
        if (!activeEl) return;

        const containerRect = container.getBoundingClientRect();
        const activeRect = activeEl.getBoundingClientRect();

        // Compute exact offset of active element relative to container's scroll content top
        const relativeTop = activeRect.top - containerRect.top + container.scrollTop;
        const targetScrollTop = relativeTop - (container.clientHeight / 2) + (activeRect.height / 2);
        const finalScrollTop = Math.max(0, targetScrollTop);

        // Skip micro-adjustments
        if (Math.abs(container.scrollTop - finalScrollTop) < 4) return;

        isProgrammaticScrollRef.current = true;
        container.scrollTo({
            top: finalScrollTop,
            behavior: smooth ? 'smooth' : 'auto'
        });

        if (programmaticScrollTimerRef.current) {
            clearTimeout(programmaticScrollTimerRef.current);
        }
        programmaticScrollTimerRef.current = setTimeout(() => {
            isProgrammaticScrollRef.current = false;
        }, smooth ? 600 : 50);
    }, [activeLineIndex]);

    // Handle user wheel interaction
    const handleWheel = useCallback(() => {
        if (!enabled) return;
        setIsAutoScrollPaused(true);
        if (userScrollTimerRef.current) {
            clearTimeout(userScrollTimerRef.current);
        }
        userScrollTimerRef.current = setTimeout(() => {
            setIsAutoScrollPaused(false);
        }, resumeDelayMs);
    }, [enabled, resumeDelayMs]);

    // Handle user manual scroll interaction
    const handleScroll = useCallback(() => {
        if (isProgrammaticScrollRef.current) return;
        if (!enabled) return;

        setIsAutoScrollPaused(true);

        if (userScrollTimerRef.current) {
            clearTimeout(userScrollTimerRef.current);
        }
        userScrollTimerRef.current = setTimeout(() => {
            setIsAutoScrollPaused(false);
        }, resumeDelayMs);
    }, [enabled, resumeDelayMs]);

    // Manual or programmatic re-sync
    const resumeAutoScroll = useCallback(() => {
        if (userScrollTimerRef.current) {
            clearTimeout(userScrollTimerRef.current);
        }
        setIsAutoScrollPaused(false);
        scrollToActiveLine(true);
    }, [scrollToActiveLine]);

    // Reset paused state when view becomes enabled
    useEffect(() => {
        if (enabled) {
            setIsAutoScrollPaused(false);
        }
    }, [enabled]);

    // Auto-scroll when active line index changes while not paused
    useEffect(() => {
        if (!enabled || activeLineIndex < 0) return;
        if (isAutoScrollPaused) return;

        const rafId = requestAnimationFrame(() => {
            scrollToActiveLine(true);
        });
        return () => cancelAnimationFrame(rafId);
    }, [activeLineIndex, enabled, isAutoScrollPaused, scrollToActiveLine]);

    // Instant alignment when enabled state toggles (e.g. switching to lyrics tab/view)
    useEffect(() => {
        if (!enabled || activeLineIndex < 0) return;
        const timer = setTimeout(() => {
            scrollToActiveLine(false);
        }, 80);
        return () => clearTimeout(timer);
    }, [enabled, scrollToActiveLine, activeLineIndex]);

    // Cleanup timers on unmount
    useEffect(() => {
        return () => {
            if (programmaticScrollTimerRef.current) clearTimeout(programmaticScrollTimerRef.current);
            if (userScrollTimerRef.current) clearTimeout(userScrollTimerRef.current);
        };
    }, []);

    return {
        containerRef,
        isAutoScrollPaused,
        resumeAutoScroll,
        handleScroll,
        handleWheel,
        scrollToActiveLine
    };
}
