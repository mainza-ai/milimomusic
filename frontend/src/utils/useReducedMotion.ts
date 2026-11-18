import { useEffect, useState } from 'react';

/**
 * Whether the user asked their OS for less motion (plan 7.21).
 *
 * The global CSS rule in `index.css` covers CSS animations and transitions, but
 * JS-driven animation (the karaoke backdrop bars) keeps moving regardless — this
 * hook is for those cases.
 */
export function useReducedMotion(): boolean {
    const query = '(prefers-reduced-motion: reduce)';
    const [reduced, setReduced] = useState<boolean>(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return false;
        return window.matchMedia(query).matches;
    });

    useEffect(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return;
        const media = window.matchMedia(query);
        const onChange = () => setReduced(media.matches);
        onChange();
        media.addEventListener?.('change', onChange);
        return () => media.removeEventListener?.('change', onChange);
    }, []);

    return reduced;
}
