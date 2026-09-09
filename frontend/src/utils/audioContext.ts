export const getAudioContext = (): AudioContext => {
    const w = window as any;
    if (!w._milimoAudioContext || w._milimoAudioContext.state === 'closed') {
        const AudioContextClass = w.AudioContext || w.webkitAudioContext;
        w._milimoAudioContext = new AudioContextClass();
    }
    return w._milimoAudioContext;
};

/**
 * Ensures the Web Audio context is unsuspended.
 * Must be called in response to or attached to a user gesture.
 */
export const unlockAudioContext = async (): Promise<void> => {
    try {
        const ctx = getAudioContext();
        if (ctx && ctx.state === 'suspended') {
            await ctx.resume();
        }
    } catch {
        // Ignored if browser still restricts
    }
};

// Auto-bind first user interaction to guarantee AudioContext is never left suspended,
// but defer execution out of the synchronous pointerdown stack so the initial frame paints immediately.
if (typeof window !== 'undefined') {
    const onUserInteraction = () => {
        window.removeEventListener('pointerdown', onUserInteraction);
        window.removeEventListener('keydown', onUserInteraction);
        window.removeEventListener('touchstart', onUserInteraction);
        setTimeout(() => {
            void unlockAudioContext();
        }, 0);
    };
    window.addEventListener('pointerdown', onUserInteraction, { passive: true });
    window.addEventListener('keydown', onUserInteraction, { passive: true });
    window.addEventListener('touchstart', onUserInteraction, { passive: true });
}
