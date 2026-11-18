/**
 * Accessibility guard for the video view (plan 7.21).
 *
 * A full axe run needs a browser; this is the cheap regression net that keeps the
 * patterns axe would flag from creeping back: unlabelled selects, dialogs without
 * semantics, mouse-only controls, sub-10px load-bearing type, and motion that
 * ignores `prefers-reduced-motion`.
 *
 * It reads the component sources as text, so it fails in the same commit that
 * introduces the regression rather than weeks later in an audit.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', 'src');

const read = (rel: string) => readFileSync(join(src, rel), 'utf8');

const VIDEO_COMPONENTS = [
    'components/video/VideoCanvasPlayer.tsx',
    'components/video/VideoInspectorDock.tsx',
    'components/video/VideoTimelineTrack.tsx',
    'components/video/VideoTopBar.tsx',
    'components/video/LyricCanvasOverlay.tsx',
    'components/video/ClipRetakeModal.tsx',
    'components/video/KeyframeZoomModal.tsx',
    'components/views/MusicVideosView.tsx',
];

describe('video view accessibility', () => {
    it('labels every select', () => {
        for (const file of VIDEO_COMPONENTS) {
            const text = read(file);
            const selects = text.split('<select');
            // The first chunk is everything before the first <select.
            for (const chunk of selects.slice(1)) {
                // Slice to the closing tag: `>` also appears inside JSX attribute
                // expressions, so "up to the first >" truncates the element.
                const close = chunk.indexOf('</select>');
                const block = close === -1 ? chunk : chunk.slice(0, close);
                expect(block, `${file}: <select> needs an accessible name`).toContain('aria-label');
            }
        }
    });

    it('gives modals dialog semantics and an escape hatch', () => {
        for (const file of ['components/video/ClipRetakeModal.tsx', 'components/video/KeyframeZoomModal.tsx']) {
            const text = read(file);
            expect(text, `${file}: needs role="dialog"`).toContain('role="dialog"');
            expect(text, `${file}: needs aria-modal`).toContain('aria-modal="true"');
            expect(text, `${file}: needs Escape to close`).toContain("e.key === 'Escape'");
        }
    });

    it('keeps clickable timeline blocks keyboard reachable', () => {
        const text = read('components/video/VideoTimelineTrack.tsx');
        // Every onClick on a div must be paired with a role and a tabIndex.
        const clickableDivs = text.match(/<div[^>]*onClick/g) || [];
        if (clickableDivs.length) {
            expect(text).toContain('role="button"');
            expect(text).toContain('tabIndex={0}');
            expect(text).toContain("e.key === 'Enter'");
        }
    });

    it('never ships sub-10px load-bearing type', () => {
        for (const file of VIDEO_COMPONENTS) {
            const text = read(file);
            expect(text, `${file}: text-[8px] is too small to read`).not.toContain('text-[8px]');
            expect(text, `${file}: text-[9px] is too small to read`).not.toContain('text-[9px]');
        }
    });

    it('honours prefers-reduced-motion globally and for JS animation', () => {
        const css = readFileSync(join(src, 'index.css'), 'utf8');
        expect(css).toContain('prefers-reduced-motion');
        // CSS cannot stop a rAF/`Math.sin` animation, so the overlay must gate it.
        const overlay = read('components/video/LyricCanvasOverlay.tsx');
        expect(overlay).toContain('useReducedMotion');
        expect(overlay).toContain('reducedMotion');
    });
});
