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

/**
 * Split a JSX element into its opening-tag attributes and its children.
 *
 * `>` appears inside attribute expressions (`onClick={() => ...}`), so the first
 * `>` is not the end of the tag — depth counting on `{}`/quotes is.
 */
function splitJsxTag(source: string): { attributes: string; children: string } {
    let depth = 0;
    let quote: string | null = null;
    for (let i = 0; i < source.length; i += 1) {
        const ch = source[i];
        if (quote) {
            if (ch === quote && source[i - 1] !== '\\') quote = null;
            continue;
        }
        if (ch === '"' || ch === "'" || ch === '`') { quote = ch; continue; }
        if (ch === '{') depth += 1;
        else if (ch === '}') depth = Math.max(0, depth - 1);
        else if (ch === '>' && depth === 0) {
            return { attributes: source.slice(0, i), children: source.slice(i + 1) };
        }
    }
    return { attributes: source, children: '' };
}

/**
 * True when a button's children can render visible text.
 *
 * Limitations, stated rather than hidden: a conditional that renders only elements
 * (`{busy ? <Loader2/> : <Trash2/>}`) is treated as "unknown" and not flagged,
 * because static analysis cannot prove it has no text. A pure icon child
 * (`<Trash2 size={12} />`) is flagged.
 */
function childrenRenderText(children: string): boolean {
    if (/[A-Za-z0-9]/.test(literalTextOf(children))) return true;

    for (const expr of topLevelExpressions(children)) {
        const trimmed = expr.trim();
        // Any conditional or non-trivial logic: cannot be judged statically.
        if (/[?:]/.test(trimmed)) return true;
        // A quoted string renders text.
        if (/['"][^'"]*[A-Za-z0-9][^'"]*['"]/.test(trimmed)) return true;
        // `{cat.label}` renders a value.
        if (/^[A-Za-z_$][\w$]*(\.[\w$]+)*$/.test(trimmed)) return true;
        // Numbers/booleans render text too.
        if (/^[0-9.+-]+$/.test(trimmed)) return true;
        // An element inside the expression may itself carry text.
        if (trimmed.includes('<') && childrenRenderText(trimmed)) return true;
    }
    return false;
}

/** Literal text: everything outside `<elements>` and `{expressions}`. */
function literalTextOf(source: string): string {
    let out = '';
    for (let i = 0; i < source.length; i += 1) {
        const ch = source[i];
        if (ch === '<') { i = skipBalanced(source, i, '<', '>'); continue; }
        if (ch === '{') { i = skipBalanced(source, i, '{', '}'); continue; }
        out += ch;
    }
    return out;
}

/**
 * Expressions directly inside this JSX level. Expressions inside an element's
 * *attributes* belong to that element, not to its parent's content.
 */
function topLevelExpressions(source: string): string[] {
    const found: string[] = [];
    for (let i = 0; i < source.length; i += 1) {
        const ch = source[i];
        if (ch === '<') { i = skipBalanced(source, i, '<', '>'); continue; }
        if (ch === '{') {
            const end = skipBalanced(source, i, '{', '}');
            found.push(source.slice(i + 1, end));
            i = end;
        }
    }
    return found;
}

/** Index of the closing delimiter, honouring nesting and quoted strings. */
function skipBalanced(source: string, start: number, open: string, close: string): number {
    let depth = 0;
    let quote: string | null = null;
    for (let i = start; i < source.length; i += 1) {
        const ch = source[i];
        if (quote) {
            if (ch === quote && source[i - 1] !== '\\') quote = null;
            continue;
        }
        if (ch === '"' || ch === "'" || ch === '`') { quote = ch; continue; }
        if (ch === open) depth += 1;
        else if (ch === close) {
            depth -= 1;
            if (depth === 0) return i;
        }
    }
    return source.length - 1;
}

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

    it('gives every icon-only button an accessible name', () => {
        // Icon-only buttons are the most common axe failure. A static check cannot
        // resolve what an arbitrary expression renders, so this rule is deliberately
        // conservative: it flags a button whose children are pure elements (optionally
        // wrapped in identifier-path expressions) and nothing that renders text.
        for (const file of VIDEO_COMPONENTS) {
            const text = read(file);
            const blocks = text.split('<button');
            for (let i = 1; i < blocks.length; i += 1) {
                const close = blocks[i].indexOf('</button>');
                const block = blocks[i].slice(0, close === -1 ? undefined : close);
                const node = splitJsxTag(block);

                const named =
                    node.attributes.includes('aria-label') ||
                    node.attributes.includes('title=') ||
                    childrenRenderText(node.children);
                expect(named, `${file}: <button> #${i} has no accessible name`).toBe(true);
            }
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

describe('accessible-name heuristic', () => {
    it('flags a pure icon child', () => {
        expect(childrenRenderText('<Trash2 size={12} />')).toBe(false);
        expect(childrenRenderText('')).toBe(false);
    });

    it('accepts literal text, quoted strings and identifier paths', () => {
        expect(childrenRenderText('<span>Delete take</span>')).toBe(true);
        expect(childrenRenderText("{busy ? 'Saving…' : 'Save'}")).toBe(true);
        expect(childrenRenderText('{cat.label}')).toBe(true);
        expect(childrenRenderText('<span>{cast}</span>')).toBe(true);
    });

    it('does not false-positive on a conditional that may render text', () => {
        expect(childrenRenderText('{busy ? <Loader2 /> : <Trash2 />}')).toBe(true);
    });
});
