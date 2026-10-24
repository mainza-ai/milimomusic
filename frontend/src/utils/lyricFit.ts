/**
 * Lyric-fit estimator: singable words vs requested duration.
 * Central rule — a track needs roughly (words / pace) * 60 seconds of audio
 * or the tail gets cut off. Pace guesses by genre tags (words per minute).
 */

export function countSingableWords(lyrics: string): number {
    if (!lyrics) return 0;
    const stripped = lyrics
        .replace(/\[.*?\]/g, ' ') // section headers
        .replace(/\(.*?\)/g, ' ') // ad-libs / stage directions
        .replace(/[^A-Za-z'\s]/g, ' ');
    const words = stripped.split(/\s+/).filter(w => w.replace(/'/g, '').length > 1);
    return words.length;
}

export function paceForTags(tags?: string): number {
    const t = (tags || '').toLowerCase();
    if (/(drill|trap|rap|hip-?hop|grime|afrobeats|drum\s*&?\s*bass|dnb|speed)/.test(t)) return 160;
    if (/(ballad|soul|gospel|worship|acoustic|folk|lullaby)/.test(t)) return 105;
    return 125;
}

export interface LyricFit {
    words: number;
    paceWpm: number;
    neededSec: number;
    fits: boolean;
    shortfallSec: number;
}

export function estimateLyricFit(lyrics: string, tags: string | undefined, durationSec: number): LyricFit {
    const words = countSingableWords(lyrics);
    const paceWpm = paceForTags(tags);
    const neededSec = words > 0 ? Math.ceil((words / paceWpm) * 60) : 0;
    const shortfallSec = Math.max(0, neededSec - durationSec);
    return { words, paceWpm, neededSec, fits: shortfallSec <= 0, shortfallSec };
}

export function formatFitTime(secs: number): string {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
}

export interface TimedLyricLine {
    text?: string;
    is_section?: boolean;
    start?: number;
    end?: number;
}

/** Mirror of the backend split: lines at/after the cut (1.5s overlap), with headers. */
export function splitRemainingLyrics(lines: TimedLyricLine[] | null, cutSec: number): string {
    if (!lines || !Array.isArray(lines)) return '';
    const kept: string[] = [];
    let pendingHeader: string | null = null;
    let headerEmittedFor: string | null = null;
    for (const ln of lines) {
        const text = (ln.text || '').trim();
        if (!text) continue;
        if (ln.is_section) {
            pendingHeader = text;
            continue;
        }
        const start = Number(ln.start || 0);
        const end = Number(ln.end ?? start);
        if (!(end > cutSec - 1.5)) continue;
        if (pendingHeader && headerEmittedFor !== pendingHeader) {
            kept.push(pendingHeader);
            headerEmittedFor = pendingHeader;
        }
        kept.push(text);
    }
    return kept.join('\n').trim();
}

/** First N non-header lines of a remainder, for preview display. */
export function previewRemainder(remainder: string, maxLines = 3): string[] {
    return remainder.split('\n').filter(l => l.trim() && !/^\[.*\]$/.test(l.trim())).slice(0, maxLines);
}
