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
