/**
 * Parse a DB-backed JSON column without ever throwing during render.
 * A single corrupt row must degrade to an empty default — not white-screen
 * the studio (the exact failure class that motivated the root ErrorBoundary).
 */
export function safeJsonParse<T>(raw: unknown, fallback: T, label = 'json payload'): T {
    if (raw === null || raw === undefined) return fallback;
    if (typeof raw !== 'string') return raw as T;
    try {
        const parsed = JSON.parse(raw);
        // Guard against primitives where an object/array contract exists.
        if (parsed === null || parsed === undefined) return fallback;
        return parsed as T;
    } catch (e) {
        console.warn(`[Milimo] Corrupt ${label} ignored, falling back to empty:`, e);
        return fallback;
    }
}
