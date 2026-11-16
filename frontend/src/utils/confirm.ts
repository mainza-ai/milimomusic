/**
 * Promise-based confirm dialog (plan 5.10).
 *
 * `window.confirm` and `window.alert` block the whole app, are unstyled, have no
 * accessibility semantics on mobile, cannot be tested, and several call sites
 * discarded the server's reason entirely. This module provides `confirmAction()`
 * and `showError()` built on the same modal primitives as the rest of the app.
 *
 * Usage:
 *     if (!(await confirmAction({ title: 'Delete track?', tone: 'danger' }))) return;
 *     showError(err, 'Delete failed');
 */
import { create } from 'zustand';
import { toast } from './toast';

interface ConfirmRequest {
    title: string;
    body?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    tone?: 'default' | 'danger';
    resolve: (ok: boolean) => void;
}

interface ConfirmStore {
    request: ConfirmRequest | null;
    open: (req: Omit<ConfirmRequest, 'resolve'>) => Promise<boolean>;
    settle: (ok: boolean) => void;
}

export const useConfirmStore = create<ConfirmStore>((set, get) => ({
    request: null,
    open: (req) => new Promise<boolean>((resolve) => {
        set({ request: { ...req, resolve } });
    }),
    settle: (ok) => {
        const current = get().request;
        if (current) current.resolve(ok);
        set({ request: null });
    },
}));

export function confirmAction(req: Omit<ConfirmRequest, 'resolve'>): Promise<boolean> {
    return useConfirmStore.getState().open(req);
}

/** Pull the most useful message out of an axios/fetch/Error failure. */
export function errorMessage(err: unknown, fallback = 'Something went wrong'): string {
    const e = err as {
        response?: { data?: { error?: { message?: string }; detail?: unknown } };
        message?: string;
    };
    const data = e?.response?.data;
    const fromApi = data?.error?.message;
    if (fromApi) return fromApi;
    const detail = data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') {
        const msg = (detail as { error?: { message?: string }; message?: string }).error?.message
            || (detail as { message?: string }).message;
        if (msg) return msg;
    }
    return e?.message || fallback;
}

/** Non-blocking error surface that always shows the server's own reason. */
export function showError(err: unknown, fallback = 'Something went wrong'): void {
    toast(errorMessage(err, fallback), 'error');
    // Keep a trace for debugging without blocking the UI.
    console.error(fallback, err);
}
