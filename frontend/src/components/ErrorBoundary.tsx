import React from 'react';
import { Music, RefreshCw, Undo2, ChevronDown } from 'lucide-react';

interface ErrorBoundaryProps {
    children: React.ReactNode;
    /** Scoped boundary name (e.g. "Artists") shown in the crash card so the
     *  user knows which section failed — the rest of the studio stays alive. */
    sectionName?: string;
}

interface ErrorBoundaryState {
    error: Error | null;
    componentStack: string | null;
}

/**
 * Root render-crash guard.
 *
 * Before this existed, a single malformed DB payload (e.g. an unparseable
 * `*_json` column) threw during render and took the whole studio down to a
 * white screen with no recovery path. The boundary converts that into a
 * branded recovery surface: the user can return to Explore (dropping only the
 * offending view state) or reload cleanly, and the crash is captured in the
 * console with its full component stack for diagnosis.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = { error: null, componentStack: null };

    static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
        return { error };
    }

    componentDidCatch(error: Error, info: React.ErrorInfo) {
        // Full stack + component tree lands in the console for diagnosis.
        console.error('[Milimo] Render crash captured:', error, info.componentStack);
        this.setState({ componentStack: info.componentStack ?? null });
    }

    handleBackToExplore = () => {
        try {
            const url = new URL(window.location.href);
            url.searchParams.delete('view');
            url.searchParams.delete('track');
            window.history.replaceState({}, '', url.toString());
            window.dispatchEvent(new PopStateEvent('popstate'));
        } catch { /* deep-link parse failure must not block recovery */ }
        this.setState({ error: null, componentStack: null });
    };

    handleReload = () => {
        window.location.reload();
    };

    render() {
        const { error, componentStack } = this.state;
        if (!error) return this.props.children;

        return (
            <div className="fixed inset-0 z-[200] flex items-center justify-center p-6 bg-slate-100 dark:bg-[#0c0e14] text-slate-900 dark:text-slate-100">
                <div className="w-full max-w-xl rounded-3xl border border-black/[0.06] dark:border-white/10 bg-white/90 dark:bg-[#141620]/95 backdrop-blur-2xl shadow-apple-2xl p-8 space-y-5">
                    <div className="flex items-center gap-3">
                        <div className="w-11 h-11 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
                            <Music size={20} className="text-rose-500" />
                        </div>
                        <div>
                            <h1 className="text-lg font-extrabold tracking-tight">
                                {this.props.sectionName ? `The ${this.props.sectionName} section hit a wrong note` : 'The studio hit a wrong note'}
                            </h1>
                            <p className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-0.5">
                                {this.props.sectionName
                                    ? `A rendering error occurred in ${this.props.sectionName} — nothing was lost.`
                                    : 'A rendering error occurred — nothing was lost.'}
                            </p>
                        </div>
                    </div>

                    <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                        This view crashed while drawing. Your projects and tracks are safe on disk —
                        jump back to Explore, or reload the studio for a clean start.
                    </p>

                    <pre className="max-h-24 overflow-auto rounded-xl bg-black/[0.04] dark:bg-white/[0.05] border border-black/[0.06] dark:border-white/10 p-3 text-[11px] font-mono text-slate-600 dark:text-slate-400 whitespace-pre-wrap break-words">
                        {error.message || String(error)}
                    </pre>

                    <div className="flex items-center gap-2.5">
                        <button
                            onClick={this.handleBackToExplore}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 font-bold text-xs shadow-md shadow-teal-500/20 active:scale-[0.98] transition-all"
                        >
                            <Undo2 size={14} />
                            <span>Back to Explore</span>
                        </button>
                        <button
                            onClick={this.handleReload}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-black/[0.04] dark:bg-white/5 hover:bg-black/[0.08] dark:hover:bg-white/10 border border-black/[0.06] dark:border-white/10 text-slate-700 dark:text-slate-200 font-bold text-xs active:scale-[0.98] transition-all"
                        >
                            <RefreshCw size={14} />
                            <span>Reload Studio</span>
                        </button>
                    </div>

                    {componentStack && (
                        <details className="group">
                            <summary className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 cursor-pointer select-none">
                                <ChevronDown size={12} className="group-open:rotate-180 transition-transform" />
                                Component stack
                            </summary>
                            <pre className="mt-2 max-h-48 overflow-auto rounded-xl bg-black/[0.04] dark:bg-white/[0.05] p-3 text-[10px] font-mono text-slate-500 dark:text-slate-500 whitespace-pre-wrap break-words select-text">
                                {componentStack}
                            </pre>
                        </details>
                    )}
                </div>
            </div>
        );
    }
}
