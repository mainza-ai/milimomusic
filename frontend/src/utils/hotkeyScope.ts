// Global hotkey arbitration between the app-level audio engine and surfaces
// that own keys while mounted (DAW workspace transport, piano-roll editor).
//
// Without this, pressing Space inside the workspace resumed the *global*
// player on top of the session multitrack — two simultaneous audio streams.
//
// Scopes form a STACK: the most recently pushed handler gets first claim.
// A surface consumes a keystroke by returning true; unconsumed keys fall
// through to lower scopes and finally the global engine.

export type HotkeyScopeHandler = (e: KeyboardEvent) => boolean;

const stack: HotkeyScopeHandler[] = [];

/**
 * Push a hotkey scope onto the stack. Returns an unregister function.
 * Later registrations (e.g. an editor inside the workspace) take precedence
 * over earlier ones (e.g. the workspace transport).
 */
export function pushHotkeyScope(handler: HotkeyScopeHandler): () => void {
    stack.push(handler);
    return () => {
        const i = stack.indexOf(handler);
        if (i >= 0) stack.splice(i, 1);
    };
}

/** Returns true if any scope on the stack consumed this keystroke. */
export function consumeHotkey(e: KeyboardEvent): boolean {
    for (let i = stack.length - 1; i >= 0; i--) {
        try {
            if (stack[i](e) === true) return true;
        } catch {
            /* a broken scope must not break the chain */
        }
    }
    return false;
}

/** Shared guard: is this keystroke aimed at a text-entry target? */
export function isTextEntryTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

/** Shared guard: never swallow OS/browser shortcuts (Cmd+Left = Back, etc). */
export function hasModifier(e: KeyboardEvent): boolean {
    return e.ctrlKey || e.metaKey || e.altKey;
}
