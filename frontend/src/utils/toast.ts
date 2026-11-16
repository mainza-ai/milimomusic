// Self-contained toast system (no provider wiring needed).
type ToastKind = 'info' | 'success' | 'error';
const COLORS: Record<ToastKind, string> = {
  info: 'border-slate-400/40', success: 'border-emerald-500/60',
  error: 'border-red-500/60',
};
export function toast(message: string, kind: ToastKind = 'info', ms = 3800) {
  if (typeof document === 'undefined') return;
  const host =
    document.getElementById('milimo-toasts') ??
    Object.assign(document.body.appendChild(Object.assign(document.createElement('div'), { id: 'milimo-toasts' })), {
      style: 'position:fixed;bottom:1rem;right:1rem;z-index:9999;display:flex;flex-direction:column;gap:.5rem;',
    });
  // Keep at most three visible; older ones are dropped, not stacked off-screen.
  while (host.childElementCount >= 3 && host.firstElementChild) {
    host.firstElementChild.remove();
  }
  const el = document.createElement('div');
  el.className = `apple-card ${COLORS[kind]}`;
  // Errors must interrupt a screen reader; polite status is right for the rest.
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.setAttribute('aria-live', kind === 'error' ? 'assertive' : 'polite');
  el.setAttribute('aria-atomic', 'true');
  el.style.cssText = 'padding:.65rem .9rem;font-size:12px;font-weight:600;max-width:320px;border-width:1px;animation:milimo-toast-in .18s ease-out;';
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .25s'; setTimeout(() => el.remove(), 260); }, ms);
}
