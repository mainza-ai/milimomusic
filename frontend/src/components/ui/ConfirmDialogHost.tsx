/**
 * Global confirm dialog host (plan 5.10).
 *
 * Mount once near the app root. Renders the pending `confirmAction()` request as a
 * real accessible modal (focus trap + Escape via `Modal`), replacing
 * `window.confirm` throughout the app.
 */
import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { Modal, Button } from './primitives';
import { useConfirmStore } from '../../utils/confirm';

export const ConfirmDialogHost: React.FC = () => {
    const request = useConfirmStore(s => s.request);
    const settle = useConfirmStore(s => s.settle);

    return (
        <Modal
            isOpen={Boolean(request)}
            onClose={() => settle(false)}
            title={request?.title || ''}
            widthClass="max-w-md"
        >
            <div className="space-y-4">
                <div className="flex items-start gap-3">
                    {request?.tone === 'danger' && (
                        <span className="mt-0.5 shrink-0 rounded-lg bg-rose-500/10 p-2 text-rose-600 dark:text-rose-400">
                            <AlertTriangle size={16} />
                        </span>
                    )}
                    <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                        {request?.body || 'This action cannot be undone.'}
                    </p>
                </div>

                <div className="flex items-center justify-end gap-2 pt-1">
                    <Button variant="secondary" size="sm" onClick={() => settle(false)}>
                        {request?.cancelLabel || 'Cancel'}
                    </Button>
                    <Button
                        variant={request?.tone === 'danger' ? 'danger' : 'primary'}
                        size="sm"
                        onClick={() => settle(true)}
                    >
                        {request?.confirmLabel || 'Confirm'}
                    </Button>
                </div>
            </div>
        </Modal>
    );
};
