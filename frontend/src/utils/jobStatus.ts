/**
 * Canonical job lifecycle predicates.
 *
 * `JobStatus` gained CANCELLED and INTERRUPTED (plan task 0.9) so a user cancel is no
 * longer recorded as a failure. Every place that decided "is this job still running?"
 * by testing `status === 'failed'` must ask these helpers instead, or a cancelled job
 * polls forever and renders as an endless spinner.
 */

export type JobStatusValue =
    | 'queued'
    | 'processing'
    | 'completed'
    | 'failed'
    | 'cancelled'
    | 'interrupted'
    | string;

const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const FAILURE = new Set(['failed', 'interrupted']);

export function isTerminalJobStatus(status?: JobStatusValue | null): boolean {
    return !!status && TERMINAL.has(status);
}

export function isFailedJobStatus(status?: JobStatusValue | null): boolean {
    return !!status && FAILURE.has(status);
}

export function isCancelledJobStatus(status?: JobStatusValue | null): boolean {
    return status === 'cancelled';
}

/** Human label for a terminal or in-flight job. */
export function jobStatusLabel(status?: JobStatusValue | null): string {
    switch (status) {
        case 'queued': return 'Queued';
        case 'processing': return 'Processing';
        case 'completed': return 'Completed';
        case 'failed': return 'Failed';
        case 'cancelled': return 'Cancelled';
        case 'interrupted': return 'Interrupted';
        default: return status ? String(status) : 'Unknown';
    }
}
