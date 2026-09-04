import { useCallback, useMemo, useState } from 'react';

type Validators<T> = Partial<Record<keyof T, (value: string) => string | null>>;

/**
 * Shared form state for the artist editors: values, per-field validation
 * errors (computed, never stale), touched tracking, and validity.
 *
 * Fields are strings — every artist form field is (name, bio, tags, lore).
 * `showError(field)` only reports after blur or submit attempt, so users
 * aren't yelled at mid-typing.
 */
export function useValidatedForm<T extends Record<string, string>>(initial: T, validators: Validators<T>) {
    const [values, setValues] = useState<T>(initial);
    const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});
    const [submitAttempted, setSubmitAttempted] = useState(false);

    const setField = useCallback((field: keyof T, value: string) => {
        setValues(prev => ({ ...prev, [field]: value }));
    }, []);

    const errors = useMemo(() => {
        const out: Partial<Record<keyof T, string>> = {};
        for (const key of Object.keys(values) as (keyof T)[]) {
            const message = validators[key]?.(values[key]);
            if (message) out[key] = message;
        }
        return out;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [values]);

    const isValid = useMemo(() => Object.keys(errors).length === 0, [errors]);

    const markTouched = useCallback((field: keyof T) => {
        setTouched(prev => ({ ...prev, [field]: true }));
    }, []);

    const markSubmitAttempted = useCallback(() => setSubmitAttempted(true), []);

    const showError = useCallback((field: keyof T): string | null => {
        if (!touched[field] && !submitAttempted) return null;
        return errors[field] || null;
    }, [touched, submitAttempted, errors]);

    const reset = useCallback(() => {
        setValues(initial);
        setTouched({});
        setSubmitAttempted(false);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const setAll = useCallback((next: T) => setValues(next), []);

    return { values, setField, errors, isValid, markTouched, markSubmitAttempted, showError, reset, setAll };
}
