import { defineConfig } from 'vitest/config';

// Unit tests live in `tests/`; Playwright specs live in `e2e/` and must not be
// collected by vitest (they use a different runner and a browser context).
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
    environment: 'node',
  },
});
