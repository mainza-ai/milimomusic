/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Display"', '"SF Pro Text"', 'Inter', 'system-ui', 'sans-serif'],
                mono: ['"SF Mono"', '"JetBrains Mono"', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
                serif: ['"New York"', 'Georgia', 'serif'],
            },
            colors: {
                // ── Disciplined-glass surface tiers ─────────────────────────
                // Canonical replacements for the ~20 ad-hoc dark hexes that were
                // scattered across components. New/edited code MUST use these
                // instead of raw hex values.
                surface: {
                    light: {
                        DEFAULT: '#f5f5f7',   // app canvas (light)
                        raised: '#ffffff',    // cards / panels
                        overlay: '#ffffff',   // modals / popovers
                        sunken: '#ebebf0',    // editor wells (piano roll bg)
                    },
                    // dark:* variants are used via `dark:bg-surface-*` below.
                    DEFAULT: '#0c0e14',       // app canvas (dark)
                    raised: '#12141c',        // primary card surface (was #12141c/#141620/#181a24 soup)
                    overlay: '#181a24',       // elevated surfaces: menus, popovers
                    sunken: '#0a0c12',        // editor wells (piano roll / notation bg)
                    deep: '#090b10',          // workspace shell background
                },
                apple: {
                    blue: {
                        light: "#0071e3",
                        DEFAULT: "#0077ed",
                        dark: "#2997ff"
                    },
                    cyan: {
                        light: "#32ade6",
                        DEFAULT: "#00c7be",
                        dark: "#00d2ff"
                    },
                    teal: {
                        light: "#30b0c7",
                        DEFAULT: "#20c997",
                        dark: "#64d2ff"
                    },
                    purple: {
                        light: "#af52de",
                        DEFAULT: "#9b51e0",
                        dark: "#bf5af2"
                    },
                    gray: {
                        50: "#fbfbfd",
                        100: "#f5f5f7",
                        200: "#e8e8ed",
                        300: "#d2d2d7",
                        400: "#86868b",
                        500: "#6e6e73",
                        600: "#424245",
                        700: "#333336",
                        800: "#242426",
                        900: "#1d1d1f",
                        950: "#0c0c0d"
                    }
                }
            },
            boxShadow: {
                'apple-sm': '0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06)',
                'apple-md': '0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04)',
                'apple-lg': '0 12px 32px rgba(0, 0, 0, 0.12), 0 4px 12px rgba(0, 0, 0, 0.06)',
                // Was used 8x while undefined → flagship player/modals silently
                // rendered with NO elevation. Now defined.
                'apple-2xl': '0 24px 64px rgba(0, 0, 0, 0.22), 0 8px 24px rgba(0, 0, 0, 0.12)',
                'apple-glow': '0 0 20px rgba(0, 210, 255, 0.25)'
            },
            backdropBlur: {
                '2xl': '40px',
                '3xl': '64px'
            },
            // ── Motion system ───────────────────────────────────────────────
            // These utility names were used ~43x across the app while the
            // keyframes never existed anywhere — every entrance animation was
            // a silent no-op. Defining them here switches the app's entire
            // motion personality back on.
            keyframes: {
                'fade-in': {
                    from: { opacity: '0' },
                    to: { opacity: '1' },
                },
                'slide-up': {
                    from: { opacity: '0', transform: 'translateY(12px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
                'slide-down': {
                    from: { opacity: '0', transform: 'translateY(-12px)' },
                    to: { opacity: '1', transform: 'translateY(0)' },
                },
                'scale-up': {
                    from: { opacity: '0', transform: 'scale(0.96)' },
                    to: { opacity: '1', transform: 'scale(1)' },
                },
                indeterminate: {
                    from: { transform: 'translateX(-120%)' },
                    to: { transform: 'translateX(320%)' },
                },
            },
            animation: {
                'fade-in': 'fade-in 0.25s ease-out both',
                'slide-up': 'slide-up 0.3s ease-out both',
                'slide-down': 'slide-down 0.3s ease-out both',
                'scale-up': 'scale-up 0.2s ease-out both',
                'spin-slow': 'spin 3s linear infinite',
            },
        },
    },
    plugins: [],
}
