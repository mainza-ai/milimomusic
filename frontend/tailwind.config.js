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
                'apple-glow': '0 0 20px rgba(0, 210, 255, 0.25)'
            },
            backdropBlur: {
                '2xl': '40px',
                '3xl': '64px'
            }
        },
    },
    plugins: [],
}
