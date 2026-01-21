/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            colors: {
                glass: {
                    surface: "rgba(255, 255, 255, 0.4)",
                    border: "rgba(255, 255, 255, 0.2)",
                    text: "rgba(0, 0, 0, 0.8)",
                }
            },
            backdropBlur: {
                xs: '2px',
            }
        },
    },
    plugins: [],
}
