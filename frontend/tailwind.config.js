/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // A cool blue-black scale. The app is dark only: a single photograph
        // reads best against near-black, and the HUD accents need the dark.
        canvas: {
          50: '#f5f8fc', 100: '#e6ecf5', 200: '#cfd8e6', 300: '#aab6c8',
          400: '#8494ad', 500: '#5b6b85', 600: '#2a3445', 700: '#1c2433',
          800: '#121826', 900: '#0a0e17', 950: '#05070c',
        },
        accent: { DEFAULT: '#22d3ee', soft: '#67e8f9', deep: '#0891b2' },
        violet: { DEFAULT: '#a78bfa', deep: '#7c3aed' },
        like: '#34d399',
        dislike: '#fb7185',
        save: '#fbbf24',
      },
      fontFamily: {
        sans: ['"Space Grotesk Variable"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono Variable"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(34,211,238,0.35), 0 0 24px -4px rgba(34,211,238,0.45)',
        'glow-like': '0 0 0 1px rgba(52,211,153,0.45), 0 0 22px -4px rgba(52,211,153,0.6)',
        'glow-dislike': '0 0 0 1px rgba(251,113,133,0.45), 0 0 22px -4px rgba(251,113,133,0.6)',
        'glow-save': '0 0 0 1px rgba(251,191,36,0.45), 0 0 22px -4px rgba(251,191,36,0.6)',
        card: '0 30px 80px -20px rgba(0,0,0,0.8), 0 0 60px -30px rgba(34,211,238,0.35)',
      },
      keyframes: {
        'pulse-dot': { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.35' } },
        spin: { to: { transform: 'rotate(360deg)' } },
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'pulse-dot': 'pulse-dot 1.8s ease-in-out infinite',
        'spin-slow': 'spin 1.1s linear infinite',
        'fade-up': 'fade-up 0.35s ease-out both',
      },
    },
  },
  plugins: [],
}
