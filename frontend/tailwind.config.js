/** @type {import('tailwindcss').Config} */
// Concept: a darkroom viewfinder. A dark, blue-biased instrument body around
// one lit photograph; readouts in mono; squared display type like engraved
// instrument labels; phosphor cyan for the instrument, violet for the model.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Instrument body: neutrals biased toward the cyan accent.
        canvas: {
          50: '#f3f7fb', 100: '#e2eaf3', 200: '#c9d5e3', 300: '#a3b2c6',
          400: '#7b8ca5', 500: '#55657e', 600: '#2b3647', 700: '#1b2433',
          800: '#111824', 900: '#0a0f18', 950: '#05080d',
        },
        accent: { DEFAULT: '#3fd8f0', soft: '#8ae9f7', deep: '#0e9bb4' },
        violet: { DEFAULT: '#a594f9', deep: '#6d5ae6' },
        like: '#3ddc97',
        dislike: '#ff6b81',
        save: '#ffc145',
      },
      fontFamily: {
        display: ['"Chakra Petch"', '"Geist Variable"', 'system-ui', 'sans-serif'],
        sans: ['"Geist Variable"', 'system-ui', 'sans-serif'],
        mono: ['"Geist Mono Variable"', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        // Squared by default; roundness is reserved for pills and the photo.
        plate: '5px',
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(63,216,240,0.35), 0 0 24px -6px rgba(63,216,240,0.5)',
        'glow-like': '0 0 0 1px rgba(61,220,151,0.5), 0 0 26px -4px rgba(61,220,151,0.65)',
        'glow-dislike': '0 0 0 1px rgba(255,107,129,0.5), 0 0 26px -4px rgba(255,107,129,0.65)',
        'glow-save': '0 0 0 1px rgba(255,193,69,0.5), 0 0 26px -4px rgba(255,193,69,0.65)',
        photo: '0 40px 90px -30px rgba(0,0,0,0.9)',
      },
      keyframes: {
        'pulse-dot': { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.35' } },
        spin: { to: { transform: 'rotate(360deg)' } },
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        scan: {
          from: { transform: 'translateY(-100%)' },
          to: { transform: 'translateY(100%)' },
        },
      },
      animation: {
        'pulse-dot': 'pulse-dot 1.8s ease-in-out infinite',
        'spin-slow': 'spin 1.1s linear infinite',
        'fade-up': 'fade-up 0.35s ease-out both',
        scan: 'scan 1.6s cubic-bezier(.4,0,.2,1) both',
      },
    },
  },
  plugins: [],
}
