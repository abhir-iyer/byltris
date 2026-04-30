/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['var(--font-display)'],
        body:    ['var(--font-body)'],
        mono:    ['var(--font-mono)'],
      },
      colors: {
        ink:    '#08090a',
        panel:  '#0e1014',
        border: '#1c1f26',
        muted:  '#3a3f4a',
        ghost:  '#6b7280',
        soft:   '#9ca3af',
        bright: '#e2e8f0',
        white:  '#f8fafc',
        signal: '#00d4aa',
        warn:   '#f59e0b',
        danger: '#ef4444',
        blue:   '#3b82f6',
      },
      backgroundImage: {
        'grid-faint': `linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                       linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)`,
      },
      backgroundSize: {
        'grid': '60px 60px',
      },
      animation: {
        'fade-up':   'fadeUp 0.6s ease forwards',
        'fade-in':   'fadeIn 0.8s ease forwards',
        'pulse-slow':'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'count':     'countUp 2s ease-out forwards',
      },
      keyframes: {
        fadeUp:  { from: { opacity: 0, transform: 'translateY(24px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        countUp: { from: { opacity: 0 }, to: { opacity: 1 } },
      },
    },
  },
  plugins: [],
}
