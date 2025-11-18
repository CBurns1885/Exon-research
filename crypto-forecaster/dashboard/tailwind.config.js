/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        crypto: {
          green: '#10b981',
          red: '#ef4444',
          blue: '#3b82f6',
          purple: '#8b5cf6',
          orange: '#f59e0b',
        },
        bitcoin: '#F7931A',
        ethereum: '#627EEA',
        dark: {
          bg: '#0f172a',
          card: '#1e293b',
          border: '#334155',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
      }
    },
  },
  plugins: [],
}
