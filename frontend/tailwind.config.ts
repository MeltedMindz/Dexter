import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#6366F1', // Indigo - Trust, technology
        success: '#10B981', // Emerald - Profits, positive
        warning: '#F59E0B', // Amber - Caution, pending
        error: '#EF4444', // Red - Losses, danger
        neutral: '#6B7280', // Gray - Text, backgrounds
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
export default config