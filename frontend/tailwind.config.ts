import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Neo-Brutalism Color Palette
        primary: {
          DEFAULT: '#00FF00', // Bright green accent
          light: '#66FF66',
          dark: '#00CC00',
        },
        accent: {
          yellow: '#FFFF00',
          cyan: '#00FFFF',
          magenta: '#FF00FF',
        },
        // High contrast monochrome base
        white: '#FFFFFF',
        black: '#000000',
        gray: {
          100: '#F5F5F5',
          200: '#E5E5E5',
          300: '#D4D4D4',
          400: '#A3A3A3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
        },
        // Legacy colors for compatibility
        success: {
          DEFAULT: '#00FF00',
          50: '#F0FFF0',
          500: '#00FF00',
        },
        warning: {
          DEFAULT: '#FFFF00',
          50: '#FFFFC7',
          500: '#FFFF00',
        },
        error: {
          DEFAULT: '#FF0000',
          50: '#FFF0F0',
          500: '#FF0000',
        },
      },
      fontFamily: {
        sans: ['Space Grotesk', 'Inter', 'sans-serif'],
        mono: ['IBM Plex Mono', 'JetBrains Mono', 'monospace'],
        display: ['Space Grotesk', 'sans-serif'],
      },
      borderRadius: {
        none: '0px',
        sm: '2px',
        DEFAULT: '4px',
        lg: '6px',
        // Remove excessive rounding for brutalist aesthetic
      },
      boxShadow: {
        // Remove soft shadows, replace with hard borders
        'brutal': '4px 4px 0px 0px #000000',
        'brutal-sm': '2px 2px 0px 0px #000000',
        'brutal-lg': '8px 8px 0px 0px #000000',
        'brutal-green': '4px 4px 0px 0px #00FF00',
        'brutal-yellow': '4px 4px 0px 0px #FFFF00',
        none: 'none',
      },
      animation: {
        // Sharp, instant transitions
        'snap-in': 'snapIn 0.1s ease-out',
        'glitch': 'glitch 0.3s ease-in-out',
      },
      keyframes: {
        snapIn: {
          '0%': { transform: 'scale(0.95)' },
          '100%': { transform: 'scale(1)' },
        },
        glitch: {
          '0%, 100%': { transform: 'translateX(0)' },
          '20%': { transform: 'translateX(-2px)' },
          '40%': { transform: 'translateX(2px)' },
          '60%': { transform: 'translateX(-1px)' },
          '80%': { transform: 'translateX(1px)' },
        },
      },
    },
  },
  plugins: [
    function({ addUtilities }: { addUtilities: any }) {
      const newUtilities = {
        '.text-brutal': {
          fontWeight: 'bold',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        },
        '.border-brutal': {
          borderWidth: '2px',
          borderColor: '#000000',
        },
        '.bg-grid': {
          backgroundImage: 'linear-gradient(90deg, #00000008 1px, transparent 1px), linear-gradient(180deg, #00000008 1px, transparent 1px)',
          backgroundSize: '20px 20px',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}
export default config