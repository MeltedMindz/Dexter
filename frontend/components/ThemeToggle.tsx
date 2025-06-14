'use client'

import { Moon, Sun } from 'lucide-react'
import { useTheme } from '@/lib/theme-context'

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      className="relative w-10 h-10 rounded-lg bg-white/10 dark:bg-white/5 border border-slate-200 dark:border-white/10 hover:bg-slate-100 dark:hover:bg-white/10 transition-colors flex items-center justify-center group"
      aria-label="Toggle theme"
    >
      <Sun className="w-5 h-5 text-slate-700 dark:text-white transition-all scale-100 rotate-0 dark:scale-0 dark:-rotate-90" />
      <Moon className="absolute w-5 h-5 text-slate-700 dark:text-white transition-all scale-0 rotate-90 dark:scale-100 dark:rotate-0" />
      <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/20 to-success/20 opacity-0 group-hover:opacity-100 transition-opacity" />
    </button>
  )
}