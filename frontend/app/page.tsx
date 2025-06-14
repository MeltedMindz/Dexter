import { Dashboard } from '@/components/Dashboard'
import { Navbar } from '@/components/Navbar'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-dark-900 dark:to-dark-800 transition-colors">
      <Navbar />
      <Dashboard />
    </main>
  )
}