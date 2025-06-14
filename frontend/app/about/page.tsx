import { Navbar } from '@/components/Navbar'
import { FlywheelExplainer } from '@/components/FlywheelExplainer'

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Navbar />
      <FlywheelExplainer />
    </main>
  )
}