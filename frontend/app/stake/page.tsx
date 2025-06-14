import { Navbar } from '@/components/Navbar'
import { StakingDashboard } from '@/components/StakingDashboard'

export default function StakePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Navbar />
      <StakingDashboard />
    </main>
  )
}