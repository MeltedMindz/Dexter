'use client'

import { useAccount } from 'wagmi'
import { PortfolioOverview } from './PortfolioOverview'
import { QuickActions } from './QuickActions'
import { PositionsList } from './PositionsList'
import { ConnectPrompt } from './ConnectPrompt'

export function Dashboard() {
  const { isConnected } = useAccount()

  if (!isConnected) {
    return <ConnectPrompt />
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-8">
        {/* Portfolio Overview */}
        <PortfolioOverview />
        
        {/* Quick Actions */}
        <QuickActions />
        
        {/* Positions List */}
        <PositionsList />
      </div>
    </div>
  )
}