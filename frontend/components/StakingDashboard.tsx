'use client'

import { useAccount } from 'wagmi'
import { ConnectPrompt } from './ConnectPrompt'
import { StakingStats } from './StakingStats'
import { RevenuePool } from './RevenuePool'
import { StakeInterface } from './StakeInterface'

export function StakingDashboard() {
  const { isConnected } = useAccount()

  if (!isConnected) {
    return <ConnectPrompt />
  }

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-3xl font-bold text-slate-900 flex items-center justify-center space-x-3">
            <span>💎</span>
            <span>Stake $DEX Tokens - Earn Protocol Revenue</span>
          </h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Stake your $DEX tokens to earn WETH rewards from protocol fees. All fees are converted to WETH for consistent returns.
          </p>
        </div>

        {/* Staking Stats */}
        <StakingStats />
        
        {/* Revenue Pool */}
        <RevenuePool />
        
        {/* Stake Interface */}
        <StakeInterface />
      </div>
    </div>
  )
}