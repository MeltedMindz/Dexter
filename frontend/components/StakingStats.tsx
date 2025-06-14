'use client'

import { TrendingUp, Users, Coins, Percent } from 'lucide-react'

export function StakingStats() {
  // Mock data - in real app this would come from API/contracts
  const stakingData = {
    stakedDEX: 125000,
    yourShare: 2.5,
    wethEarned: 1.47,
    estimatedAPY: 18.7
  }

  const formatNumber = (value: number, decimals = 0) => 
    new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(value)

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center space-x-2">
        <span>🏆 Your Staking Stats</span>
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Staked DEX */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Staked DEX</span>
            <Coins className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mono-numbers">
            {formatNumber(stakingData.stakedDEX)}
          </div>
        </div>

        {/* Your Share */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Your Share</span>
            <Users className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-primary mono-numbers">
            {stakingData.yourShare}%
          </div>
        </div>

        {/* WETH Earned */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">WETH Earned</span>
            <Coins className="w-4 h-4 text-success" />
          </div>
          <div className="text-2xl font-bold text-success mono-numbers">
            {formatNumber(stakingData.wethEarned, 2)} WETH
          </div>
        </div>

        {/* Estimated APY */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Est. APY</span>
            <TrendingUp className="w-4 h-4 text-success" />
          </div>
          <div className="text-2xl font-bold text-success mono-numbers">
            {formatNumber(stakingData.estimatedAPY, 1)}%
          </div>
        </div>
      </div>
    </div>
  )
}