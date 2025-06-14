'use client'

import { useState } from 'react'
import { Coins, ArrowUpRight, ArrowDownLeft } from 'lucide-react'

export function StakeInterface() {
  const [stakeAmount, setStakeAmount] = useState('')
  const [activeTab, setActiveTab] = useState<'stake' | 'unstake'>('stake')
  
  // Mock data - in real app this would come from API/contracts
  const userBalance = 50000
  const stakedBalance = 125000

  const formatNumber = (value: number) => 
    new Intl.NumberFormat('en-US').format(value)

  const percentageButtons = [25, 50, 75, 100]

  const handlePercentageClick = (percentage: number) => {
    const balance = activeTab === 'stake' ? userBalance : stakedBalance
    const amount = (balance * percentage) / 100
    setStakeAmount(amount.toString())
  }

  const handleMaxClick = () => {
    const balance = activeTab === 'stake' ? userBalance : stakedBalance
    setStakeAmount(balance.toString())
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center space-x-2">
        <span>⚡ Stake More DEX</span>
      </h2>
      
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        {/* Tab Selector */}
        <div className="flex space-x-1 mb-6 bg-slate-100 rounded-lg p-1">
          <button
            onClick={() => setActiveTab('stake')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
              activeTab === 'stake'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Stake DEX
          </button>
          <button
            onClick={() => setActiveTab('unstake')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
              activeTab === 'unstake'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Unstake DEX
          </button>
        </div>

        {/* Amount Input */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Amount
            </label>
            <div className="relative">
              <input
                type="number"
                value={stakeAmount}
                onChange={(e) => setStakeAmount(e.target.value)}
                placeholder="0"
                className="w-full px-4 py-3 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-lg font-mono"
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
                <span className="text-slate-600 font-medium">DEX</span>
              </div>
            </div>
            <div className="flex justify-between items-center mt-2 text-sm text-slate-600">
              <span>
                {activeTab === 'stake' ? 'Balance' : 'Staked'}: {formatNumber(activeTab === 'stake' ? userBalance : stakedBalance)} DEX
              </span>
            </div>
          </div>

          {/* Percentage Buttons */}
          <div className="flex space-x-2">
            {percentageButtons.map((percentage) => (
              <button
                key={percentage}
                onClick={() => handlePercentageClick(percentage)}
                className="flex-1 py-2 px-3 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm font-medium transition-colors"
              >
                {percentage}%
              </button>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-4">
            <button 
              className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2 ${
                activeTab === 'stake'
                  ? 'bg-primary text-white hover:bg-primary/90'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              <Coins className="w-5 h-5" />
              <span>{activeTab === 'stake' ? '💎 Stake DEX' : '📤 Unstake'}</span>
            </button>
            
            <button className="px-6 py-3 bg-slate-100 text-slate-700 rounded-lg font-medium hover:bg-slate-200 transition-colors flex items-center justify-center space-x-2">
              <ArrowDownLeft className="w-5 h-5" />
              <span>📤 Unstake</span>
            </button>
            
            <button className="px-6 py-3 bg-success text-white rounded-lg font-medium hover:bg-success/90 transition-colors flex items-center justify-center space-x-2">
              <ArrowUpRight className="w-5 h-5" />
              <span>💰 Claim WETH</span>
            </button>
          </div>

          {/* Info Box */}
          <div className="mt-6 p-4 bg-primary/5 border border-primary/20 rounded-lg">
            <h4 className="font-medium text-primary mb-2">How Staking Works</h4>
            <ul className="text-sm text-slate-600 space-y-1">
              <li>• Stake $DEX tokens to earn protocol revenue</li>
              <li>• All fees converted to WETH for consistent returns</li>
              <li>• Rewards distributed pro-rata based on your stake %</li>
              <li>• No lock-up period - unstake anytime</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}