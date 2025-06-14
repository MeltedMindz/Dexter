'use client'

import { TrendingUp, DollarSign, Target, BarChart3 } from 'lucide-react'

export function PortfolioOverview() {
  // Mock data - in real app this would come from API/contracts
  const portfolioData = {
    totalValue: 127450,
    dailyYield: { amount: 420, percentage: 1.2 },
    totalEarned: 12540,
    successRate: 94.2
  }

  const formatCurrency = (amount: number) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)

  const formatPercentage = (value: number) => 
    `${value.toFixed(1)}%`

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-slate-900 flex items-center space-x-2">
        <BarChart3 className="w-6 h-6" />
        <span>Portfolio Overview</span>
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Value */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Total Value</span>
            <DollarSign className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mono-numbers">
            {formatCurrency(portfolioData.totalValue)}
          </div>
        </div>

        {/* Daily Yield */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Daily Yield</span>
            <TrendingUp className="w-4 h-4 text-success" />
          </div>
          <div className="space-y-1">
            <div className="text-2xl font-bold text-success mono-numbers">
              +{formatCurrency(portfolioData.dailyYield.amount)}
            </div>
            <div className="text-sm text-success mono-numbers">
              ({formatPercentage(portfolioData.dailyYield.percentage)})
            </div>
          </div>
        </div>

        {/* Total Earned */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Total Earned</span>
            <DollarSign className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mono-numbers">
            {formatCurrency(portfolioData.totalEarned)}
          </div>
        </div>

        {/* Success Rate */}
        <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">Success Rate</span>
            <Target className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-success mono-numbers">
            {formatPercentage(portfolioData.successRate)}
          </div>
        </div>
      </div>
    </div>
  )
}