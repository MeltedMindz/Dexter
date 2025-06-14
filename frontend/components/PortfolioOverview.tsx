'use client'

import { TrendingUp, DollarSign, Target, BarChart3, TrendingDown } from 'lucide-react'
import { useData } from '@/lib/data-context'

export function PortfolioOverview() {
  const { portfolioStats, isLoading } = useData()

  const formatCurrency = (amount: number) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)

  const formatPercentage = (value: number) => 
    `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`

  const isPositive = portfolioStats.change24h.percentage >= 0

  if (isLoading && portfolioStats.totalValue === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
          <BarChart3 className="w-6 h-6" />
          <span>Portfolio Overview</span>
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-gray-800 dark:bg-gray-900 rounded-xl p-6 border-2 border-white shadow-brutal animate-pulse">
              <div className="flex items-center justify-between mb-2">
                <div className="h-4 bg-slate-300 dark:bg-dark-600 rounded w-20"></div>
                <div className="w-4 h-4 bg-slate-300 dark:bg-dark-600 rounded"></div>
              </div>
              <div className="h-8 bg-slate-300 dark:bg-dark-600 rounded"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center space-x-2">
        <BarChart3 className="w-6 h-6" />
        <span>Portfolio Overview</span>
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Value */}
        <div className="bg-white dark:bg-black rounded-xl p-6 border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-900 dark:text-white">Total Value</span>
            <DollarSign className="w-4 h-4 text-slate-900 dark:text-white" />
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white font-mono">
            {formatCurrency(portfolioStats.totalValue)}
          </div>
          <div className={`flex items-center space-x-1 text-sm mt-2 ${
            isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
          }`}>
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span className="font-mono">
              {formatCurrency(portfolioStats.change24h.amount)} ({formatPercentage(portfolioStats.change24h.percentage)})
            </span>
          </div>
        </div>

        {/* Total Fees */}
        <div className="bg-white dark:bg-black rounded-xl p-6 border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-900 dark:text-white">Fees Earned</span>
            <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400 font-mono">
            {formatCurrency(portfolioStats.totalFees)}
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">
            All-time earnings
          </div>
        </div>

        {/* Total Profit */}
        <div className="bg-white dark:bg-black rounded-xl p-6 border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-900 dark:text-white">Total Profit</span>
            <DollarSign className="w-4 h-4 text-slate-900 dark:text-white" />
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white font-mono">
            {formatCurrency(portfolioStats.totalProfit)}
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">
            Including compounding
          </div>
        </div>

        {/* Active Positions & APR */}
        <div className="bg-white dark:bg-black rounded-xl p-6 border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-900 dark:text-white">Avg APR</span>
            <Target className="w-4 h-4 text-slate-900 dark:text-white" />
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400 font-mono">
            {portfolioStats.avgApr.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">
            {portfolioStats.activePositions} active positions
          </div>
        </div>
      </div>
    </div>
  )
}