'use client'

import { useState } from 'react'
import { useAccount } from 'wagmi'
import { PortfolioOverview } from './PortfolioOverview'
import { EnhancedPortfolioOverview } from './EnhancedPortfolioOverview'
import { QuickActions } from './QuickActions'
import { PositionsList } from './PositionsList'
import { ConnectPrompt } from './ConnectPrompt'
import { Analytics } from './Analytics'
import { HistoricalChart, generateHistoricalData } from './HistoricalChart'
import { TrendingTokensTable } from './TrendingTokensTable'
import { BarChart3, ArrowRight } from 'lucide-react'

export function Dashboard() {
  const { isConnected } = useAccount()
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [useEnhancedPortfolio, setUseEnhancedPortfolio] = useState(true)

  if (!isConnected) {
    return <ConnectPrompt />
  }

  if (showAnalytics) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <button
            onClick={() => setShowAnalytics(false)}
            className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
          >
            <ArrowRight className="w-4 h-4 rotate-180" />
            <span>Back to Dashboard</span>
          </button>
        </div>
        <Analytics />
      </div>
    )
  }

  const historicalData = generateHistoricalData(30)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-8">
        {/* Portfolio Overview */}
        {useEnhancedPortfolio ? <EnhancedPortfolioOverview /> : <PortfolioOverview />}
        
        {/* Quick Actions */}
        <QuickActions />
        
        {/* Performance Chart Preview */}
        <div className="bg-white dark:bg-black rounded-xl border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                Portfolio Performance
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                30-day historical data
              </p>
            </div>
            <button
              onClick={() => setShowAnalytics(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-green-400 text-black rounded-lg hover:bg-green-300 border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200"
            >
              <BarChart3 className="w-4 h-4" />
              <span>View Analytics</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
          
          <HistoricalChart 
            data={historicalData}
            title=""
            timeframe="30D"
            className="border-0 p-0 bg-transparent"
          />
        </div>
        
        {/* Trending Tokens Table */}
        <TrendingTokensTable />
        
        {/* Positions List */}
        <PositionsList />
      </div>
    </div>
  )
}