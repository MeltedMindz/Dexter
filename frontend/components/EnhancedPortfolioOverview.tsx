'use client'

import { useState } from 'react'
import { TrendingUp, DollarSign, Target, BarChart3, TrendingDown, RefreshCw, Eye, EyeOff } from 'lucide-react'
import { useData } from '@/lib/data-context'
import { useEnhancedWalletHoldings } from '@/lib/hooks/useEnhancedWalletHoldings'
import { useAccount } from 'wagmi'

interface NetworkToggleProps {
  network: 'base' | 'mainnet'
  onNetworkChange: (network: 'base' | 'mainnet') => void
}

function NetworkToggle({ network, onNetworkChange }: NetworkToggleProps) {
  return (
    <div className="flex items-center space-x-2 bg-slate-100 dark:bg-dark-600 rounded-lg p-1">
      <button
        onClick={() => onNetworkChange('base')}
        className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
          network === 'base'
            ? 'bg-primary text-white'
            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
        }`}
      >
        Base
      </button>
      <button
        onClick={() => onNetworkChange('mainnet')}
        className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
          network === 'mainnet'
            ? 'bg-primary text-white'
            : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
        }`}
      >
        Mainnet
      </button>
    </div>
  )
}

export function EnhancedPortfolioOverview() {
  const { portfolioStats, isLoading: portfolioLoading } = useData()
  const { isConnected } = useAccount()
  const [selectedNetwork, setSelectedNetwork] = useState<'base' | 'mainnet'>('base')
  const [showRealData, setShowRealData] = useState(false)
  
  const { 
    tokens, 
    totalValue, 
    totalChange24h, 
    isLoading: walletLoading, 
    error: walletError,
    lastUpdated
  } = useEnhancedWalletHoldings(selectedNetwork)

  const formatCurrency = (amount: number) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)

  const formatPercentage = (value: number) => 
    `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`

  // Use real wallet data or mock portfolio data
  const displayData = showRealData ? {
    totalValue,
    change24h: {
      amount: totalChange24h,
      percentage: totalValue > 0 ? (totalChange24h / totalValue) * 100 : 0
    },
    totalFees: portfolioStats.totalFees,
    totalProfit: portfolioStats.totalProfit,
    avgApr: portfolioStats.avgApr,
    activePositions: tokens.length
  } : portfolioStats

  const isPositive = displayData.change24h.percentage >= 0
  const isLoading = portfolioLoading || (showRealData && walletLoading)

  if (!isConnected) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <BarChart3 className="w-6 h-6" />
          <span>Portfolio Overview</span>
        </h2>
        <div className="bg-slate-100 dark:bg-dark-700 rounded-xl p-8 text-center">
          <p className="text-slate-600 dark:text-slate-400">
            Connect your wallet to view portfolio overview
          </p>
        </div>
      </div>
    )
  }

  if (isLoading && displayData.totalValue === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center space-x-2">
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
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <BarChart3 className="w-6 h-6" />
          <span>Portfolio Overview</span>
        </h2>
        
        <div className="flex items-center space-x-3">
          {/* Data Source Toggle */}
          <button
            onClick={() => setShowRealData(!showRealData)}
            className="flex items-center space-x-2 px-3 py-2 bg-slate-100 dark:bg-dark-600 rounded-lg hover:bg-slate-200 dark:hover:bg-dark-500 transition-colors"
            title={showRealData ? 'Switch to mock data' : 'Switch to real wallet data'}
          >
            {showRealData ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            <span className="text-sm">{showRealData ? 'Real Data' : 'Mock Data'}</span>
          </button>
          
          {/* Network Toggle - only show when using real data */}
          {showRealData && (
            <NetworkToggle network={selectedNetwork} onNetworkChange={setSelectedNetwork} />
          )}
          
          {/* Last Updated Indicator */}
          {showRealData && lastUpdated && (
            <div className="flex items-center space-x-1 text-xs text-slate-500 dark:text-slate-400">
              <RefreshCw className="w-3 h-3" />
              <span>{lastUpdated.toLocaleTimeString()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {showRealData && walletError && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-600 dark:text-red-400 text-sm">
            Error loading wallet data: {walletError}
          </p>
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Value */}
        <div className="bg-white dark:bg-black rounded-xl p-6 border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-900 dark:text-white">
              {showRealData ? 'Wallet Value' : 'Total Value'}
            </span>
            <DollarSign className="w-4 h-4 text-slate-900 dark:text-white" />
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white font-mono">
            {formatCurrency(displayData.totalValue)}
          </div>
          <div className={`flex items-center space-x-1 text-sm mt-2 ${
            isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
          }`}>
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            <span className="font-mono">
              {formatCurrency(displayData.change24h.amount)} ({formatPercentage(displayData.change24h.percentage)})
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
            {formatCurrency(displayData.totalFees)}
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
            {formatCurrency(displayData.totalProfit)}
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
            {displayData.avgApr.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">
            {displayData.activePositions} {showRealData ? 'tokens' : 'active positions'}
          </div>
        </div>
      </div>

      {/* Token Holdings Preview - only show when using real data */}
      {showRealData && tokens.length > 0 && (
        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Top Holdings on {selectedNetwork === 'base' ? 'Base' : 'Mainnet'}
          </h3>
          <div className="space-y-3">
            {tokens.slice(0, 5).map((token) => (
              <div key={token.address} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {token.logoURI && (
                    <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-xs font-bold">
                      {token.symbol.slice(0, 2)}
                    </div>
                  )}
                  <div>
                    <div className="font-medium text-slate-900 dark:text-white">
                      {token.balanceFormatted} {token.symbol}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      {token.name}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-slate-900 dark:text-white">
                    {formatCurrency(token.usdValue || 0)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}