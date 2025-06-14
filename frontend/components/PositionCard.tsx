'use client'

import { Pause, Download, Settings, BarChart3, Play, Clock, TrendingUp } from 'lucide-react'
import { MiniChart, generateSampleData } from './MiniChart'

interface Position {
  id: number
  pair: string
  value: number
  change24h: { amount: number; percentage: number }
  range: { min: number; max: number; current: number }
  feeTier: number
  fees: {
    earned: number
    compounded: number
    protocolFee: number
    totalProfit: number
  }
  isActive: boolean
  nextCheck: string
  volume24h?: number
  apr?: number
  chartData?: number[]
}

interface PositionCardProps {
  position: Position
}

export function PositionCard({ position }: PositionCardProps) {
  const formatCurrency = (amount: number, decimals = 2) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(amount)

  const formatPercentage = (value: number) => 
    `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`

  const formatRange = (min: number, max: number, current: number) => {
    if (min < 1) {
      return `${min.toFixed(6)} - ${max.toFixed(6)} (Current: ${current.toFixed(6)})`
    }
    return `${formatCurrency(min, 0)} - ${formatCurrency(max, 0)} (Current: ${formatCurrency(current, 0)})`
  }

  // Generate sample chart data if not provided
  const chartData = position.chartData || generateSampleData(30, 'up')
  const volume24h = position.volume24h || Math.random() * 1000000
  const apr = position.apr || Math.random() * 50

  return (
    <div className="bg-white dark:bg-black rounded-xl border-brutal shadow-brutal hover:shadow-brutal-lg transition-all duration-200 overflow-hidden group">
      {/* Header */}
      <div className="px-6 py-4 border-b-2 border-black dark:border-white flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="relative w-10 h-10 bg-gradient-to-br from-primary to-primary-600 rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform">
            <span className="text-white font-bold text-sm">🔷</span>
            <div className="absolute inset-0 rounded-lg bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">{position.pair} Pool</h3>
            <span className="text-sm text-slate-500 dark:text-slate-400">Last Compound: 2h ago</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-slate-500 dark:text-slate-400">Next Check</div>
          <div className="text-sm font-medium text-slate-700 dark:text-slate-300">{position.nextCheck}</div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-4 space-y-4">
        {/* Position Value & Chart */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-sm text-slate-600 dark:text-slate-400">Position Value</div>
            <div className="text-2xl font-bold text-slate-900 dark:text-white font-mono">
              {formatCurrency(position.value)}
            </div>
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-green-600 dark:text-green-400 font-mono">
                +{formatCurrency(position.change24h.amount)} ({formatPercentage(position.change24h.percentage)})
              </span>
            </div>
          </div>
          <div className="text-right">
            <MiniChart 
              data={chartData} 
              width={120} 
              height={60}
              color={position.change24h.percentage >= 0 ? '#10B981' : '#EF4444'}
            />
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Volume 24h</div>
            <div className="text-lg font-bold text-slate-900 dark:text-white font-mono">
              {volume24h >= 1000000 ? `$${(volume24h / 1000000).toFixed(1)}M` : `$${(volume24h / 1000).toFixed(0)}K`}
            </div>
          </div>
          <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">APR</div>
            <div className="text-lg font-bold text-green-600 dark:text-green-400 font-mono">
              {apr.toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Position Details */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-600 dark:text-slate-400">Range:</span>
            <span className="text-sm text-slate-700 dark:text-slate-300 font-mono">
              {formatRange(position.range.min, position.range.max, position.range.current)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-600 dark:text-slate-400">Fee Tier:</span>
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              {position.feeTier}%
            </span>
          </div>
        </div>

        {/* Fees Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 p-4 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-center">
            <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">Fees Earned</div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white font-mono">
              {formatCurrency(position.fees.earned)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">Compounded</div>
            <div className="text-sm font-semibold text-green-600 dark:text-green-400 font-mono">
              {formatCurrency(position.fees.compounded)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">Protocol Fee</div>
            <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 font-mono">
              {formatCurrency(position.fees.protocolFee)} (8%)
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">Total Profit</div>
            <div className="text-sm font-semibold text-green-600 dark:text-green-400 font-mono">
              {formatCurrency(position.fees.totalProfit)}
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center justify-between p-3 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-2">
            {position.isActive ? (
              <>
                <div className="w-2 h-2 bg-green-600 dark:bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">Auto-Compound Active</span>
              </>
            ) : (
              <>
                <div className="w-2 h-2 bg-yellow-600 dark:bg-yellow-400 rounded-full"></div>
                <span className="text-sm font-medium text-yellow-600 dark:text-yellow-400">Paused</span>
              </>
            )}
          </div>
          <div className="flex items-center space-x-1 text-xs text-slate-500 dark:text-slate-400">
            <Clock className="w-3 h-3" />
            <span>Next Check: {position.nextCheck}</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2 pt-2">
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-900 dark:text-white bg-white dark:bg-black hover:bg-slate-50 dark:hover:bg-slate-900 rounded-lg border-brutal shadow-brutal-sm hover:shadow-brutal transition-all duration-200">
            {position.isActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{position.isActive ? 'Pause' : 'Resume'}</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-900 dark:text-white bg-white dark:bg-black hover:bg-slate-50 dark:hover:bg-slate-900 rounded-lg border-brutal shadow-brutal-sm hover:shadow-brutal transition-all duration-200">
            <Download className="w-4 h-4" />
            <span>Withdraw</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-900 dark:text-white bg-white dark:bg-black hover:bg-slate-50 dark:hover:bg-slate-900 rounded-lg border-brutal shadow-brutal-sm hover:shadow-brutal transition-all duration-200">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-900 dark:text-white bg-white dark:bg-black hover:bg-slate-50 dark:hover:bg-slate-900 rounded-lg border-brutal shadow-brutal-sm hover:shadow-brutal transition-all duration-200">
            <BarChart3 className="w-4 h-4" />
            <span>Analytics</span>
          </button>
        </div>
      </div>
    </div>
  )
}