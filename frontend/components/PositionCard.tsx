'use client'

import { Pause, Download, Settings, BarChart3, Play, Clock } from 'lucide-react'

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

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">🔷</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">{position.pair} Pool</h3>
            <span className="text-sm text-slate-500">Last Compound: 2h ago</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-slate-500">Next Check</div>
          <div className="text-sm font-medium text-slate-700">{position.nextCheck}</div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-4 space-y-4">
        {/* Position Value & Change */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-600">Position Value:</span>
            <span className="text-xl font-bold text-slate-900 mono-numbers">
              {formatCurrency(position.value)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-600">24h Change:</span>
            <span className="text-sm font-medium text-success mono-numbers">
              +{formatCurrency(position.change24h.amount)} ({formatPercentage(position.change24h.percentage)})
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-600">Range:</span>
            <span className="text-sm text-slate-700 mono-numbers">
              {formatRange(position.range.min, position.range.max, position.range.current)}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-600">Fee Tier:</span>
            <span className="text-sm font-medium text-slate-700">
              {position.feeTier}%
            </span>
          </div>
        </div>

        {/* Fees Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-slate-50 rounded-lg">
          <div className="text-center">
            <div className="text-xs text-slate-600 mb-1">Fees Earned</div>
            <div className="text-sm font-semibold text-slate-900 mono-numbers">
              {formatCurrency(position.fees.earned)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-600 mb-1">Compounded</div>
            <div className="text-sm font-semibold text-success mono-numbers">
              {formatCurrency(position.fees.compounded)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-600 mb-1">Protocol Fee</div>
            <div className="text-sm font-semibold text-slate-700 mono-numbers">
              {formatCurrency(position.fees.protocolFee)} (8%)
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-600 mb-1">Total Profit</div>
            <div className="text-sm font-semibold text-success mono-numbers">
              {formatCurrency(position.fees.totalProfit)}
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
          <div className="flex items-center space-x-2">
            {position.isActive ? (
              <>
                <div className="w-2 h-2 bg-success rounded-full"></div>
                <span className="text-sm font-medium text-success">Auto-Compound Active</span>
              </>
            ) : (
              <>
                <div className="w-2 h-2 bg-warning rounded-full"></div>
                <span className="text-sm font-medium text-warning">Paused</span>
              </>
            )}
          </div>
          <div className="flex items-center space-x-1 text-xs text-slate-500">
            <Clock className="w-3 h-3" />
            <span>Next Check: {position.nextCheck}</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2 pt-2">
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors">
            {position.isActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{position.isActive ? 'Pause' : 'Resume'}</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors">
            <Download className="w-4 h-4" />
            <span>Withdraw</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors">
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors">
            <BarChart3 className="w-4 h-4" />
            <span>Analytics</span>
          </button>
        </div>
      </div>
    </div>
  )
}