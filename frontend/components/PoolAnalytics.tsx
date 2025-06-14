'use client'

import React, { useState } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Users, 
  Activity, 
  DollarSign,
  Clock,
  Target,
  Zap,
  AlertTriangle
} from 'lucide-react'

interface PoolAnalytics {
  address: string
  token0: { symbol: string; logoURI: string }
  token1: { symbol: string; logoURI: string }
  feeTier: number
  currentPrice: number
  priceChange24h: number
  volume24h: number
  volumeChange24h: number
  tvl: number
  tvlChange24h: number
  fees24h: number
  apr: number
  utilizationRate: number
  activePositions: number
  topLiquidityRange: { min: number; max: number; percentage: number }
  riskMetrics: {
    volatility: number
    impermanentLoss: number
    concentrationRisk: number
  }
  historicalData: {
    prices: number[]
    volumes: number[]
    timestamps: string[]
  }
}

// Mock analytics data
const POOL_ANALYTICS: PoolAnalytics[] = [
  {
    address: '0x1234...5678',
    token0: { symbol: 'ETH', logoURI: 'https://ethereum-optimism.github.io/data/ETH/logo.png' },
    token1: { symbol: 'USDC', logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png' },
    feeTier: 0.05,
    currentPrice: 2485.67,
    priceChange24h: 2.34,
    volume24h: 12500000,
    volumeChange24h: 15.2,
    tvl: 85000000,
    tvlChange24h: -2.1,
    fees24h: 62500,
    apr: 28.5,
    utilizationRate: 78.5,
    activePositions: 1247,
    topLiquidityRange: { min: 2200, max: 2800, percentage: 65.4 },
    riskMetrics: {
      volatility: 24.5,
      impermanentLoss: 1.8,
      concentrationRisk: 32.1
    },
    historicalData: {
      prices: [2400, 2420, 2435, 2450, 2465, 2475, 2485],
      volumes: [10000000, 11000000, 12500000, 9500000, 13000000, 11500000, 12500000],
      timestamps: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
  },
  {
    address: '0x2345...6789',
    token0: { symbol: 'USDC', logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png' },
    token1: { symbol: 'DAI', logoURI: 'https://ethereum-optimism.github.io/data/DAI/logo.png' },
    feeTier: 0.01,
    currentPrice: 1.0001,
    priceChange24h: 0.02,
    volume24h: 8500000,
    volumeChange24h: 8.7,
    tvl: 45000000,
    tvlChange24h: 1.2,
    fees24h: 8500,
    apr: 15.2,
    utilizationRate: 92.3,
    activePositions: 892,
    topLiquidityRange: { min: 0.999, max: 1.001, percentage: 84.2 },
    riskMetrics: {
      volatility: 2.1,
      impermanentLoss: 0.1,
      concentrationRisk: 15.3
    },
    historicalData: {
      prices: [1.0000, 1.0001, 0.9999, 1.0001, 1.0000, 1.0002, 1.0001],
      volumes: [7500000, 8000000, 8500000, 7800000, 8200000, 8100000, 8500000],
      timestamps: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
  }
]

export function PoolAnalytics() {
  const [selectedPool, setSelectedPool] = useState<PoolAnalytics>(POOL_ANALYTICS[0])
  const [timeframe, setTimeframe] = useState<'24h' | '7d' | '30d'>('24h')

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
          Pool Analytics
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Deep insights into Uniswap V4 pool performance and liquidity distribution
        </p>
      </div>

      {/* Pool Selector */}
      <div className="mb-8">
        <div className="flex space-x-4 overflow-x-auto">
          {POOL_ANALYTICS.map((pool) => (
            <button
              key={pool.address}
              onClick={() => setSelectedPool(pool)}
              className={`flex items-center space-x-3 px-6 py-4 rounded-xl border-2 transition-all ${
                selectedPool.address === pool.address
                  ? 'border-primary bg-primary/10 text-slate-900 dark:text-white'
                  : 'border-slate-200 dark:border-white/10 bg-white dark:bg-dark-700 text-slate-600 dark:text-slate-400 hover:border-primary/50'
              }`}
            >
              <div className="flex items-center -space-x-2">
                <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary-600 rounded-full border-2 border-white dark:border-dark-700"></div>
                <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full border-2 border-white dark:border-dark-700"></div>
              </div>
              <div className="text-left">
                <h3 className="font-semibold">
                  {pool.token0.symbol}/{pool.token1.symbol}
                </h3>
                <p className="text-sm opacity-75">
                  {pool.feeTier}% Fee
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Current Price"
          value={`$${selectedPool.currentPrice.toLocaleString()}`}
          change={selectedPool.priceChange24h}
          icon={DollarSign}
          color="blue"
        />
        <MetricCard
          title="24h Volume"
          value={`$${(selectedPool.volume24h / 1000000).toFixed(1)}M`}
          change={selectedPool.volumeChange24h}
          icon={BarChart3}
          color="green"
        />
        <MetricCard
          title="Total Value Locked"
          value={`$${(selectedPool.tvl / 1000000).toFixed(1)}M`}
          change={selectedPool.tvlChange24h}
          icon={Target}
          color="purple"
        />
        <MetricCard
          title="24h Fees"
          value={`$${selectedPool.fees24h.toLocaleString()}`}
          change={null}
          icon={Zap}
          color="yellow"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Charts and Performance */}
        <div className="lg:col-span-2 space-y-8">
          {/* Price Chart */}
          <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                Price Chart
              </h3>
              <div className="flex space-x-2">
                {(['24h', '7d', '30d'] as const).map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                      timeframe === tf
                        ? 'bg-primary text-white'
                        : 'bg-slate-100 dark:bg-dark-600 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-dark-500'
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
            <SimpleChart data={selectedPool.historicalData.prices} />
          </div>

          {/* Volume Chart */}
          <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
              Volume Trend
            </h3>
            <SimpleChart data={selectedPool.historicalData.volumes} color="green" />
          </div>

          {/* Performance Metrics */}
          <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
              Performance Metrics
            </h3>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Current APR</p>
                <p className="text-2xl font-bold text-green-600 mono-numbers">
                  {selectedPool.apr.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Utilization Rate</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                  {selectedPool.utilizationRate.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Active Positions</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                  {selectedPool.activePositions.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Fee Efficiency</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                  {(selectedPool.fees24h / selectedPool.volume24h * 10000).toFixed(2)} bps
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Risk Analysis and Liquidity Distribution */}
        <div className="space-y-8">
          {/* Risk Metrics */}
          <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
              Risk Analysis
            </h3>
            <div className="space-y-4">
              <RiskMetric
                label="Volatility (30d)"
                value={selectedPool.riskMetrics.volatility}
                unit="%"
                threshold={25}
                type="volatility"
              />
              <RiskMetric
                label="Impermanent Loss"
                value={selectedPool.riskMetrics.impermanentLoss}
                unit="%"
                threshold={5}
                type="loss"
              />
              <RiskMetric
                label="Concentration Risk"
                value={selectedPool.riskMetrics.concentrationRisk}
                unit="%"
                threshold={40}
                type="concentration"
              />
            </div>
          </div>

          {/* Liquidity Distribution */}
          <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
              Liquidity Distribution
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-600 dark:text-slate-400">Top Range</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {selectedPool.topLiquidityRange.percentage}%
                  </span>
                </div>
                <div className="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-primary to-primary-600 rounded-full transition-all duration-1000"
                    style={{ width: `${selectedPool.topLiquidityRange.percentage}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>${selectedPool.topLiquidityRange.min.toLocaleString()}</span>
                  <span>${selectedPool.topLiquidityRange.max.toLocaleString()}</span>
                </div>
              </div>

              <div className="border-t border-slate-200 dark:border-white/10 pt-4">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">In-Range Liquidity</span>
                    <span className="text-sm font-medium text-green-600">
                      {selectedPool.utilizationRate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Out-of-Range</span>
                    <span className="text-sm font-medium text-yellow-600">
                      {(100 - selectedPool.utilizationRate).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Pool Health Score */}
          <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
              Pool Health Score
            </h3>
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-4">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 36 36">
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="2"
                  />
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    strokeDasharray="85, 100"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-slate-900 dark:text-white">85</span>
                </div>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Excellent health based on volume, liquidity efficiency, and risk metrics
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  color 
}: { 
  title: string
  value: string
  change: number | null
  icon: React.ElementType
  color: string 
}) {
  const colorClasses = {
    blue: 'bg-blue-100 dark:bg-blue-900/20 text-blue-600',
    green: 'bg-green-100 dark:bg-green-900/20 text-green-600',
    purple: 'bg-purple-100 dark:bg-purple-900/20 text-purple-600',
    yellow: 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-600'
  }

  return (
    <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
      <div className="flex items-center space-x-3 mb-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colorClasses[color as keyof typeof colorClasses]}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div>
          <p className="text-sm text-slate-600 dark:text-slate-400">{title}</p>
          <p className="text-xl font-bold text-slate-900 dark:text-white mono-numbers">
            {value}
          </p>
        </div>
      </div>
      {change !== null && (
        <div className={`flex items-center space-x-1 text-sm ${
          change >= 0 ? 'text-green-600' : 'text-red-600'
        }`}>
          {change >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span className="mono-numbers">
            {change >= 0 ? '+' : ''}{change.toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  )
}

function RiskMetric({ 
  label, 
  value, 
  unit, 
  threshold, 
  type 
}: { 
  label: string
  value: number
  unit: string
  threshold: number
  type: 'volatility' | 'loss' | 'concentration'
}) {
  const isHighRisk = value > threshold
  const percentage = Math.min((value / threshold) * 100, 100)

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-slate-600 dark:text-slate-400">{label}</span>
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-slate-900 dark:text-white mono-numbers">
            {value.toFixed(1)}{unit}
          </span>
          {isHighRisk && <AlertTriangle className="w-4 h-4 text-yellow-500" />}
        </div>
      </div>
      <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-1000 ${
            percentage > 80 ? 'bg-red-500' : 
            percentage > 60 ? 'bg-yellow-500' : 
            'bg-green-500'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

function SimpleChart({ data, color = 'blue' }: { data: number[]; color?: string }) {
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1

  return (
    <div className="h-32 flex items-end space-x-1">
      {data.map((value, index) => {
        const height = ((value - min) / range) * 100
        return (
          <div
            key={index}
            className={`flex-1 rounded-t transition-all duration-500 ${
              color === 'green' ? 'bg-green-500' : 'bg-blue-500'
            }`}
            style={{ height: `${Math.max(height, 5)}%` }}
          />
        )
      })}
    </div>
  )
}