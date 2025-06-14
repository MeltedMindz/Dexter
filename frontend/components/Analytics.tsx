'use client'

import { useState } from 'react'
import { HistoricalChart, generateHistoricalData } from './HistoricalChart'
import { MiniChart, generateSampleData } from './MiniChart'
import { TrendingUp, DollarSign, Activity, Percent, ArrowUpRight, ArrowDownRight } from 'lucide-react'

interface AnalyticsMetric {
  id: string
  title: string
  value: string
  change: number
  changeType: 'increase' | 'decrease'
  icon: React.ComponentType<any>
  color: string
  chartData: number[]
}

const metricsData: AnalyticsMetric[] = [
  {
    id: 'total-value',
    title: 'Total Portfolio Value',
    value: '$47,892.34',
    change: 12.34,
    changeType: 'increase',
    icon: DollarSign,
    color: '#6366F1',
    chartData: generateSampleData(30, 'up')
  },
  {
    id: 'fees-earned',
    title: 'Fees Earned (30D)',
    value: '$2,847.12',
    change: 8.91,
    changeType: 'increase',
    icon: TrendingUp,
    color: '#10B981',
    chartData: generateSampleData(30, 'up')
  },
  {
    id: 'volume',
    title: 'Volume (24H)',
    value: '$1.2M',
    change: -5.67,
    changeType: 'decrease',
    icon: Activity,
    color: '#F59E0B',
    chartData: generateSampleData(30, 'volatile')
  },
  {
    id: 'avg-apr',
    title: 'Average APR',
    value: '24.8%',
    change: 2.1,
    changeType: 'increase',
    icon: Percent,
    color: '#EF4444',
    chartData: generateSampleData(30, 'up')
  }
]

export function Analytics() {
  const [selectedPosition, setSelectedPosition] = useState('all')
  const historicalData = generateHistoricalData(90)

  const positions = [
    { id: 'all', name: 'All Positions' },
    { id: 'eth-usdc', name: 'ETH/USDC 0.05%' },
    { id: 'btc-eth', name: 'BTC/ETH 0.3%' },
    { id: 'usdc-usdt', name: 'USDC/USDT 0.01%' }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">Analytics</h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Detailed performance metrics and historical data
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
            Position:
          </label>
          <select
            value={selectedPosition}
            onChange={(e) => setSelectedPosition(e.target.value)}
            className="px-3 py-2 bg-white dark:bg-dark-700 border border-slate-200 dark:border-white/10 rounded-lg text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {positions.map(pos => (
              <option key={pos.id} value={pos.id}>{pos.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metricsData.map((metric) => {
          const Icon = metric.icon
          const isPositive = metric.changeType === 'increase'
          
          return (
            <div
              key={metric.id}
              className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 hover:shadow-lg dark:hover:shadow-xl transition-all duration-200"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center`} style={{ backgroundColor: `${metric.color}20` }}>
                  <Icon className="w-6 h-6" style={{ color: metric.color }} />
                </div>
                <div className="text-right">
                  <MiniChart 
                    data={metric.chartData}
                    width={60}
                    height={30}
                    color={metric.color}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  {metric.title}
                </h3>
                <div className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                  {metric.value}
                </div>
                <div className={`flex items-center space-x-1 text-sm font-medium ${
                  isPositive ? 'text-success' : 'text-error'
                }`}>
                  {isPositive ? (
                    <ArrowUpRight className="w-4 h-4" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4" />
                  )}
                  <span className="mono-numbers">
                    {isPositive ? '+' : ''}{metric.change.toFixed(2)}%
                  </span>
                  <span className="text-slate-500 dark:text-slate-400 text-xs">
                    vs last month
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Historical Performance Chart */}
      <HistoricalChart 
        data={historicalData}
        title="Historical Performance"
        className="col-span-full"
      />

      {/* Additional Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Volume Chart */}
        <HistoricalChart 
          data={historicalData}
          title="Volume Trends"
          metric="volume"
          timeframe="30D"
        />
        
        {/* Fees Chart */}
        <HistoricalChart 
          data={historicalData}
          title="Fees Earned"
          metric="fees"
          timeframe="30D"
        />
      </div>

      {/* Performance Breakdown */}
      <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6">
        <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
          Performance Breakdown by Position
        </h3>
        
        <div className="space-y-4">
          {[
            { pair: 'ETH/USDC', fee: '0.05%', value: '$18,234.56', fees: '$1,234.78', apr: '28.4%', change: 15.2 },
            { pair: 'BTC/ETH', fee: '0.30%', value: '$15,789.23', fees: '$892.45', apr: '22.1%', change: 8.7 },
            { pair: 'USDC/USDT', fee: '0.01%', value: '$13,868.55', fees: '$719.89', apr: '18.9%', change: -2.3 }
          ].map((position, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-4 bg-slate-50 dark:bg-dark-600 rounded-lg hover:bg-slate-100 dark:hover:bg-dark-500 transition-colors"
            >
              <div className="flex items-center space-x-4">
                <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">📊</span>
                </div>
                <div>
                  <div className="font-semibold text-slate-900 dark:text-white">
                    {position.pair} Pool
                  </div>
                  <div className="text-sm text-slate-500 dark:text-slate-400">
                    Fee Tier: {position.fee}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-8">
                <div className="text-right">
                  <div className="text-sm text-slate-500 dark:text-slate-400">Position Value</div>
                  <div className="font-semibold text-slate-900 dark:text-white mono-numbers">
                    {position.value}
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm text-slate-500 dark:text-slate-400">Fees Earned</div>
                  <div className="font-semibold text-slate-900 dark:text-white mono-numbers">
                    {position.fees}
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm text-slate-500 dark:text-slate-400">APR</div>
                  <div className="font-semibold text-success mono-numbers">
                    {position.apr}
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm text-slate-500 dark:text-slate-400">30D Change</div>
                  <div className={`font-semibold mono-numbers ${
                    position.change >= 0 ? 'text-success' : 'text-error'
                  }`}>
                    {position.change >= 0 ? '+' : ''}{position.change.toFixed(1)}%
                  </div>
                </div>
                
                <div className="w-16">
                  <MiniChart 
                    data={generateSampleData(30, position.change >= 0 ? 'up' : 'down')}
                    width={60}
                    height={30}
                    color={position.change >= 0 ? '#10B981' : '#EF4444'}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}