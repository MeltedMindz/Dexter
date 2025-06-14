'use client'

import { useState, useMemo } from 'react'
import { Calendar, TrendingUp, TrendingDown, BarChart3, LineChart, Activity } from 'lucide-react'

interface DataPoint {
  timestamp: number
  value: number
  volume?: number
  fees?: number
}

interface HistoricalChartProps {
  data: DataPoint[]
  title?: string
  metric?: 'value' | 'volume' | 'fees' | 'apr'
  timeframe?: '1D' | '7D' | '30D' | '90D' | '1Y'
  onTimeframeChange?: (timeframe: string) => void
  className?: string
}

const timeframes = [
  { id: '1D', label: '1D' },
  { id: '7D', label: '7D' },
  { id: '30D', label: '30D' },
  { id: '90D', label: '90D' },
  { id: '1Y', label: '1Y' }
]

const metrics = [
  { id: 'value', label: 'Position Value', icon: BarChart3, color: '#6366F1' },
  { id: 'volume', label: 'Volume', icon: Activity, color: '#10B981' },
  { id: 'fees', label: 'Fees Earned', icon: TrendingUp, color: '#F59E0B' },
  { id: 'apr', label: 'APR', icon: LineChart, color: '#EF4444' }
]

export function HistoricalChart({ 
  data, 
  title = 'Historical Performance',
  metric = 'value',
  timeframe = '30D',
  onTimeframeChange,
  className = ''
}: HistoricalChartProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe)
  const [selectedMetric, setSelectedMetric] = useState(metric)
  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null)

  const chartData = useMemo(() => {
    if (!data.length) return []
    
    // Filter data based on timeframe
    const now = Date.now()
    const timeframeDays = {
      '1D': 1,
      '7D': 7,
      '30D': 30,
      '90D': 90,
      '1Y': 365
    }
    
    const cutoff = now - (timeframeDays[selectedTimeframe] * 24 * 60 * 60 * 1000)
    return data.filter(point => point.timestamp >= cutoff)
  }, [data, selectedTimeframe])

  const { path, area, points, stats } = useMemo(() => {
    if (chartData.length < 2) return { path: '', area: '', points: [], stats: null }
    
    const values = chartData.map(d => {
      switch (selectedMetric) {
        case 'volume': return d.volume || 0
        case 'fees': return d.fees || 0
        case 'apr': return (d.fees || 0) / (d.value || 1) * 365 * 100 // Rough APR calculation
        default: return d.value
      }
    })
    
    const min = Math.min(...values)
    const max = Math.max(...values)
    const range = max - min || 1
    
    const width = 800
    const height = 300
    const padding = 40
    
    const chartWidth = width - (padding * 2)
    const chartHeight = height - (padding * 2)
    
    const pathPoints = values.map((value, index) => {
      const x = padding + (index / (values.length - 1)) * chartWidth
      const y = padding + chartHeight - ((value - min) / range) * chartHeight
      return { x, y, value, data: chartData[index] }
    })
    
    const pathString = `M ${pathPoints.map(p => `${p.x},${p.y}`).join(' L ')}`
    const areaString = `${pathString} L ${pathPoints[pathPoints.length - 1].x},${height - padding} L ${padding},${height - padding} Z`
    
    // Calculate stats
    const firstValue = values[0]
    const lastValue = values[values.length - 1]
    const change = lastValue - firstValue
    const changePercent = (change / firstValue) * 100
    
    return {
      path: pathString,
      area: areaString,
      points: pathPoints,
      stats: {
        current: lastValue,
        change,
        changePercent,
        min,
        max,
        avg: values.reduce((a, b) => a + b, 0) / values.length
      }
    }
  }, [chartData, selectedMetric])

  const currentMetric = metrics.find(m => m.id === selectedMetric)
  const isPositive = stats ? stats.change >= 0 : true

  const formatValue = (value: number) => {
    if (selectedMetric === 'apr') return `${value.toFixed(2)}%`
    if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`
    if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`
    return `$${value.toFixed(2)}`
  }

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: selectedTimeframe === '1D' ? 'numeric' : undefined,
      minute: selectedTimeframe === '1D' ? '2-digit' : undefined
    })
  }

  return (
    <div className={`bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 ${className}`}>
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between mb-6 space-y-4 lg:space-y-0">
        <div>
          <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
          {stats && (
            <div className="flex items-center space-x-4">
              <div className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                {formatValue(stats.current)}
              </div>
              <div className={`flex items-center space-x-1 text-sm font-medium ${
                isPositive ? 'text-success' : 'text-error'
              }`}>
                {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                <span className="mono-numbers">
                  {isPositive ? '+' : ''}{formatValue(stats.change)} ({stats.changePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Metric Selector */}
          <div className="flex bg-slate-100 dark:bg-dark-600 rounded-lg p-1">
            {metrics.map((m) => {
              const Icon = m.icon
              return (
                <button
                  key={m.id}
                  onClick={() => setSelectedMetric(m.id as 'value' | 'volume' | 'fees' | 'apr')}
                  className={`flex items-center space-x-1 px-3 py-2 text-xs font-medium rounded-md transition-all ${
                    selectedMetric === m.id
                      ? 'bg-white dark:bg-dark-700 text-slate-900 dark:text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  <Icon className="w-3 h-3" />
                  <span>{m.label}</span>
                </button>
              )
            })}
          </div>
          
          {/* Timeframe Selector */}
          <div className="flex bg-slate-100 dark:bg-dark-600 rounded-lg p-1">
            {timeframes.map((tf) => (
              <button
                key={tf.id}
                onClick={() => {
                  setSelectedTimeframe(tf.id as '1D' | '7D' | '30D' | '90D' | '1Y')
                  onTimeframeChange?.(tf.id)
                }}
                className={`px-3 py-2 text-xs font-medium rounded-md transition-all ${
                  selectedTimeframe === tf.id
                    ? 'bg-white dark:bg-dark-700 text-slate-900 dark:text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        <svg
          width="100%"
          height="300"
          viewBox="0 0 800 300"
          className="overflow-visible"
          onMouseLeave={() => setHoveredPoint(null)}
        >
          <defs>
            <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={currentMetric?.color} stopOpacity="0.2" />
              <stop offset="100%" stopColor={currentMetric?.color} stopOpacity="0.02" />
            </linearGradient>
          </defs>
          
          {/* Grid lines */}
          {[0, 1, 2, 3, 4].map(i => (
            <line
              key={i}
              x1="40"
              y1={40 + (i * 52.5)}
              x2="760"
              y2={40 + (i * 52.5)}
              stroke="currentColor"
              strokeWidth="1"
              className="text-slate-200 dark:text-white/10"
            />
          ))}
          
          {/* Area under curve */}
          {area && (
            <path
              d={area}
              fill="url(#chartGradient)"
            />
          )}
          
          {/* Main line */}
          {path && (
            <path
              d={path}
              stroke={currentMetric?.color}
              strokeWidth="3"
              fill="none"
              className="drop-shadow-sm"
            />
          )}
          
          {/* Data points */}
          {points.map((point, index) => (
            <circle
              key={index}
              cx={point.x}
              cy={point.y}
              r="4"
              fill={currentMetric?.color}
              className="cursor-pointer hover:r-6 transition-all"
              onMouseEnter={() => setHoveredPoint(point.data)}
            />
          ))}
        </svg>
        
        {/* Tooltip */}
        {hoveredPoint && (
          <div className="absolute top-4 right-4 bg-white dark:bg-dark-800 rounded-lg shadow-lg border border-slate-200 dark:border-white/10 p-3 min-w-48">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
              {formatDate(hoveredPoint.timestamp)}
            </div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white">
              {selectedMetric === 'value' && formatValue(hoveredPoint.value)}
              {selectedMetric === 'volume' && formatValue(hoveredPoint.volume || 0)}
              {selectedMetric === 'fees' && formatValue(hoveredPoint.fees || 0)}
              {selectedMetric === 'apr' && `${(((hoveredPoint.fees || 0) / (hoveredPoint.value || 1)) * 365 * 100).toFixed(2)}%`}
            </div>
          </div>
        )}
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-200 dark:border-white/10">
          <div className="text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">High</div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white mono-numbers">
              {formatValue(stats.max)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Low</div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white mono-numbers">
              {formatValue(stats.min)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Average</div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white mono-numbers">
              {formatValue(stats.avg)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-1">Period</div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white">
              {selectedTimeframe}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Generate sample historical data
export function generateHistoricalData(days: number = 30): DataPoint[] {
  const data: DataPoint[] = []
  const now = Date.now()
  let baseValue = 10000
  let baseVolume = 50000
  let baseFees = 100
  
  for (let i = days; i >= 0; i--) {
    const timestamp = now - (i * 24 * 60 * 60 * 1000)
    
    // Add some realistic variation
    const valueChange = (Math.random() - 0.5) * 500
    const volumeChange = (Math.random() - 0.5) * 10000
    const feesChange = (Math.random() - 0.5) * 20
    
    baseValue = Math.max(baseValue + valueChange, 1000)
    baseVolume = Math.max(baseVolume + volumeChange, 0)
    baseFees = Math.max(baseFees + feesChange, 0)
    
    data.push({
      timestamp,
      value: baseValue,
      volume: baseVolume,
      fees: baseFees
    })
  }
  
  return data
}