'use client'

import { useMemo } from 'react'

interface MiniChartProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  className?: string
}

export function MiniChart({ 
  data, 
  width = 120, 
  height = 40, 
  color = '#10B981',
  className = ''
}: MiniChartProps) {
  const path = useMemo(() => {
    if (data.length < 2) return ''
    
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width
      const y = height - ((value - min) / range) * height
      return `${x},${y}`
    })
    
    return `M ${points.join(' L ')}`
  }, [data, width, height])

  const gradient = useMemo(() => {
    const isPositive = data[data.length - 1] >= data[0]
    return isPositive ? color : '#EF4444'
  }, [data, color])

  return (
    <div className={`relative ${className}`}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
      >
        <defs>
          <linearGradient id={`gradient-${Math.random()}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={gradient} stopOpacity="0.3" />
            <stop offset="100%" stopColor={gradient} stopOpacity="0.05" />
          </linearGradient>
        </defs>
        
        {/* Area under the curve */}
        {path && (
          <path
            d={`${path} L ${width},${height} L 0,${height} Z`}
            fill={`url(#gradient-${Math.random()})`}
            className="opacity-50"
          />
        )}
        
        {/* Main line */}
        {path && (
          <path
            d={path}
            stroke={gradient}
            strokeWidth="2"
            fill="none"
            className="drop-shadow-sm"
          />
        )}
        
        {/* Data points */}
        {data.map((value, index) => {
          const min = Math.min(...data)
          const max = Math.max(...data)
          const range = max - min || 1
          const x = (index / (data.length - 1)) * width
          const y = height - ((value - min) / range) * height
          
          return (
            <circle
              key={index}
              cx={x}
              cy={y}
              r="2"
              fill={gradient}
              className="opacity-80"
            />
          )
        })}
      </svg>
    </div>
  )
}

// Generate sample data for demonstration
export function generateSampleData(days: number = 30, trend: 'up' | 'down' | 'volatile' = 'up'): number[] {
  const data: number[] = []
  let baseValue = 100
  
  for (let i = 0; i < days; i++) {
    let change = (Math.random() - 0.5) * 10
    
    if (trend === 'up') {
      change += 0.5
    } else if (trend === 'down') {
      change -= 0.5
    }
    // volatile uses random change
    
    baseValue += change
    data.push(Math.max(baseValue, 0))
  }
  
  return data
}