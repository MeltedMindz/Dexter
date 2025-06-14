'use client'

import { useState } from 'react'
import { MiniChart, generateSampleData } from './MiniChart'

interface FeeTier {
  percentage: number
  volume24h: number
  tvl: number
  apr: number
  fees24h: number
  data: number[]
}

const feeTiers: FeeTier[] = [
  {
    percentage: 0.01,
    volume24h: 0,
    tvl: 1155.75,
    apr: 0.000000011,
    fees24h: 0.033599775,
    data: generateSampleData(30, 'volatile')
  },
  {
    percentage: 0.05,
    volume24h: 2246094,
    tvl: 3342625,
    apr: 0.04520624,
    fees24h: 0.033599775,
    data: generateSampleData(30, 'up')
  },
  {
    percentage: 0.30,
    volume24h: 1204228,
    tvl: 7991560,
    apr: 0.04520624,
    fees24h: 0.033599775,
    data: generateSampleData(30, 'up')
  },
  {
    percentage: 1.00,
    volume24h: 1.48,
    tvl: 9191.67,
    apr: 0.000106060,
    fees24h: 0.033599775,
    data: generateSampleData(30, 'down')
  }
]

interface FeeTierSelectorProps {
  selectedTier?: number
  onTierSelect?: (tier: number) => void
}

export function FeeTierSelector({ selectedTier = 0.05, onTierSelect }: FeeTierSelectorProps) {
  const [selected, setSelected] = useState(selectedTier)

  const handleSelect = (percentage: number) => {
    setSelected(percentage)
    onTierSelect?.(percentage)
  }

  const formatNumber = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`
    } else if (value < 0.01 && value > 0) {
      return value.toExponential(2)
    }
    return `$${value.toFixed(2)}`
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
          Select fee tier
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Select a pool tier to add liquidity to
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {feeTiers.map((tier) => (
          <button
            key={tier.percentage}
            onClick={() => handleSelect(tier.percentage)}
            className={`relative p-4 rounded-xl border-2 text-left transition-all duration-200 hover:scale-105 ${
              selected === tier.percentage
                ? 'border-primary bg-primary/5 dark:bg-primary/10 shadow-lg shadow-primary/25'
                : 'border-slate-200 dark:border-white/10 bg-white dark:bg-dark-700 hover:border-primary/50 hover:shadow-md'
            }`}
          >
            {/* Fee Percentage */}
            <div className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
              {tier.percentage}%
            </div>
            
            {/* Volume */}
            <div className="space-y-1 mb-3">
              <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                Volume 24h
              </div>
              <div className="font-mono text-sm font-semibold text-slate-700 dark:text-slate-300">
                {formatNumber(tier.volume24h)}
              </div>
            </div>
            
            {/* Mini Chart */}
            <div className="mb-3">
              <MiniChart 
                data={tier.data} 
                width={100} 
                height={30}
                color={selected === tier.percentage ? '#6366F1' : '#10B981'}
              />
            </div>
            
            {/* TVL */}
            <div className="space-y-1 mb-3">
              <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                TVL 24h
              </div>
              <div className="font-mono text-sm font-semibold text-slate-700 dark:text-slate-300">
                {formatNumber(tier.tvl)}
              </div>
            </div>
            
            {/* APR/Fees */}
            <div className="space-y-1">
              <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                Fees/TVL 24h
              </div>
              <div className="font-mono text-xs text-slate-600 dark:text-slate-400">
                {tier.apr.toExponential(2)}
              </div>
            </div>
            
            {/* Selection Indicator */}
            {selected === tier.percentage && (
              <div className="absolute top-2 right-2 w-3 h-3 bg-primary rounded-full animate-pulse-glow" />
            )}
          </button>
        ))}
      </div>
    </div>
  )
}