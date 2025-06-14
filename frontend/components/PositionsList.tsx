'use client'

import { PositionCard } from './PositionCard'

export function PositionsList() {
  // Mock data - in real app this would come from API/contracts
  const positions = [
    {
      id: 1,
      pair: 'ETH/USDC',
      value: 45230.45,
      change24h: { amount: 127.34, percentage: 0.28 },
      range: { min: 1850, max: 2150, current: 1987 },
      feeTier: 0.05,
      fees: {
        earned: 234.56,
        compounded: 215.79,
        protocolFee: 18.77,
        totalProfit: 1247.89
      },
      isActive: true,
      nextCheck: '~3h'
    },
    {
      id: 2,
      pair: 'WBTC/ETH',
      value: 32100.00,
      change24h: { amount: 89.12, percentage: 0.28 },
      range: { min: 15.2, max: 16.8, current: 16.1 },
      feeTier: 0.05,
      fees: {
        earned: 156.78,
        compounded: 144.24,
        protocolFee: 12.54,
        totalProfit: 892.45
      },
      isActive: true,
      nextCheck: '~1h'
    },
    {
      id: 3,
      pair: 'PEPE/ETH',
      value: 12400.00,
      change24h: { amount: 34.21, percentage: 0.28 },
      range: { min: 0.000008, max: 0.000012, current: 0.000009 },
      feeTier: 0.3,
      fees: {
        earned: 89.34,
        compounded: 82.19,
        protocolFee: 7.15,
        totalProfit: 234.67
      },
      isActive: false,
      nextCheck: 'Paused'
    }
  ]

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center space-x-2">
        <span>📈 Your Positions</span>
      </h2>
      
      <div className="space-y-4">
        {positions.map((position) => (
          <PositionCard key={position.id} position={position} />
        ))}
      </div>
    </div>
  )
}