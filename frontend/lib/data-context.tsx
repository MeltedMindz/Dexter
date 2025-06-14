'use client'

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { generateSampleData } from '@/components/MiniChart'
import { generateHistoricalData } from '@/components/HistoricalChart'

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

interface PortfolioStats {
  totalValue: number
  totalFees: number
  totalProfit: number
  avgApr: number
  activePositions: number
  change24h: { amount: number; percentage: number }
}

interface DataContextType {
  positions: Position[]
  portfolioStats: PortfolioStats
  isLoading: boolean
  lastUpdated: Date
  refreshData: () => void
  subscribeToUpdates: (callback: () => void) => () => void
}

const DataContext = createContext<DataContextType | undefined>(undefined)

// Mock API service to simulate real-time data
class MockDataService {
  private subscribers: Array<() => void> = []
  private updateInterval: NodeJS.Timeout | null = null
  
  constructor() {
    this.startRealTimeUpdates()
  }

  private startRealTimeUpdates() {
    // Update data every 30 seconds to simulate real-time changes
    this.updateInterval = setInterval(() => {
      this.notifySubscribers()
    }, 30000)
  }

  subscribe(callback: () => void) {
    this.subscribers.push(callback)
    
    return () => {
      this.subscribers = this.subscribers.filter(sub => sub !== callback)
    }
  }

  private notifySubscribers() {
    this.subscribers.forEach(callback => callback())
  }

  // Simulate fetching fresh data
  generatePositions(): Position[] {
    return [
      {
        id: 1,
        pair: 'ETH/USDC',
        value: this.generateRandomValue(18000, 20000),
        change24h: this.generateRandomChange(),
        range: { min: 1800, max: 2200, current: this.generateRandomValue(1900, 2100) },
        feeTier: 0.05,
        fees: {
          earned: this.generateRandomValue(800, 1200),
          compounded: this.generateRandomValue(600, 900),
          protocolFee: this.generateRandomValue(50, 100),
          totalProfit: this.generateRandomValue(1200, 1800)
        },
        isActive: Math.random() > 0.2,
        nextCheck: this.generateNextCheck(),
        volume24h: this.generateRandomValue(800000, 1200000),
        apr: this.generateRandomValue(20, 35),
        chartData: generateSampleData(30, Math.random() > 0.3 ? 'up' : 'volatile')
      },
      {
        id: 2,
        pair: 'BTC/ETH',
        value: this.generateRandomValue(15000, 17000),
        change24h: this.generateRandomChange(),
        range: { min: 0.065, max: 0.075, current: this.generateRandomValue(0.068, 0.072) },
        feeTier: 0.3,
        fees: {
          earned: this.generateRandomValue(600, 1000),
          compounded: this.generateRandomValue(500, 800),
          protocolFee: this.generateRandomValue(40, 80),
          totalProfit: this.generateRandomValue(1000, 1500)
        },
        isActive: Math.random() > 0.15,
        nextCheck: this.generateNextCheck(),
        volume24h: this.generateRandomValue(600000, 900000),
        apr: this.generateRandomValue(18, 28),
        chartData: generateSampleData(30, Math.random() > 0.4 ? 'up' : 'down')
      },
      {
        id: 3,
        pair: 'USDC/USDT',
        value: this.generateRandomValue(12000, 14000),
        change24h: this.generateRandomChange(),
        range: { min: 0.998, max: 1.002, current: this.generateRandomValue(0.999, 1.001) },
        feeTier: 0.01,
        fees: {
          earned: this.generateRandomValue(400, 700),
          compounded: this.generateRandomValue(300, 600),
          protocolFee: this.generateRandomValue(30, 60),
          totalProfit: this.generateRandomValue(600, 1100)
        },
        isActive: Math.random() > 0.1,
        nextCheck: this.generateNextCheck(),
        volume24h: this.generateRandomValue(1000000, 1500000),
        apr: this.generateRandomValue(15, 25),
        chartData: generateSampleData(30, 'volatile')
      }
    ]
  }

  private generateRandomValue(min: number, max: number): number {
    return min + Math.random() * (max - min)
  }

  private generateRandomChange(): { amount: number; percentage: number } {
    const percentage = (Math.random() - 0.5) * 20 // -10% to +10%
    const amount = Math.abs(percentage) * 100 // Mock amount
    return { amount, percentage }
  }

  private generateNextCheck(): string {
    const minutes = Math.floor(Math.random() * 60)
    return `${minutes}m`
  }

  destroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
    }
  }
}

const mockDataService = new MockDataService()

export function DataProvider({ children }: { children: React.ReactNode }) {
  const [positions, setPositions] = useState<Position[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [mounted, setMounted] = useState(false)

  const refreshData = useCallback(() => {
    setIsLoading(true)
    // Simulate API delay
    setTimeout(() => {
      setPositions(mockDataService.generatePositions())
      setLastUpdated(new Date())
      setIsLoading(false)
    }, 500)
  }, [])

  const subscribeToUpdates = useCallback((callback: () => void) => {
    return mockDataService.subscribe(callback)
  }, [])

  // Calculate portfolio stats from positions
  const portfolioStats: PortfolioStats = React.useMemo(() => {
    const totalValue = positions.reduce((sum, pos) => sum + pos.value, 0)
    const totalFees = positions.reduce((sum, pos) => sum + pos.fees.earned, 0)
    const totalProfit = positions.reduce((sum, pos) => sum + pos.fees.totalProfit, 0)
    const avgApr = positions.length > 0 
      ? positions.reduce((sum, pos) => sum + (pos.apr || 0), 0) / positions.length 
      : 0
    const activePositions = positions.filter(pos => pos.isActive).length
    
    // Calculate total portfolio change
    const totalChange24h = positions.reduce((sum, pos) => sum + pos.change24h.amount, 0)
    const totalChangePercent = totalValue > 0 ? (totalChange24h / totalValue) * 100 : 0

    return {
      totalValue,
      totalFees,
      totalProfit,
      avgApr,
      activePositions,
      change24h: { amount: totalChange24h, percentage: totalChangePercent }
    }
  }, [positions])

  // Initial data load
  useEffect(() => {
    setMounted(true)
    refreshData()
  }, [refreshData])

  // Subscribe to real-time updates
  useEffect(() => {
    if (mounted) {
      const unsubscribe = mockDataService.subscribe(() => {
        refreshData()
      })

      return unsubscribe
    }
  }, [refreshData, mounted])

  return (
    <DataContext.Provider value={{
      positions,
      portfolioStats,
      isLoading,
      lastUpdated,
      refreshData,
      subscribeToUpdates
    }}>
      {children}
    </DataContext.Provider>
  )
}

export function useData() {
  const context = useContext(DataContext)
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider')
  }
  return context
}