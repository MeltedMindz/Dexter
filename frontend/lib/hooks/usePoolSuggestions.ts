'use client'

import { useMemo } from 'react'
import { useWalletHoldings, TokenHolding } from './useWalletHoldings'

export interface PoolSuggestion {
  id: string
  token0: {
    symbol: string
    address: string
    logoURI: string
  }
  token1: {
    symbol: string
    address: string
    logoURI: string
  }
  feeTier: number
  estimatedApr: number
  volume24h: number
  tvl: number
  reasonScore: number
  reasons: string[]
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH'
  category: 'STABLECOIN' | 'ETH_PAIR' | 'BLUE_CHIP' | 'YIELD_FARMING'
  userHoldings?: {
    token0Balance?: string
    token1Balance?: string
    canCreatePosition: boolean
    suggestedAllocation: number
  }
}

// Popular Uniswap V4 pools on Base (mock data with real-like metrics)
const POPULAR_POOLS: Omit<PoolSuggestion, 'userHoldings' | 'reasonScore' | 'reasons'>[] = [
  {
    id: 'eth-usdc-005',
    token0: {
      symbol: 'ETH',
      address: '0x0000000000000000000000000000000000000000',
      logoURI: 'https://ethereum-optimism.github.io/data/ETH/logo.png'
    },
    token1: {
      symbol: 'USDC',
      address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
      logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png'
    },
    feeTier: 0.05,
    estimatedApr: 28.5,
    volume24h: 12500000,
    tvl: 85000000,
    riskLevel: 'MEDIUM',
    category: 'ETH_PAIR'
  },
  {
    id: 'usdc-usdbc-001',
    token0: {
      symbol: 'USDC',
      address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
      logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png'
    },
    token1: {
      symbol: 'USDbC',
      address: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',
      logoURI: 'https://ethereum-optimism.github.io/data/USDbC/logo.png'
    },
    feeTier: 0.01,
    estimatedApr: 15.2,
    volume24h: 8500000,
    tvl: 45000000,
    riskLevel: 'LOW',
    category: 'STABLECOIN'
  },
  {
    id: 'weth-cbeth-005',
    token0: {
      symbol: 'WETH',
      address: '0x4200000000000000000000000000000000000006',
      logoURI: 'https://ethereum-optimism.github.io/data/WETH/logo.png'
    },
    token1: {
      symbol: 'cbETH',
      address: '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
      logoURI: 'https://ethereum-optimism.github.io/data/cbETH/logo.png'
    },
    feeTier: 0.05,
    estimatedApr: 22.8,
    volume24h: 6200000,
    tvl: 32000000,
    riskLevel: 'MEDIUM',
    category: 'ETH_PAIR'
  },
  {
    id: 'eth-dai-03',
    token0: {
      symbol: 'ETH',
      address: '0x0000000000000000000000000000000000000000',
      logoURI: 'https://ethereum-optimism.github.io/data/ETH/logo.png'
    },
    token1: {
      symbol: 'DAI',
      address: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
      logoURI: 'https://ethereum-optimism.github.io/data/DAI/logo.png'
    },
    feeTier: 0.3,
    estimatedApr: 35.1,
    volume24h: 4800000,
    tvl: 18000000,
    riskLevel: 'HIGH',
    category: 'ETH_PAIR'
  },
  {
    id: 'usdc-dai-001',
    token0: {
      symbol: 'USDC',
      address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
      logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png'
    },
    token1: {
      symbol: 'DAI',
      address: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
      logoURI: 'https://ethereum-optimism.github.io/data/DAI/logo.png'
    },
    feeTier: 0.01,
    estimatedApr: 12.7,
    volume24h: 3200000,
    tvl: 25000000,
    riskLevel: 'LOW',
    category: 'STABLECOIN'
  }
]

export function usePoolSuggestions() {
  const { tokens: holdings, isLoading: holdingsLoading } = useWalletHoldings()

  const suggestions = useMemo(() => {
    if (holdingsLoading || holdings.length === 0) {
      return POPULAR_POOLS.map(pool => ({
        ...pool,
        reasonScore: 50,
        reasons: ['Popular pool with good volume'],
        userHoldings: {
          canCreatePosition: false,
          suggestedAllocation: 0
        }
      }))
    }

    return POPULAR_POOLS.map(pool => {
      const userHoldings = analyzeUserHoldings(pool, holdings)
      const { score, reasons } = calculateReasonScore(pool, holdings, userHoldings)

      return {
        ...pool,
        reasonScore: score,
        reasons,
        userHoldings
      }
    }).sort((a, b) => b.reasonScore - a.reasonScore)
  }, [holdings, holdingsLoading])

  return {
    suggestions,
    isLoading: holdingsLoading
  }
}

function analyzeUserHoldings(
  pool: Omit<PoolSuggestion, 'userHoldings' | 'reasonScore' | 'reasons'>,
  holdings: TokenHolding[]
) {
  const token0Holding = holdings.find(h => 
    h.address.toLowerCase() === pool.token0.address.toLowerCase() ||
    h.symbol === pool.token0.symbol
  )
  
  const token1Holding = holdings.find(h => 
    h.address.toLowerCase() === pool.token1.address.toLowerCase() ||
    h.symbol === pool.token1.symbol
  )

  const hasToken0 = token0Holding && parseFloat(token0Holding.balanceFormatted) > 0.001
  const hasToken1 = token1Holding && parseFloat(token1Holding.balanceFormatted) > 0.001
  const canCreatePosition = !!(hasToken0 && hasToken1)

  // Calculate suggested allocation based on holdings
  let suggestedAllocation = 0
  if (canCreatePosition) {
    const token0Value = token0Holding?.usdValue || 0
    const token1Value = token1Holding?.usdValue || 0
    const totalValue = token0Value + token1Value
    
    // Suggest allocating 20-50% of relevant holdings based on pool characteristics
    if (pool.category === 'STABLECOIN') {
      suggestedAllocation = Math.min(totalValue * 0.5, 10000) // Max $10k for stables
    } else if (pool.category === 'ETH_PAIR') {
      suggestedAllocation = Math.min(totalValue * 0.3, 5000) // Max $5k for ETH pairs
    } else {
      suggestedAllocation = Math.min(totalValue * 0.2, 2000) // Max $2k for others
    }
  }

  return {
    token0Balance: token0Holding?.balanceFormatted,
    token1Balance: token1Holding?.balanceFormatted,
    canCreatePosition,
    suggestedAllocation
  }
}

function calculateReasonScore(
  pool: Omit<PoolSuggestion, 'userHoldings' | 'reasonScore' | 'reasons'>,
  holdings: TokenHolding[],
  userHoldings: NonNullable<PoolSuggestion['userHoldings']>
): { score: number; reasons: string[] } {
  let score = 0
  const reasons: string[] = []

  // Base score from pool metrics
  if (pool.estimatedApr > 20) {
    score += 20
    reasons.push(`High APR of ${pool.estimatedApr.toFixed(1)}%`)
  } else if (pool.estimatedApr > 15) {
    score += 15
    reasons.push(`Good APR of ${pool.estimatedApr.toFixed(1)}%`)
  }

  if (pool.volume24h > 10000000) {
    score += 15
    reasons.push('High trading volume ($10M+)')
  } else if (pool.volume24h > 5000000) {
    score += 10
    reasons.push('Good trading volume ($5M+)')
  }

  if (pool.tvl > 50000000) {
    score += 10
    reasons.push('Large liquidity pool ($50M+ TVL)')
  }

  // User holdings bonus
  if (userHoldings.canCreatePosition) {
    score += 30
    reasons.push('You hold both tokens')
    
    if (userHoldings.suggestedAllocation > 1000) {
      score += 10
      reasons.push('Significant allocation possible')
    }
  } else {
    // Check if user has one of the tokens
    const hasOneToken = userHoldings.token0Balance || userHoldings.token1Balance
    if (hasOneToken) {
      score += 10
      reasons.push('You hold one of the tokens')
    }
  }

  // Risk level adjustment
  if (pool.riskLevel === 'LOW') {
    score += 5
    reasons.push('Low risk stablecoin pair')
  }

  // Category bonuses
  if (pool.category === 'ETH_PAIR' && holdings.some(h => h.symbol === 'ETH' || h.symbol === 'WETH')) {
    score += 10
    reasons.push('ETH pair matches your holdings')
  }

  if (pool.category === 'STABLECOIN' && holdings.some(h => ['USDC', 'DAI', 'USDbC'].includes(h.symbol))) {
    score += 8
    reasons.push('Stablecoin pair for lower risk')
  }

  return { score: Math.min(score, 100), reasons }
}