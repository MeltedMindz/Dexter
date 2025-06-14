'use client'

import { useState, useEffect, useMemo } from 'react'
import { useAccount, useBalance } from 'wagmi'
import { base } from 'wagmi/chains'

// Simplified token list - only essential tokens to reduce API calls
const ESSENTIAL_TOKENS = [
  {
    address: '0x4200000000000000000000000000000000000006' as `0x${string}`,
    symbol: 'WETH',
    name: 'Wrapped Ether',
    decimals: 18,
  },
  {
    address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`,
    symbol: 'USDC',
    name: 'USD Coin', 
    decimals: 6,
  },
  {
    address: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb' as `0x${string}`,
    symbol: 'DAI',
    name: 'Dai Stablecoin',
    decimals: 18,
  }
]

export interface TokenHolding {
  address: string
  symbol: string
  name: string
  balance: string
  balanceFormatted: string
  decimals: number
  usdValue?: number
}

export interface WalletHoldings {
  tokens: TokenHolding[]
  totalValue: number
  isLoading: boolean
  error?: string
}

// Fixed prices to avoid API calls that cause re-renders
const FIXED_PRICES: Record<string, number> = {
  'ETH': 2500,
  'WETH': 2500,
  'USDC': 1,
  'DAI': 1,
  'cbETH': 2600,
}

export function useWalletHoldingsFixed() {
  const { address, isConnected } = useAccount()
  const [holdings, setHoldings] = useState<WalletHoldings>({
    tokens: [],
    totalValue: 0,
    isLoading: false
  })

  // Single ETH balance call
  const { data: ethBalance, isLoading: ethLoading } = useBalance({
    address,
    chainId: base.id,
    query: {
      enabled: !!address && !!isConnected
    }
  })

  // Single WETH balance call (most important for DEX)
  const { data: wethBalance, isLoading: wethLoading } = useBalance({
    address,
    token: ESSENTIAL_TOKENS[0].address,
    chainId: base.id,
    query: {
      enabled: !!address && !!isConnected
    }
  })

  // Single USDC balance call
  const { data: usdcBalance, isLoading: usdcLoading } = useBalance({
    address,
    token: ESSENTIAL_TOKENS[1].address,
    chainId: base.id,
    query: {
      enabled: !!address && !!isConnected
    }
  })

  // Memoized to prevent re-calculation on every render
  const processedHoldings = useMemo(() => {
    if (!isConnected || !address) {
      return {
        tokens: [],
        totalValue: 0,
        isLoading: false
      }
    }

    const isLoading = ethLoading || wethLoading || usdcLoading
    if (isLoading) {
      return {
        tokens: [],
        totalValue: 0,
        isLoading: true
      }
    }

    const tokens: TokenHolding[] = []
    let totalValue = 0

    // Add ETH if balance exists
    if (ethBalance && parseFloat(ethBalance.formatted) > 0.001) {
      const usdValue = parseFloat(ethBalance.formatted) * FIXED_PRICES.ETH
      tokens.push({
        address: '0x0000000000000000000000000000000000000000',
        symbol: 'ETH',
        name: 'Ethereum',
        balance: ethBalance.value.toString(),
        balanceFormatted: parseFloat(ethBalance.formatted).toFixed(4),
        decimals: 18,
        usdValue
      })
      totalValue += usdValue
    }

    // Add WETH if balance exists
    if (wethBalance && parseFloat(wethBalance.formatted) > 0.001) {
      const usdValue = parseFloat(wethBalance.formatted) * FIXED_PRICES.WETH
      tokens.push({
        address: ESSENTIAL_TOKENS[0].address,
        symbol: 'WETH',
        name: 'Wrapped Ether',
        balance: wethBalance.value.toString(),
        balanceFormatted: parseFloat(wethBalance.formatted).toFixed(4),
        decimals: 18,
        usdValue
      })
      totalValue += usdValue
    }

    // Add USDC if balance exists
    if (usdcBalance && parseFloat(usdcBalance.formatted) > 0.001) {
      const usdValue = parseFloat(usdcBalance.formatted) * FIXED_PRICES.USDC
      tokens.push({
        address: ESSENTIAL_TOKENS[1].address,
        symbol: 'USDC',
        name: 'USD Coin',
        balance: usdcBalance.value.toString(),
        balanceFormatted: parseFloat(usdcBalance.formatted).toFixed(2),
        decimals: 6,
        usdValue
      })
      totalValue += usdValue
    }

    return {
      tokens: tokens.sort((a, b) => (b.usdValue || 0) - (a.usdValue || 0)),
      totalValue,
      isLoading: false
    }
  }, [ethBalance, wethBalance, usdcBalance, ethLoading, wethLoading, usdcLoading, isConnected, address])

  // Simple effect that only updates when processedHoldings changes
  useEffect(() => {
    setHoldings(processedHoldings)
  }, [processedHoldings])

  return holdings
}