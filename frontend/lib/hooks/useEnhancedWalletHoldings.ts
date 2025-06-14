'use client'

import { useState, useEffect } from 'react'
import { useAccount } from 'wagmi'
import { useTokenPrices } from './useTokenPrices'
import { useAlchemyTokenBalances } from './useAlchemyData'
import { formatUnits } from 'viem'

export interface EnhancedTokenHolding {
  address: string
  symbol: string
  name: string
  balance: string
  balanceFormatted: string
  decimals: number
  logoURI?: string | null
  usdValue?: number
  priceChange24h?: number
  metadata?: {
    name?: string | null
    symbol?: string | null
    logo?: string | null
    decimals?: number | null
  }
}

export interface EnhancedWalletHoldings {
  tokens: EnhancedTokenHolding[]
  totalValue: number
  totalChange24h: number
  isLoading: boolean
  error?: string
  lastUpdated?: Date
}

export function useEnhancedWalletHoldings(network: 'base' | 'mainnet' = 'base') {
  const { address, isConnected } = useAccount()
  const { prices, isLoading: pricesLoading } = useTokenPrices()
  const { balances, loading: alchemyLoading, error: alchemyError } = useAlchemyTokenBalances(network)
  
  const [holdings, setHoldings] = useState<EnhancedWalletHoldings>({
    tokens: [],
    totalValue: 0,
    totalChange24h: 0,
    isLoading: false
  })

  useEffect(() => {
    if (!isConnected || !address) {
      setHoldings({
        tokens: [],
        totalValue: 0,
        totalChange24h: 0,
        isLoading: false
      })
      return
    }

    if (alchemyLoading || pricesLoading) {
      setHoldings(prev => ({ ...prev, isLoading: true }))
      return
    }

    if (alchemyError) {
      setHoldings({
        tokens: [],
        totalValue: 0,
        totalChange24h: 0,
        isLoading: false,
        error: alchemyError
      })
      return
    }

    const processHoldings = async () => {
      try {
        const tokens: EnhancedTokenHolding[] = []

        // Process Alchemy token balances
        for (const balance of balances) {
          if (balance.tokenBalance && balance.tokenBalance !== '0x0') {
            const decimals = balance.metadata?.decimals || 18
            const balanceFormatted = formatUnits(BigInt(balance.tokenBalance), decimals)
            const balanceNumber = parseFloat(balanceFormatted)
            
            if (balanceNumber > 0.001) {
              const symbol = balance.metadata?.symbol || 'Unknown'
              const currentPrice = prices[symbol] || 0
              const usdValue = balanceNumber * currentPrice
              
              tokens.push({
                address: balance.contractAddress,
                symbol,
                name: balance.metadata?.name || 'Unknown Token',
                balance: balance.tokenBalance,
                balanceFormatted: balanceNumber.toFixed(6),
                decimals,
                logoURI: balance.metadata?.logo,
                usdValue,
                metadata: balance.metadata
              })
            }
          }
        }

        // Sort by USD value (highest first)
        tokens.sort((a, b) => (b.usdValue || 0) - (a.usdValue || 0))

        const totalValue = tokens.reduce((sum, token) => sum + (token.usdValue || 0), 0)
        
        // Calculate total 24h change (simplified - in a real app you'd track historical values)
        const totalChange24h = totalValue * 0.02 // Mock 2% change

        setHoldings({
          tokens,
          totalValue,
          totalChange24h,
          isLoading: false,
          lastUpdated: new Date()
        })
      } catch (error) {
        setHoldings({
          tokens: [],
          totalValue: 0,
          totalChange24h: 0,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to fetch enhanced wallet holdings'
        })
      }
    }

    processHoldings()
  }, [address, isConnected, balances, prices, alchemyLoading, pricesLoading, alchemyError])

  return holdings
}

// Enhanced token filtering and search
export function useFilteredTokenHoldings(
  holdings: EnhancedWalletHoldings,
  searchTerm: string = '',
  minUsdValue: number = 0.01
) {
  const [filteredTokens, setFilteredTokens] = useState<EnhancedTokenHolding[]>([])

  useEffect(() => {
    const filtered = holdings.tokens.filter(token => {
      const matchesSearch = !searchTerm || 
        token.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        token.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        token.address.toLowerCase().includes(searchTerm.toLowerCase())
      
      const meetsMinValue = (token.usdValue || 0) >= minUsdValue
      
      return matchesSearch && meetsMinValue
    })

    setFilteredTokens(filtered)
  }, [holdings.tokens, searchTerm, minUsdValue])

  return filteredTokens
}