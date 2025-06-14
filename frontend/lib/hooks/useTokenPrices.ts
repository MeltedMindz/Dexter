'use client'

import { useState, useEffect } from 'react'

interface TokenPrices {
  [symbol: string]: number
}

export function useTokenPrices() {
  const [prices, setPrices] = useState<TokenPrices>({})
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchPrices = async () => {
      try {
        // Using CoinGecko API for real token prices
        const response = await fetch(
          'https://api.coingecko.com/api/v3/simple/price?ids=ethereum,usd-coin,dai,coinbase-wrapped-staked-eth&vs_currencies=usd'
        )
        const data = await response.json()
        
        setPrices({
          ETH: data.ethereum?.usd || 2500,
          WETH: data.ethereum?.usd || 2500,
          USDC: data['usd-coin']?.usd || 1,
          USDbC: 1, // Assume 1:1 with USD
          DAI: data.dai?.usd || 1,
          cbETH: data['coinbase-wrapped-staked-eth']?.usd || 2600
        })
      } catch (error) {
        console.warn('Failed to fetch token prices, using fallback:', error)
        // Fallback prices
        setPrices({
          ETH: 2500,
          WETH: 2500,
          USDC: 1,
          USDbC: 1,
          DAI: 1,
          cbETH: 2600
        })
      }
      setIsLoading(false)
    }

    fetchPrices()
    // Refresh prices every 5 minutes
    const interval = setInterval(fetchPrices, 5 * 60 * 1000)

    return () => clearInterval(interval)
  }, [])

  return { prices, isLoading }
}