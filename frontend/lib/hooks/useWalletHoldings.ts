'use client'

import { useState, useEffect } from 'react'
import { useAccount, useBalance } from 'wagmi'
import { base } from 'wagmi/chains'
import { formatUnits } from 'viem'
import { useTokenPrices } from './useTokenPrices'

// Popular Base network tokens
const BASE_TOKENS = [
  {
    address: '0x4200000000000000000000000000000000000006' as `0x${string}`,
    symbol: 'WETH',
    name: 'Wrapped Ether',
    decimals: 18,
    logoURI: 'https://ethereum-optimism.github.io/data/WETH/logo.png'
  },
  {
    address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`,
    symbol: 'USDC',
    name: 'USD Coin',
    decimals: 6,
    logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png'
  },
  {
    address: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb' as `0x${string}`,
    symbol: 'DAI',
    name: 'Dai Stablecoin',
    decimals: 18,
    logoURI: 'https://ethereum-optimism.github.io/data/DAI/logo.png'
  },
  {
    address: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA' as `0x${string}`,
    symbol: 'USDbC',
    name: 'USD Base Coin',
    decimals: 6,
    logoURI: 'https://ethereum-optimism.github.io/data/USDbC/logo.png'
  },
  {
    address: '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22' as `0x${string}`,
    symbol: 'cbETH',
    name: 'Coinbase Wrapped Staked ETH',
    decimals: 18,
    logoURI: 'https://ethereum-optimism.github.io/data/cbETH/logo.png'
  }
]

export interface TokenHolding {
  address: string
  symbol: string
  name: string
  balance: string
  balanceFormatted: string
  decimals: number
  logoURI: string
  usdValue?: number
}

export interface WalletHoldings {
  tokens: TokenHolding[]
  totalValue: number
  isLoading: boolean
  error?: string
}

export function useWalletHoldings() {
  const { address, isConnected } = useAccount()
  const { prices, isLoading: pricesLoading } = useTokenPrices()
  const [holdings, setHoldings] = useState<WalletHoldings>({
    tokens: [],
    totalValue: 0,
    isLoading: false
  })

  // ETH balance
  const { data: ethBalance, isLoading: ethLoading } = useBalance({
    address,
    chainId: base.id,
  })

  // Token balances for each Base token
  const tokenBalances = BASE_TOKENS.map(token => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useBalance({
      address,
      token: token.address,
      chainId: base.id,
    })
  })

  useEffect(() => {
    if (!isConnected || !address) {
      setHoldings({
        tokens: [],
        totalValue: 0,
        isLoading: false
      })
      return
    }

    setHoldings(prev => ({ ...prev, isLoading: true }))

    const processHoldings = async () => {
      try {
        const tokens: TokenHolding[] = []

        // Add ETH if balance exists
        if (ethBalance && parseFloat(ethBalance.formatted) > 0.001) {
          tokens.push({
            address: '0x0000000000000000000000000000000000000000',
            symbol: 'ETH',
            name: 'Ethereum',
            balance: ethBalance.value.toString(),
            balanceFormatted: parseFloat(ethBalance.formatted).toFixed(4),
            decimals: 18,
            logoURI: 'https://ethereum-optimism.github.io/data/ETH/logo.png',
            usdValue: parseFloat(ethBalance.formatted) * (prices.ETH || 2500)
          })
        }

        // Add ERC20 tokens with balances
        tokenBalances.forEach((balance, index) => {
          if (balance.data && parseFloat(balance.data.formatted) > 0.001) {
            const token = BASE_TOKENS[index]
            const formattedBalance = parseFloat(balance.data.formatted)
            
            tokens.push({
              address: token.address,
              symbol: token.symbol,
              name: token.name,
              balance: balance.data.value.toString(),
              balanceFormatted: formattedBalance.toFixed(4),
              decimals: token.decimals,
              logoURI: token.logoURI,
              usdValue: (prices[token.symbol] || getMockTokenPrice(token.symbol)) * formattedBalance
            })
          }
        })

        const totalValue = tokens.reduce((sum, token) => sum + (token.usdValue || 0), 0)

        setHoldings({
          tokens: tokens.sort((a, b) => (b.usdValue || 0) - (a.usdValue || 0)),
          totalValue,
          isLoading: false
        })
      } catch (error) {
        setHoldings({
          tokens: [],
          totalValue: 0,
          isLoading: false,
          error: 'Failed to fetch wallet holdings'
        })
      }
    }

    const isAnyLoading = ethLoading || tokenBalances.some(balance => balance.isLoading) || pricesLoading
    
    if (!isAnyLoading) {
      processHoldings()
    }
  }, [address, isConnected, ethBalance, ethLoading, tokenBalances, prices, pricesLoading])

  return holdings
}

// Mock token prices for demo
function getMockTokenPrice(symbol: string): number {
  const prices: Record<string, number> = {
    'ETH': 2500,
    'WETH': 2500,
    'USDC': 1,
    'USDbC': 1,
    'DAI': 1,
    'cbETH': 2600,
    'COMP': 45,
    'WBTC': 35000
  }
  return prices[symbol] || 1
}