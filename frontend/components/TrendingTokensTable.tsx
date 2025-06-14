'use client'

import { useState, useMemo } from 'react'
import { ChevronUp, ChevronDown, ArrowUpDown, TrendingUp, TrendingDown } from 'lucide-react'
import { TokenIcon } from './TokenIcon'

interface TokenData {
  id: string
  name: string
  symbol: string
  logo: string
  price: number
  age: string
  transactions: number
  volume: number
  makers: number
  priceChange5m: number
  priceChange1h: number
  priceChange6h: number
  priceChange24h: number
  liquidity: number
  marketCap: number
}

// Mock data for demonstration
const mockTokens: TokenData[] = [
  {
    id: '1',
    name: 'Ethereum',
    symbol: 'ETH',
    logo: '/tokens/eth.png',
    price: 2456.78,
    age: '9y 3m',
    transactions: 1234567,
    volume: 45678900,
    makers: 8901,
    priceChange5m: 0.45,
    priceChange1h: -1.23,
    priceChange6h: 2.34,
    priceChange24h: -0.89,
    liquidity: 123456789,
    marketCap: 295000000000
  },
  {
    id: '2',
    name: 'Uniswap',
    symbol: 'UNI',
    logo: '/tokens/uni.png',
    price: 12.45,
    age: '4y 2m',
    transactions: 567890,
    volume: 12345678,
    makers: 3456,
    priceChange5m: 2.15,
    priceChange1h: 0.78,
    priceChange6h: -0.45,
    priceChange24h: 3.21,
    liquidity: 45678901,
    marketCap: 9800000000
  },
  {
    id: '3',
    name: 'Chainlink',
    symbol: 'LINK',
    logo: '/tokens/link.png',
    price: 18.92,
    age: '6y 1m',
    transactions: 234567,
    volume: 8901234,
    makers: 2345,
    priceChange5m: -0.34,
    priceChange1h: 1.45,
    priceChange6h: 0.89,
    priceChange24h: -2.12,
    liquidity: 23456789,
    marketCap: 11200000000
  },
  {
    id: '4',
    name: 'Aave',
    symbol: 'AAVE',
    logo: '/tokens/aave.png',
    price: 156.78,
    age: '4y 8m',
    transactions: 123456,
    volume: 5678901,
    makers: 1789,
    priceChange5m: 1.23,
    priceChange1h: -0.56,
    priceChange6h: 2.78,
    priceChange24h: 1.45,
    liquidity: 34567890,
    marketCap: 2300000000
  },
  {
    id: '5',
    name: 'Compound',
    symbol: 'COMP',
    logo: '/tokens/comp.png',
    price: 67.23,
    age: '4y 1m',
    transactions: 89012,
    volume: 3456789,
    makers: 1234,
    priceChange5m: -1.45,
    priceChange1h: 0.23,
    priceChange6h: -0.78,
    priceChange24h: 0.56,
    liquidity: 12345678,
    marketCap: 1100000000
  }
]

type SortField = keyof TokenData
type SortDirection = 'asc' | 'desc'

const formatPrice = (price: number) => {
  if (price >= 1000) {
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }
  return `$${price.toFixed(4)}`
}

const formatNumber = (num: number) => {
  if (num >= 1e9) {
    return `${(num / 1e9).toFixed(2)}B`
  }
  if (num >= 1e6) {
    return `${(num / 1e6).toFixed(2)}M`
  }
  if (num >= 1e3) {
    return `${(num / 1e3).toFixed(2)}K`
  }
  return num.toString()
}

const formatPercentage = (change: number) => {
  const isPositive = change >= 0
  return (
    <span className={`flex items-center font-mono text-xs ${
      isPositive ? 'text-primary' : 'text-red-500'
    }`}>
      {isPositive ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
      {isPositive ? '+' : ''}{change.toFixed(2)}%
    </span>
  )
}

export function TrendingTokensTable() {
  const [sortField, setSortField] = useState<SortField>('volume')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  const sortedTokens = useMemo(() => {
    return [...mockTokens].sort((a, b) => {
      const aValue = a[sortField]
      const bValue = b[sortField]
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue)
      }
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortDirection === 'asc' ? aValue - bValue : bValue - aValue
      }
      
      return 0
    })
  }, [sortField, sortDirection])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField === field) {
      return sortDirection === 'asc' ? (
        <ChevronUp className="w-4 h-4" />
      ) : (
        <ChevronDown className="w-4 h-4" />
      )
    }
    return <ArrowUpDown className="w-4 h-4 opacity-50" />
  }

  return (
    <div className="bg-white dark:bg-black border-2 border-black dark:border-white">
      {/* Header */}
      <div className="px-6 py-4 border-b-2 border-black dark:border-white">
        <h2 className="text-xl font-bold text-black dark:text-white text-brutal">
          TRENDING TOKENS
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Real-time market data and performance metrics
        </p>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[800px]">
          <thead>
            <tr className="border-b-2 border-black dark:border-white bg-gray-100 dark:bg-gray-900">
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('name')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors"
                >
                  <span>TOKEN</span>
                  <SortIcon field="name" />
                </button>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('price')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors ml-auto"
                >
                  <span>PRICE</span>
                  <SortIcon field="price" />
                </button>
              </th>
              <th className="px-4 py-3 text-center">
                <button
                  onClick={() => handleSort('age')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors mx-auto"
                >
                  <span>AGE</span>
                  <SortIcon field="age" />
                </button>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('transactions')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors ml-auto"
                >
                  <span>TXN</span>
                  <SortIcon field="transactions" />
                </button>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('volume')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors ml-auto"
                >
                  <span>VOLUME</span>
                  <SortIcon field="volume" />
                </button>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('makers')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors ml-auto"
                >
                  <span>MAKERS</span>
                  <SortIcon field="makers" />
                </button>
              </th>
              <th className="px-4 py-3 text-center">
                <span className="text-xs text-brutal text-black dark:text-white">PRICE CHANGES</span>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('liquidity')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors ml-auto"
                >
                  <span>LIQUIDITY</span>
                  <SortIcon field="liquidity" />
                </button>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('marketCap')}
                  className="flex items-center space-x-2 text-xs text-brutal text-black dark:text-white hover:text-primary transition-colors ml-auto"
                >
                  <span>MARKET CAP</span>
                  <SortIcon field="marketCap" />
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedTokens.map((token, index) => (
              <tr
                key={token.id}
                className={`border-b border-gray-200 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors ${
                  index % 2 === 0 ? 'bg-white dark:bg-black' : 'bg-gray-50 dark:bg-gray-950'
                }`}
              >
                {/* Token */}
                <td className="px-4 py-4">
                  <div className="flex items-center space-x-3">
                    <TokenIcon symbol={token.symbol} name={token.name} size="md" />
                    <div>
                      <div className="font-bold text-black dark:text-white text-sm">
                        {token.name}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 font-mono">
                        {token.symbol}
                      </div>
                    </div>
                  </div>
                </td>

                {/* Price */}
                <td className="px-4 py-4 text-right">
                  <span className="font-mono text-sm font-bold text-black dark:text-white">
                    {formatPrice(token.price)}
                  </span>
                </td>

                {/* Age */}
                <td className="px-4 py-4 text-center">
                  <span className="text-xs font-mono text-gray-600 dark:text-gray-400">
                    {token.age}
                  </span>
                </td>

                {/* Transactions */}
                <td className="px-4 py-4 text-right">
                  <span className="text-sm font-mono text-black dark:text-white">
                    {formatNumber(token.transactions)}
                  </span>
                </td>

                {/* Volume */}
                <td className="px-4 py-4 text-right">
                  <span className="text-sm font-mono font-bold text-black dark:text-white">
                    ${formatNumber(token.volume)}
                  </span>
                </td>

                {/* Makers */}
                <td className="px-4 py-4 text-right">
                  <span className="text-sm font-mono text-black dark:text-white">
                    {formatNumber(token.makers)}
                  </span>
                </td>

                {/* Price Changes */}
                <td className="px-4 py-4">
                  <div className="grid grid-cols-2 gap-2 text-center">
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">5M</div>
                      {formatPercentage(token.priceChange5m)}
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">1H</div>
                      {formatPercentage(token.priceChange1h)}
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">6H</div>
                      {formatPercentage(token.priceChange6h)}
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">24H</div>
                      {formatPercentage(token.priceChange24h)}
                    </div>
                  </div>
                </td>

                {/* Liquidity */}
                <td className="px-4 py-4 text-right">
                  <span className="text-sm font-mono text-black dark:text-white">
                    ${formatNumber(token.liquidity)}
                  </span>
                </td>

                {/* Market Cap */}
                <td className="px-4 py-4 text-right">
                  <span className="text-sm font-mono font-bold text-black dark:text-white">
                    ${formatNumber(token.marketCap)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t-2 border-black dark:border-white bg-gray-100 dark:bg-gray-900">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-600 dark:text-gray-400">
            Showing {sortedTokens.length} tokens
          </span>
          <span className="text-xs text-gray-600 dark:text-gray-400">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  )
}