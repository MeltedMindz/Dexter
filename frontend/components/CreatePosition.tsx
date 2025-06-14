'use client'

import { useState, useEffect } from 'react'
import { useAccount } from 'wagmi'
import { ChevronDown, Info, Zap, TrendingUp, AlertCircle, ExternalLink } from 'lucide-react'
import { uniswapService, POPULAR_TOKENS, type Pool } from '@/lib/uniswap'

const networks = [
  { id: 'ethereum', name: 'Ethereum', chainId: 1 },
  { id: 'base', name: 'Base', chainId: 8453 },
  { id: 'arbitrum', name: 'Arbitrum', chainId: 42161 },
  { id: 'polygon', name: 'Polygon', chainId: 137 }
]

const protocols = [
  { id: 'uniswap-v3', name: 'Uniswap V3', icon: '🦄' },
  { id: 'pancakeswap-v3', name: 'PancakeSwap V3', icon: '🥞' }
]

export function CreatePosition() {
  const { address, isConnected } = useAccount()
  const [selectedNetwork, setSelectedNetwork] = useState(networks[1]) // Default to Base
  const [selectedProtocol, setSelectedProtocol] = useState(protocols[0])
  const [availableTokens, setAvailableTokens] = useState(POPULAR_TOKENS.base)
  const [token0, setToken0] = useState(POPULAR_TOKENS.base[0])
  const [token1, setToken1] = useState(POPULAR_TOKENS.base[1])
  const [selectedFeeTier, setSelectedFeeTier] = useState(0.05)
  const [isLoading, setIsLoading] = useState(false)
  const [pools, setPools] = useState<Pool[]>([])
  const [error, setError] = useState<string | null>(null)

  // Update available tokens when network changes
  useEffect(() => {
    const networkKey = selectedNetwork.id as keyof typeof POPULAR_TOKENS
    const tokens = POPULAR_TOKENS[networkKey] || POPULAR_TOKENS.base
    setAvailableTokens(tokens)
    setToken0(tokens[0])
    setToken1(tokens[1])
    uniswapService.setNetwork(networkKey)
  }, [selectedNetwork])

  // Fetch pool data when tokens change
  useEffect(() => {
    if (token0.symbol === token1.symbol) return

    const fetchPools = async () => {
      setIsLoading(true)
      setError(null)
      
      try {
        console.log(`Fetching pools for ${token0.symbol}/${token1.symbol}`)
        const poolsData = await uniswapService.fetchPoolsByPair(token0.symbol, token1.symbol)
        setPools(poolsData)
        
        // Auto-select the most liquid pool if available
        if (poolsData.length > 0) {
          const bestPool = poolsData.reduce((prev, current) => 
            current.tvl > prev.tvl ? current : prev
          )
          setSelectedFeeTier(bestPool.feeTier)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch pool data')
        setPools([])
      } finally {
        setIsLoading(false)
      }
    }

    fetchPools()
  }, [token0, token1, selectedNetwork])

  const formatNumber = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`
    }
    return `$${value.toFixed(2)}`
  }

  const formatPercentage = (value: number) => {
    if (value < 0.001) return value.toExponential(2)
    return `${(value * 100).toFixed(3)}%`
  }

  if (!isConnected) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-slate-100 dark:bg-dark-600 rounded-full flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-slate-400" />
          </div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
            Connect Your Wallet
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Please connect your wallet to create new positions
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">Create new position</h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          Create a position by adding liquidity to a pool
        </p>
      </div>

      <div className="space-y-8">
        {/* Network and Protocol Selection */}
        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <ChevronDown className="w-5 h-5 text-slate-400" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
              Select network and exchange
            </h3>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
            Select a network, AMM protocol, and pair of tokens to provide liquidity for
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Network Selection */}
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                network
              </label>
              <div className="relative">
                <select 
                  value={selectedNetwork.id}
                  onChange={(e) => setSelectedNetwork(networks.find(n => n.id === e.target.value)!)}
                  className="w-full bg-slate-50 dark:bg-dark-600 border border-slate-300 dark:border-white/20 rounded-lg px-4 py-3 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent appearance-none"
                >
                  {networks.map(network => (
                    <option key={network.id} value={network.id}>
                      {network.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
            </div>

            {/* Protocol Selection */}
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                protocol
              </label>
              <div className="relative">
                <select 
                  value={selectedProtocol.id}
                  onChange={(e) => setSelectedProtocol(protocols.find(p => p.id === e.target.value)!)}
                  className="w-full bg-slate-50 dark:bg-dark-600 border border-slate-300 dark:border-white/20 rounded-lg px-4 py-3 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent appearance-none"
                >
                  {protocols.map(protocol => (
                    <option key={protocol.id} value={protocol.id}>
                      {protocol.icon} {protocol.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
            </div>
          </div>

          {/* Token Pair Selection */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              asset pair
            </label>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="relative">
                <select 
                  value={token0.symbol}
                  onChange={(e) => setToken0(availableTokens.find(t => t.symbol === e.target.value)!)}
                  className="w-full bg-slate-50 dark:bg-dark-600 border border-slate-300 dark:border-white/20 rounded-lg px-4 py-3 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent appearance-none"
                >
                  {availableTokens.map(token => (
                    <option key={token.symbol} value={token.symbol}>
                      {token.symbol}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
              
              <div className="relative">
                <select 
                  value={token1.symbol}
                  onChange={(e) => setToken1(availableTokens.find(t => t.symbol === e.target.value)!)}
                  className="w-full bg-slate-50 dark:bg-dark-600 border border-slate-300 dark:border-white/20 rounded-lg px-4 py-3 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent appearance-none"
                >
                  {availableTokens.filter(token => token.symbol !== token0.symbol).map(token => (
                    <option key={token.symbol} value={token.symbol}>
                      {token.symbol}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
            </div>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="mt-6 space-y-2">
              <div className="text-sm text-slate-600 dark:text-slate-400">Fetching pool tick data |</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Fetching pool period data @ 100 bips |</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Fetching pool period data @ 30 bips |</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Fetching pool period data @ 5 bips |</div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Fetching pool period data @ 1 bips |</div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Fee Tier Selection */}
        {!isLoading && !error && pools.length > 0 && (
          <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6">
            <div className="flex items-center space-x-2 mb-4">
              <ChevronDown className="w-5 h-5 text-slate-400" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Select fee tier
              </h3>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
              Select a pool tier to add liquidity to
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {pools.map((pool) => {
                const apr = uniswapService.calculateAPR(pool.fees24h, pool.tvl)
                return (
                  <button
                    key={pool.id}
                    onClick={() => setSelectedFeeTier(pool.feeTier)}
                    className={`relative p-4 rounded-xl border text-left transition-all hover:scale-105 ${
                      selectedFeeTier === pool.feeTier
                        ? 'border-primary bg-primary/5 dark:bg-primary/10 shadow-lg'
                        : 'border-slate-200 dark:border-white/10 hover:border-primary/50'
                    }`}
                  >
                    {/* Fee Percentage */}
                    <div className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
                      {pool.feeTier}%
                    </div>
                    
                    {/* Volume */}
                    <div className="space-y-1 mb-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                        Volume 24h
                      </div>
                      <div className="font-mono text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {formatNumber(pool.volume24h)}
                      </div>
                    </div>
                    
                    {/* Volume Bar Chart */}
                    <div className="mb-3 h-8 bg-slate-100 dark:bg-dark-600 rounded overflow-hidden">
                      <div 
                        className="h-full bg-primary/20 flex items-end justify-center"
                        style={{ 
                          width: `${Math.min(100, (pool.volume24h / Math.max(...pools.map(p => p.volume24h))) * 100)}%` 
                        }}
                      >
                        <div className="w-full h-2 bg-primary rounded-t"></div>
                      </div>
                    </div>
                    
                    {/* TVL */}
                    <div className="space-y-1 mb-3">
                      <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                        TVL 24h
                      </div>
                      <div className="font-mono text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {formatNumber(pool.tvl)}
                      </div>
                    </div>
                    
                    {/* APR */}
                    <div className="space-y-1">
                      <div className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                        Fees/TVL 24h
                      </div>
                      <div className="font-mono text-xs text-slate-600 dark:text-slate-400">
                        {formatPercentage(apr)}
                      </div>
                    </div>
                    
                    {/* Selection Indicator */}
                    {selectedFeeTier === pool.feeTier && (
                      <div className="absolute top-2 right-2 w-3 h-3 bg-primary rounded-full"></div>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* Continue Button */}
        <div className="flex justify-center">
          <button 
            disabled={!selectedFeeTier || isLoading || pools.length === 0}
            className="bg-primary hover:bg-primary-600 disabled:bg-slate-300 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg font-semibold transition-colors flex items-center space-x-2"
          >
            <Zap className="w-5 h-5" />
            <span>Continue to Set Range</span>
          </button>
        </div>

        {/* Selected Pool Info */}
        {pools.length > 0 && selectedFeeTier && (
          <div className="bg-slate-50 dark:bg-dark-600 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {token0.symbol}/{token1.symbol}
                </div>
                <div className="px-2 py-1 bg-primary/10 text-primary text-xs font-medium rounded">
                  {selectedFeeTier}% Fee
                </div>
              </div>
              <a
                href={`https://app.uniswap.org/#/pool/${pools.find(p => p.feeTier === selectedFeeTier)?.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-xs text-slate-500 dark:text-slate-400 hover:text-primary transition-colors"
              >
                <span>View on Uniswap</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}