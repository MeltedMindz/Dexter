'use client'

import React, { useState, useEffect, useMemo } from 'react'
import { useAccount } from 'wagmi'
import { 
  Plus,
  Minus,
  DollarSign,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Settings,
  Target,
  Zap,
  ExternalLink,
  Info,
  ChevronDown,
  BarChart3
} from 'lucide-react'
import { useWalletHoldingsFixed } from '@/lib/hooks/useWalletHoldings-fixed'
import { uniswapService, POPULAR_TOKENS, type Pool } from '@/lib/uniswap'

// Step definitions
const STEPS = [
  { id: 1, name: 'Pool Selection', description: 'Choose token pair and fee tier' },
  { id: 2, name: 'Configure Position', description: 'Set price range and amounts' },
  { id: 3, name: 'Review & Confirm', description: 'Review your position details' },
  { id: 4, name: 'Deploy Position', description: 'Execute transaction and deploy' }
]

const networks = [
  { id: 'ethereum', name: 'Ethereum', chainId: 1 },
  { id: 'base', name: 'Base', chainId: 8453 },
  { id: 'arbitrum', name: 'Arbitrum', chainId: 42161 },
  { id: 'polygon', name: 'Polygon', chainId: 137 }
]

interface PositionConfig {
  network: typeof networks[0]
  token0: typeof POPULAR_TOKENS.base[0]
  token1: typeof POPULAR_TOKENS.base[0]
  feeTier: number
  amount0: string
  amount1: string
  minPrice: string
  maxPrice: string
  currentPrice: number
}

export function V4PositionCreator() {
  const { isConnected } = useAccount()
  const { tokens: holdings, totalValue, isLoading } = useWalletHoldingsFixed()
  
  // Wizard state
  const [currentStep, setCurrentStep] = useState(1)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  
  // Position configuration
  const [config, setConfig] = useState<PositionConfig>({
    network: networks[1], // Default to Base
    token0: POPULAR_TOKENS.base[0],
    token1: POPULAR_TOKENS.base[1],
    feeTier: 0.05,
    amount0: '',
    amount1: '',
    minPrice: '',
    maxPrice: '',
    currentPrice: 2500
  })

  // Pool data
  const [pools, setPools] = useState<Pool[]>([])
  const [poolsLoading, setPoolsLoading] = useState(false)
  const [poolsError, setPoolsError] = useState<string | null>(null)

  // Update available tokens when network changes
  const availableTokens = useMemo(() => {
    const networkKey = config.network.id as keyof typeof POPULAR_TOKENS
    return POPULAR_TOKENS[networkKey] || POPULAR_TOKENS.base
  }, [config.network])

  // Fetch pool data when tokens change
  useEffect(() => {
    if (config.token0.symbol === config.token1.symbol) return

    const fetchPools = async () => {
      setPoolsLoading(true)
      setPoolsError(null)
      
      try {
        const networkKey = config.network.id as keyof typeof POPULAR_TOKENS
        uniswapService.setNetwork(networkKey)
        
        const poolsData = await uniswapService.fetchPoolsByPair(config.token0.symbol, config.token1.symbol)
        setPools(poolsData)
        
        // Auto-select the most liquid pool if available
        if (poolsData.length > 0) {
          const bestPool = poolsData.reduce((prev, current) => 
            current.tvl > prev.tvl ? current : prev
          )
          setConfig(prev => ({ ...prev, feeTier: bestPool.feeTier }))
        }
      } catch (err) {
        setPoolsError(err instanceof Error ? err.message : 'Failed to fetch pool data')
        setPools([])
      } finally {
        setPoolsLoading(false)
      }
    }

    fetchPools()
  }, [config.token0, config.token1, config.network])

  // Wallet connection check
  if (!isConnected) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="text-center py-12">
          <Target className="w-16 h-16 text-slate-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">
            Connect Wallet to Create Position
          </h2>
          <p className="text-gray-300">
            Connect your wallet to start creating Uniswap V4 positions
          </p>
        </div>
      </div>
    )
  }

  // Step navigation
  const nextStep = () => {
    if (currentStep < STEPS.length) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return config.feeTier && pools.length > 0 && !poolsLoading
      case 2:
        return config.amount0 && config.amount1 && config.minPrice && config.maxPrice
      case 3:
        return true
      default:
        return false
    }
  }

  // Deploy position
  const deployPosition = async () => {
    setIsProcessing(true)
    try {
      // Simulate transaction processing
      await new Promise(resolve => setTimeout(resolve, 3000))
      setIsComplete(true)
    } catch (error) {
      console.error('Error deploying position:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  // Reset wizard
  const resetWizard = () => {
    setCurrentStep(1)
    setIsComplete(false)
    setIsProcessing(false)
    setConfig({
      network: networks[1],
      token0: POPULAR_TOKENS.base[0],
      token1: POPULAR_TOKENS.base[1],
      feeTier: 0.05,
      amount0: '',
      amount1: '',
      minPrice: '',
      maxPrice: '',
      currentPrice: 2500
    })
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">
          Create V4 Position
        </h1>
        <p className="text-gray-300 text-lg">
          Deploy advanced liquidity positions with Uniswap V4's revolutionary features
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {STEPS.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <div className={`
                relative flex items-center justify-center w-12 h-12 rounded-full border-2 font-bold
                ${currentStep >= step.id 
                  ? 'bg-primary border-primary text-white' 
                  : 'bg-gray-800 border-white text-gray-400'
                }
              `}>
                {isComplete && step.id <= 4 ? (
                  <CheckCircle className="w-6 h-6" />
                ) : (
                  step.id
                )}
              </div>
              {index < STEPS.length - 1 && (
                <div className={`
                  w-24 h-1 mx-4
                  ${currentStep > step.id ? 'bg-primary' : 'bg-gray-600'}
                `} />
              )}
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between mt-4">
          {STEPS.map((step) => (
            <div key={step.id} className="text-center flex-1">
              <p className={`
                font-medium text-sm
                ${currentStep >= step.id ? 'text-white' : 'text-gray-400'}
              `}>
                {step.name}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <div className="bg-gray-800 rounded-xl p-8 border-2 border-white min-h-[600px]">
        {currentStep === 1 && (
          <PoolSelectionStep 
            config={config}
            setConfig={setConfig}
            availableTokens={availableTokens}
            pools={pools}
            poolsLoading={poolsLoading}
            poolsError={poolsError}
          />
        )}
        
        {currentStep === 2 && (
          <ConfigurePositionStep 
            config={config}
            setConfig={setConfig}
            holdings={holdings}
          />
        )}
        
        {currentStep === 3 && (
          <ReviewConfirmStep 
            config={config}
            pools={pools}
          />
        )}
        
        {currentStep === 4 && (
          <DeployPositionStep 
            config={config}
            isProcessing={isProcessing}
            isComplete={isComplete}
            onDeploy={deployPosition}
          />
        )}
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-8">
        <button
          onClick={prevStep}
          disabled={currentStep === 1 || isProcessing}
          className="flex items-center space-x-2 px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="w-5 h-5" />
          <span>Previous</span>
        </button>

        {isComplete ? (
          <button
            onClick={resetWizard}
            className="flex items-center space-x-2 px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Plus className="w-5 h-5" />
            <span>Create Another Position</span>
          </button>
        ) : (
          <button
            onClick={currentStep === 4 ? deployPosition : nextStep}
            disabled={!canProceed() || isProcessing}
            className="flex items-center space-x-2 px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {currentStep === 4 ? (
              isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Deploying...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Deploy Position</span>
                </>
              )
            ) : (
              <>
                <span>Next</span>
                <ChevronRight className="w-5 h-5" />
              </>
            )}
          </button>
        )}
      </div>
    </div>
  )
}

// Step 1: Pool Selection
function PoolSelectionStep({ 
  config, 
  setConfig, 
  availableTokens, 
  pools, 
  poolsLoading, 
  poolsError 
}: {
  config: PositionConfig
  setConfig: (config: PositionConfig | ((prev: PositionConfig) => PositionConfig)) => void
  availableTokens: typeof POPULAR_TOKENS.base
  pools: Pool[]
  poolsLoading: boolean
  poolsError: string | null
}) {
  const formatNumber = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`
    }
    return `$${value.toFixed(2)}`
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Select Pool</h2>
        <p className="text-gray-300">Choose your token pair and fee tier</p>
      </div>

      {/* Network Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-200 mb-2">
          Network
        </label>
        <select 
          value={config.network.id}
          onChange={(e) => setConfig(prev => ({ 
            ...prev, 
            network: networks.find(n => n.id === e.target.value)! 
          }))}
          className="w-full bg-gray-700 border-2 border-white rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
        >
          {networks.map(network => (
            <option key={network.id} value={network.id}>
              {network.name}
            </option>
          ))}
        </select>
      </div>

      {/* Token Pair Selection */}
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Token A
          </label>
          <select 
            value={config.token0.symbol}
            onChange={(e) => setConfig(prev => ({ 
              ...prev, 
              token0: availableTokens.find(t => t.symbol === e.target.value)! 
            }))}
            className="w-full bg-gray-700 border-2 border-white rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {availableTokens.map(token => (
              <option key={token.symbol} value={token.symbol}>
                {token.symbol}
              </option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Token B
          </label>
          <select 
            value={config.token1.symbol}
            onChange={(e) => setConfig(prev => ({ 
              ...prev, 
              token1: availableTokens.find(t => t.symbol === e.target.value)! 
            }))}
            className="w-full bg-gray-700 border-2 border-white rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {availableTokens.filter(token => token.symbol !== config.token0.symbol).map(token => (
              <option key={token.symbol} value={token.symbol}>
                {token.symbol}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Loading State */}
      {poolsLoading && (
        <div className="text-center py-8">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-300">Loading pool data...</p>
        </div>
      )}

      {/* Error State */}
      {poolsError && (
        <div className="bg-red-900/20 border border-red-500 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">{poolsError}</span>
          </div>
        </div>
      )}

      {/* Fee Tier Selection */}
      {!poolsLoading && !poolsError && pools.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">Select Fee Tier</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {pools.map((pool) => {
              const apr = uniswapService.calculateAPR(pool.fees24h, pool.tvl)
              return (
                <button
                  key={pool.id}
                  onClick={() => setConfig(prev => ({ ...prev, feeTier: pool.feeTier }))}
                  className={`
                    p-4 rounded-xl border-2 text-left transition-all hover:scale-105 
                    ${config.feeTier === pool.feeTier
                      ? 'border-primary bg-primary/10 shadow-lg'
                      : 'border-white hover:border-primary/50'
                    }
                  `}
                >
                  <div className="text-2xl font-bold text-white mb-3">
                    {pool.feeTier}%
                  </div>
                  
                  <div className="space-y-2">
                    <div>
                      <div className="text-xs text-gray-400 uppercase tracking-wide">
                        Volume 24h
                      </div>
                      <div className="font-mono text-sm font-semibold text-gray-300">
                        {formatNumber(pool.volume24h)}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-xs text-gray-400 uppercase tracking-wide">
                        TVL
                      </div>
                      <div className="font-mono text-sm font-semibold text-gray-300">
                        {formatNumber(pool.tvl)}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-xs text-gray-400 uppercase tracking-wide">
                        APR
                      </div>
                      <div className="font-mono text-xs text-gray-400">
                        {(apr * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  
                  {config.feeTier === pool.feeTier && (
                    <div className="absolute top-2 right-2 w-3 h-3 bg-primary rounded-full"></div>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// Step 2: Configure Position
function ConfigurePositionStep({ 
  config, 
  setConfig, 
  holdings 
}: {
  config: PositionConfig
  setConfig: (config: PositionConfig | ((prev: PositionConfig) => PositionConfig)) => void
  holdings: any[]
}) {
  const [priceMode, setPriceMode] = useState<'full' | 'custom'>('full')

  const suggestedRanges = [
    { label: 'Narrow (±5%)', min: config.currentPrice * 0.95, max: config.currentPrice * 1.05 },
    { label: 'Medium (±15%)', min: config.currentPrice * 0.85, max: config.currentPrice * 1.15 },
    { label: 'Wide (±30%)', min: config.currentPrice * 0.7, max: config.currentPrice * 1.3 },
  ]

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Configure Position</h2>
        <p className="text-gray-300">Set your price range and deposit amounts</p>
      </div>

      {/* Current Price */}
      <div className="bg-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              {config.token0.symbol}/{config.token1.symbol}
            </h3>
            <p className="text-gray-300">Current Price</p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-white mono-numbers">
              ${config.currentPrice.toLocaleString()}
            </p>
            <p className="text-sm text-green-400">+2.45% (24h)</p>
          </div>
        </div>
      </div>

      {/* Price Range Selection */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Price Range</h3>
        
        {/* Range Mode Toggle */}
        <div className="flex space-x-2 mb-6">
          <button
            onClick={() => setPriceMode('full')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              priceMode === 'full' 
                ? 'bg-primary text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Full Range
          </button>
          <button
            onClick={() => setPriceMode('custom')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              priceMode === 'custom' 
                ? 'bg-primary text-white' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Custom Range
          </button>
        </div>

        {priceMode === 'custom' && (
          <>
            {/* Suggested Ranges */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              {suggestedRanges.map((range, index) => (
                <button
                  key={index}
                  onClick={() => setConfig(prev => ({ 
                    ...prev, 
                    minPrice: range.min.toString(), 
                    maxPrice: range.max.toString() 
                  }))}
                  className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg border-2 border-transparent hover:border-primary transition-colors"
                >
                  <div className="text-sm font-medium text-white mb-1">{range.label}</div>
                  <div className="text-xs text-gray-400">
                    ${range.min.toFixed(0)} - ${range.max.toFixed(0)}
                  </div>
                </button>
              ))}
            </div>

            {/* Custom Price Inputs */}
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-2">
                  Min Price
                </label>
                <input
                  type="number"
                  value={config.minPrice}
                  onChange={(e) => setConfig(prev => ({ ...prev, minPrice: e.target.value }))}
                  placeholder="0.00"
                  className="w-full px-4 py-3 border-2 border-white rounded-lg bg-gray-700 text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-200 mb-2">
                  Max Price
                </label>
                <input
                  type="number"
                  value={config.maxPrice}
                  onChange={(e) => setConfig(prev => ({ ...prev, maxPrice: e.target.value }))}
                  placeholder="0.00"
                  className="w-full px-4 py-3 border-2 border-white rounded-lg bg-gray-700 text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
                />
              </div>
            </div>
          </>
        )}

        {priceMode === 'full' && (
          <div className="bg-blue-900/20 border border-blue-500 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <Info className="w-5 h-5 text-blue-400" />
              <span className="text-blue-400">
                Full range selected (0 to ∞). Your position will always be active but earn lower fees.
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Deposit Amounts */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Deposit Amounts</h3>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              {config.token0.symbol} Amount
            </label>
            <input
              type="number"
              value={config.amount0}
              onChange={(e) => setConfig(prev => ({ ...prev, amount0: e.target.value }))}
              placeholder="0.00"
              className="w-full px-4 py-3 border-2 border-white rounded-lg bg-gray-700 text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
            />
            <p className="text-xs text-gray-400 mt-1">
              Balance: {holdings.find(h => h.symbol === config.token0.symbol)?.balance.toFixed(4) || '0.0000'}
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              {config.token1.symbol} Amount
            </label>
            <input
              type="number"
              value={config.amount1}
              onChange={(e) => setConfig(prev => ({ ...prev, amount1: e.target.value }))}
              placeholder="0.00"
              className="w-full px-4 py-3 border-2 border-white rounded-lg bg-gray-700 text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
            />
            <p className="text-xs text-gray-400 mt-1">
              Balance: {holdings.find(h => h.symbol === config.token1.symbol)?.balance.toFixed(4) || '0.0000'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Step 3: Review & Confirm
function ReviewConfirmStep({ 
  config, 
  pools 
}: {
  config: PositionConfig
  pools: Pool[]
}) {
  const selectedPool = pools.find(p => p.feeTier === config.feeTier)
  const totalValue = (parseFloat(config.amount0) || 0) * 2500 + (parseFloat(config.amount1) || 0)

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Review Position</h2>
        <p className="text-gray-300">Confirm your position details before deploying</p>
      </div>

      {/* Position Summary */}
      <div className="bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Position Summary</h3>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-gray-300 text-sm">Token Pair</p>
            <p className="text-xl font-bold text-white">
              {config.token0.symbol}/{config.token1.symbol}
            </p>
          </div>
          <div>
            <p className="text-gray-300 text-sm">Fee Tier</p>
            <p className="text-xl font-bold text-white">{config.feeTier}%</p>
          </div>
          <div>
            <p className="text-gray-300 text-sm">Network</p>
            <p className="text-xl font-bold text-white">{config.network.name}</p>
          </div>
          <div>
            <p className="text-gray-300 text-sm">Total Value</p>
            <p className="text-xl font-bold text-white">${totalValue.toFixed(2)}</p>
          </div>
        </div>
      </div>

      {/* Deposit Details */}
      <div className="bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Deposit Details</h3>
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">{config.token0.symbol} Deposit</span>
            <span className="text-white font-mono">{config.amount0 || '0.00'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-300">{config.token1.symbol} Deposit</span>
            <span className="text-white font-mono">{config.amount1 || '0.00'}</span>
          </div>
        </div>
      </div>

      {/* Price Range */}
      <div className="bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Price Range</h3>
        {config.minPrice && config.maxPrice ? (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Min Price</span>
              <span className="text-white font-mono">${config.minPrice}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Max Price</span>
              <span className="text-white font-mono">${config.maxPrice}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Current Price</span>
              <span className="text-white font-mono">${config.currentPrice.toLocaleString()}</span>
            </div>
          </div>
        ) : (
          <p className="text-gray-300">Full Range (0 to ∞)</p>
        )}
      </div>

      {/* Expected Returns */}
      {selectedPool && (
        <div className="bg-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Expected Returns</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-gray-300 text-sm">Estimated APR</p>
              <p className="text-xl font-bold text-green-400">
                {(uniswapService.calculateAPR(selectedPool.fees24h, selectedPool.tvl) * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-gray-300 text-sm">Pool TVL</p>
              <p className="text-xl font-bold text-white">
                ${(selectedPool.tvl / 1000000).toFixed(1)}M
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Step 4: Deploy Position
function DeployPositionStep({ 
  config, 
  isProcessing, 
  isComplete, 
  onDeploy 
}: {
  config: PositionConfig
  isProcessing: boolean
  isComplete: boolean
  onDeploy: () => void
}) {
  if (isComplete) {
    return (
      <div className="text-center space-y-6">
        <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto">
          <CheckCircle className="w-12 h-12 text-green-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Position Deployed!</h2>
          <p className="text-gray-300">
            Your Uniswap V4 position has been successfully created
          </p>
        </div>
        
        <div className="bg-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Transaction Details</h3>
          <div className="space-y-2 text-left">
            <div className="flex justify-between">
              <span className="text-gray-300">Transaction Hash</span>
              <span className="text-primary font-mono text-sm">0x1234...5678</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Gas Used</span>
              <span className="text-white">0.0045 ETH</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Position ID</span>
              <span className="text-white">#12345</span>
            </div>
          </div>
        </div>

        <div className="flex space-x-4 justify-center">
          <button className="flex items-center space-x-2 px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors">
            <ExternalLink className="w-5 h-5" />
            <span>View on Explorer</span>
          </button>
          <button className="flex items-center space-x-2 px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors">
            <Target className="w-5 h-5" />
            <span>Manage Position</span>
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-2">Deploy Position</h2>
        <p className="text-gray-300">Review final details and deploy your position</p>
      </div>

      {/* Final Summary */}
      <div className="bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Final Summary</h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-300">Pair</span>
            <span className="text-white font-semibold">
              {config.token0.symbol}/{config.token1.symbol}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Fee Tier</span>
            <span className="text-white font-semibold">{config.feeTier}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Deposit Value</span>
            <span className="text-white font-semibold">
              ${((parseFloat(config.amount0) || 0) * 2500 + (parseFloat(config.amount1) || 0)).toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Transaction Details */}
      <div className="bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Transaction Details</h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-300">Estimated Gas</span>
            <span className="text-white">~0.0045 ETH</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Network</span>
            <span className="text-white">{config.network.name}</span>
          </div>
        </div>
      </div>

      {/* Deploy Button */}
      <div className="text-center">
        {!isProcessing ? (
          <button
            onClick={onDeploy}
            className="flex items-center space-x-3 px-8 py-4 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors mx-auto"
          >
            <Zap className="w-6 h-6" />
            <span className="text-lg font-semibold">Deploy Position</span>
          </button>
        ) : (
          <div className="space-y-4">
            <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto"></div>
            <div>
              <p className="text-white font-semibold">Deploying Position...</p>
              <p className="text-gray-300 text-sm">Please confirm the transaction in your wallet</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}