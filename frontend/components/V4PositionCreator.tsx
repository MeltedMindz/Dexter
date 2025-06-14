'use client'

import React, { useState, useEffect } from 'react'
import { useAccount } from 'wagmi'
import { 
  TrendingUp, 
  Shield, 
  Zap, 
  ChevronRight, 
  Search,
  Filter,
  Star,
  AlertCircle,
  CheckCircle,
  DollarSign,
  BarChart3,
  Target,
  Wallet,
  ExternalLink
} from 'lucide-react'
import { usePoolSuggestions, PoolSuggestion } from '@/lib/hooks/usePoolSuggestions'
import { useWalletHoldings } from '@/lib/hooks/useWalletHoldings'

type CreateStep = 'suggestions' | 'configure' | 'confirm' | 'deploy'
type PoolFilter = 'ALL' | 'MY_TOKENS' | 'STABLECOIN' | 'ETH_PAIR' | 'HIGH_YIELD'

interface PositionConfig {
  pool: PoolSuggestion
  amount0: string
  amount1: string
  minPrice: string
  maxPrice: string
  autoCompound: boolean
  slippage: number
}

export function V4PositionCreator() {
  const { isConnected } = useAccount()
  const { suggestions, isLoading: suggestionsLoading } = usePoolSuggestions()
  const { tokens: holdings, totalValue } = useWalletHoldings()
  
  const [step, setStep] = useState<CreateStep>('suggestions')
  const [selectedPool, setSelectedPool] = useState<PoolSuggestion | null>(null)
  const [filter, setFilter] = useState<PoolFilter>('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [config, setConfig] = useState<PositionConfig | null>(null)

  // Filter suggestions based on user selections
  const filteredSuggestions = suggestions.filter(pool => {
    const matchesSearch = searchQuery === '' || 
      pool.token0.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      pool.token1.symbol.toLowerCase().includes(searchQuery.toLowerCase())
    
    switch (filter) {
      case 'MY_TOKENS':
        return matchesSearch && pool.userHoldings?.canCreatePosition
      case 'STABLECOIN':
        return matchesSearch && pool.category === 'STABLECOIN'
      case 'ETH_PAIR':
        return matchesSearch && pool.category === 'ETH_PAIR'
      case 'HIGH_YIELD':
        return matchesSearch && pool.estimatedApr > 25
      default:
        return matchesSearch
    }
  })

  if (!isConnected) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="text-center py-12">
          <Wallet className="w-16 h-16 text-slate-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
            Connect Your Wallet
          </h2>
          <p className="text-slate-600 dark:text-slate-400 mb-6">
            Connect your wallet to create Uniswap V4 positions and manage liquidity
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
          Create V4 Position
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Create and manage Uniswap V4 liquidity positions with AI-powered optimization
        </p>
      </div>

      {/* Progress Steps */}
      <div className="flex items-center justify-center mb-8">
        {[
          { key: 'suggestions', label: 'Pool Selection', icon: Target },
          { key: 'configure', label: 'Configure Position', icon: BarChart3 },
          { key: 'confirm', label: 'Review & Confirm', icon: CheckCircle },
          { key: 'deploy', label: 'Deploy Position', icon: Zap }
        ].map((stepItem, index) => {
          const isActive = step === stepItem.key
          const isCompleted = ['suggestions', 'configure', 'confirm'].indexOf(step) > index
          const Icon = stepItem.icon
          
          return (
            <React.Fragment key={stepItem.key}>
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                isActive 
                  ? 'bg-primary text-white' 
                  : isCompleted 
                    ? 'bg-green-500 text-white' 
                    : 'bg-slate-100 dark:bg-dark-600 text-slate-600 dark:text-slate-400'
              }`}>
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium">{stepItem.label}</span>
              </div>
              {index < 3 && (
                <ChevronRight className="w-4 h-4 text-slate-400 mx-2" />
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* Step Content */}
      {step === 'suggestions' && (
        <PoolSuggestionsStep
          suggestions={filteredSuggestions}
          filter={filter}
          setFilter={setFilter}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          isLoading={suggestionsLoading}
          onSelectPool={(pool) => {
            setSelectedPool(pool)
            setStep('configure')
          }}
          holdings={holdings}
          totalValue={totalValue}
        />
      )}

      {step === 'configure' && selectedPool && (
        <ConfigurePositionStep
          pool={selectedPool}
          holdings={holdings}
          onBack={() => setStep('suggestions')}
          onNext={(config) => {
            setConfig(config)
            setStep('confirm')
          }}
        />
      )}

      {step === 'confirm' && config && (
        <ConfirmPositionStep
          config={config}
          onBack={() => setStep('configure')}
          onConfirm={() => setStep('deploy')}
        />
      )}

      {step === 'deploy' && config && (
        <DeployPositionStep
          config={config}
          onComplete={() => {
            // Reset or redirect to dashboard
            setStep('suggestions')
            setSelectedPool(null)
            setConfig(null)
          }}
        />
      )}
    </div>
  )
}

// Pool Suggestions Step Component
function PoolSuggestionsStep({
  suggestions,
  filter,
  setFilter,
  searchQuery,
  setSearchQuery,
  isLoading,
  onSelectPool,
  holdings,
  totalValue
}: {
  suggestions: PoolSuggestion[]
  filter: PoolFilter
  setFilter: (filter: PoolFilter) => void
  searchQuery: string
  setSearchQuery: (query: string) => void
  isLoading: boolean
  onSelectPool: (pool: PoolSuggestion) => void
  holdings: any[]
  totalValue: number
}) {
  const filters: { key: PoolFilter; label: string; icon: React.ElementType }[] = [
    { key: 'ALL', label: 'All Pools', icon: BarChart3 },
    { key: 'MY_TOKENS', label: 'My Holdings', icon: Wallet },
    { key: 'STABLECOIN', label: 'Stablecoins', icon: Shield },
    { key: 'ETH_PAIR', label: 'ETH Pairs', icon: TrendingUp },
    { key: 'HIGH_YIELD', label: 'High Yield', icon: Zap }
  ]

  return (
    <div className="space-y-6">
      {/* Wallet Summary */}
      <div className="bg-slate-50 dark:bg-dark-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
          Your Wallet
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400">Total Value</p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
              ${totalValue.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400">Tokens</p>
            <div className="flex items-center space-x-2 mt-1">
              {holdings.slice(0, 4).map((token) => (
                <div key={token.symbol} className="flex items-center space-x-1">
                  <div className="w-6 h-6 bg-gradient-to-br from-primary to-primary-600 rounded-full"></div>
                  <span className="text-sm font-medium text-slate-900 dark:text-white">
                    {token.symbol}
                  </span>
                </div>
              ))}
              {holdings.length > 4 && (
                <span className="text-sm text-slate-500">+{holdings.length - 4} more</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search pools by token..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-slate-200 dark:border-white/10 rounded-lg bg-white dark:bg-dark-600 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>
        <div className="flex space-x-2">
          {filters.map((filterItem) => {
            const Icon = filterItem.icon
            const isActive = filter === filterItem.key
            
            return (
              <button
                key={filterItem.key}
                onClick={() => setFilter(filterItem.key)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-primary text-white'
                    : 'bg-slate-100 dark:bg-dark-600 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-dark-500'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium">{filterItem.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Pool Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {isLoading ? (
          Array(6).fill(0).map((_, i) => (
            <div key={i} className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10 animate-pulse">
              <div className="h-6 bg-slate-200 dark:bg-dark-600 rounded mb-4"></div>
              <div className="h-4 bg-slate-200 dark:bg-dark-600 rounded mb-2"></div>
              <div className="h-4 bg-slate-200 dark:bg-dark-600 rounded w-2/3"></div>
            </div>
          ))
        ) : suggestions.length === 0 ? (
          <div className="col-span-full text-center py-12">
            <Filter className="w-12 h-12 text-slate-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
              No pools found
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Try adjusting your filters or search query
            </p>
          </div>
        ) : (
          suggestions.map((pool) => (
            <PoolCard
              key={pool.id}
              pool={pool}
              onClick={() => onSelectPool(pool)}
            />
          ))
        )}
      </div>
    </div>
  )
}

// Pool Card Component
function PoolCard({ pool, onClick }: { pool: PoolSuggestion; onClick: () => void }) {
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return 'text-green-600 bg-green-50 dark:bg-green-900/20'
      case 'MEDIUM': return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20'
      case 'HIGH': return 'text-red-600 bg-red-50 dark:bg-red-900/20'
      default: return 'text-slate-600 bg-slate-50 dark:bg-slate-900/20'
    }
  }

  return (
    <div 
      onClick={onClick}
      className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10 hover:border-primary/50 transition-all cursor-pointer group"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex items-center -space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary-600 rounded-full border-2 border-white dark:border-dark-700"></div>
            <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full border-2 border-white dark:border-dark-700"></div>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
              {pool.token0.symbol}/{pool.token1.symbol}
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              {pool.feeTier}% Fee Tier
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Star className={`w-4 h-4 ${pool.reasonScore > 70 ? 'text-yellow-500 fill-current' : 'text-slate-300'}`} />
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(pool.riskLevel)}`}>
            {pool.riskLevel}
          </span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-sm text-slate-600 dark:text-slate-400">Est. APR</p>
          <p className="text-xl font-bold text-green-600 mono-numbers">
            {pool.estimatedApr.toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-sm text-slate-600 dark:text-slate-400">24h Volume</p>
          <p className="text-lg font-semibold text-slate-900 dark:text-white mono-numbers">
            ${(pool.volume24h / 1000000).toFixed(1)}M
          </p>
        </div>
      </div>

      {/* User Holdings Status */}
      {pool.userHoldings?.canCreatePosition ? (
        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3 mb-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-800 dark:text-green-300">
              Ready to create position
            </span>
          </div>
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">
            Suggested: ${pool.userHoldings.suggestedAllocation.toLocaleString()}
          </p>
        </div>
      ) : pool.userHoldings?.token0Balance || pool.userHoldings?.token1Balance ? (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mb-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-800 dark:text-blue-300">
              You hold one token
            </span>
          </div>
        </div>
      ) : null}

      {/* Reasons */}
      <div className="space-y-1 mb-4">
        {pool.reasons.slice(0, 2).map((reason, index) => (
          <div key={index} className="flex items-center space-x-2">
            <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
            <span className="text-sm text-slate-600 dark:text-slate-400">{reason}</span>
          </div>
        ))}
      </div>

      {/* Action Button */}
      <button className="w-full bg-primary text-white py-2 rounded-lg font-medium hover:bg-primary/90 transition-colors group-hover:bg-primary/80">
        Create Position
      </button>
    </div>
  )
}

// Configure Position Step
function ConfigurePositionStep({ 
  pool, 
  holdings, 
  onBack, 
  onNext 
}: {
  pool: PoolSuggestion
  holdings: any[]
  onBack: () => void
  onNext: (config: PositionConfig) => void
}) {
  const [amount0, setAmount0] = useState('')
  const [amount1, setAmount1] = useState('')
  const [priceRange, setPriceRange] = useState({ min: '', max: '' })
  const [autoCompound, setAutoCompound] = useState(true)
  const [slippage, setSlippage] = useState(0.5)
  
  const token0Holding = holdings.find(h => h.symbol === pool.token0.symbol)
  const token1Holding = holdings.find(h => h.symbol === pool.token1.symbol)

  const currentPrice = 2000 // Mock current price
  const suggestedRange = {
    min: currentPrice * 0.8,
    max: currentPrice * 1.2
  }

  useEffect(() => {
    setPriceRange({
      min: suggestedRange.min.toString(),
      max: suggestedRange.max.toString()
    })
  }, [suggestedRange.min, suggestedRange.max])

  const handleNext = () => {
    const config: PositionConfig = {
      pool,
      amount0,
      amount1,
      minPrice: priceRange.min,
      maxPrice: priceRange.max,
      autoCompound,
      slippage
    }
    onNext(config)
  }

  const isValid = amount0 && amount1 && priceRange.min && priceRange.max

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
        >
          <ChevronRight className="w-4 h-4 rotate-180" />
          <span>Back to pools</span>
        </button>
      </div>

      {/* Selected Pool Info */}
      <div className="bg-slate-50 dark:bg-dark-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
          Selected Pool
        </h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center -space-x-2">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary-600 rounded-full border-2 border-white dark:border-dark-700"></div>
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full border-2 border-white dark:border-dark-700"></div>
          </div>
          <div>
            <h4 className="text-xl font-bold text-slate-900 dark:text-white">
              {pool.token0.symbol}/{pool.token1.symbol}
            </h4>
            <p className="text-slate-600 dark:text-slate-400">
              {pool.feeTier}% • Est. APR {pool.estimatedApr.toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column - Token Amounts */}
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
            Deposit Amounts
          </h3>

          {/* Token 0 Input */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                {pool.token0.symbol} Amount
              </label>
              <span className="text-sm text-slate-500">
                Balance: {token0Holding?.balanceFormatted || '0.00'}
              </span>
            </div>
            <div className="relative">
              <input
                type="number"
                value={amount0}
                onChange={(e) => setAmount0(e.target.value)}
                placeholder="0.00"
                className="w-full px-4 py-3 border border-slate-200 dark:border-white/10 rounded-lg bg-white dark:bg-dark-600 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers text-lg"
              />
              <button className="absolute right-3 top-1/2 transform -translate-y-1/2 text-primary text-sm font-medium hover:text-primary/80">
                MAX
              </button>
            </div>
          </div>

          {/* Token 1 Input */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                {pool.token1.symbol} Amount
              </label>
              <span className="text-sm text-slate-500">
                Balance: {token1Holding?.balanceFormatted || '0.00'}
              </span>
            </div>
            <div className="relative">
              <input
                type="number"
                value={amount1}
                onChange={(e) => setAmount1(e.target.value)}
                placeholder="0.00"
                className="w-full px-4 py-3 border border-slate-200 dark:border-white/10 rounded-lg bg-white dark:bg-dark-600 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers text-lg"
              />
              <button className="absolute right-3 top-1/2 transform -translate-y-1/2 text-primary text-sm font-medium hover:text-primary/80">
                MAX
              </button>
            </div>
          </div>
        </div>

        {/* Right Column - Price Range & Settings */}
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
            Price Range
          </h3>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <AlertCircle className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-800 dark:text-blue-300">
                Current Price: ${currentPrice.toLocaleString()}
              </span>
            </div>
            <p className="text-xs text-blue-600 dark:text-blue-400">
              Set your price range carefully. Tokens will only earn fees when the price is within this range.
            </p>
          </div>

          {/* Min Price */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Minimum Price
            </label>
            <input
              type="number"
              value={priceRange.min}
              onChange={(e) => setPriceRange({ ...priceRange, min: e.target.value })}
              placeholder="0.00"
              className="w-full px-4 py-3 border border-slate-200 dark:border-white/10 rounded-lg bg-white dark:bg-dark-600 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
            />
          </div>

          {/* Max Price */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Maximum Price
            </label>
            <input
              type="number"
              value={priceRange.max}
              onChange={(e) => setPriceRange({ ...priceRange, max: e.target.value })}
              placeholder="0.00"
              className="w-full px-4 py-3 border border-slate-200 dark:border-white/10 rounded-lg bg-white dark:bg-dark-600 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
            />
          </div>

          {/* Auto-compound */}
          <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-dark-600 rounded-lg">
            <div>
              <h4 className="font-medium text-slate-900 dark:text-white">Auto-compound</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Automatically reinvest earned fees
              </p>
            </div>
            <button
              onClick={() => setAutoCompound(!autoCompound)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                autoCompound ? 'bg-primary' : 'bg-slate-300 dark:bg-slate-600'
              }`}
            >
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                autoCompound ? 'translate-x-7' : 'translate-x-1'
              }`} />
            </button>
          </div>

          {/* Slippage */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Slippage Tolerance
            </label>
            <div className="flex space-x-2">
              {[0.1, 0.5, 1.0].map((value) => (
                <button
                  key={value}
                  onClick={() => setSlippage(value)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    slippage === value
                      ? 'bg-primary text-white'
                      : 'bg-slate-100 dark:bg-dark-600 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-dark-500'
                  }`}
                >
                  {value}%
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <button
          onClick={onBack}
          className="flex-1 bg-slate-100 dark:bg-dark-600 text-slate-900 dark:text-white py-3 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-dark-500 transition-colors"
        >
          Back
        </button>
        <button
          onClick={handleNext}
          disabled={!isValid}
          className="flex-1 bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Review Position
        </button>
      </div>
    </div>
  )
}

// Confirm Position Step
function ConfirmPositionStep({ 
  config, 
  onBack, 
  onConfirm 
}: {
  config: PositionConfig
  onBack: () => void
  onConfirm: () => void
}) {
  const estimatedValue = (parseFloat(config.amount0) * 2500) + (parseFloat(config.amount1) * 1) // Mock calculation
  const estimatedFees = estimatedValue * (config.pool.estimatedApr / 100) / 365 // Daily fees

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
        >
          <ChevronRight className="w-4 h-4 rotate-180" />
          <span>Back to configure</span>
        </button>
      </div>

      {/* Position Summary */}
      <div className="bg-white dark:bg-dark-700 border border-slate-200 dark:border-white/10 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
          Position Summary
        </h3>

        <div className="space-y-6">
          {/* Pool Info */}
          <div className="flex items-center justify-between">
            <span className="text-slate-600 dark:text-slate-400">Pool</span>
            <div className="flex items-center space-x-2">
              <div className="flex items-center -space-x-1">
                <div className="w-6 h-6 bg-gradient-to-br from-primary to-primary-600 rounded-full"></div>
                <div className="w-6 h-6 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full"></div>
              </div>
              <span className="font-medium text-slate-900 dark:text-white">
                {config.pool.token0.symbol}/{config.pool.token1.symbol}
              </span>
            </div>
          </div>

          {/* Deposit Amounts */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">{config.pool.token0.symbol} Deposit</span>
              <span className="font-medium text-slate-900 dark:text-white mono-numbers">
                {parseFloat(config.amount0).toFixed(4)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">{config.pool.token1.symbol} Deposit</span>
              <span className="font-medium text-slate-900 dark:text-white mono-numbers">
                {parseFloat(config.amount1).toFixed(2)}
              </span>
            </div>
          </div>

          {/* Price Range */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Min Price</span>
              <span className="font-medium text-slate-900 dark:text-white mono-numbers">
                ${parseFloat(config.minPrice).toLocaleString()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Max Price</span>
              <span className="font-medium text-slate-900 dark:text-white mono-numbers">
                ${parseFloat(config.maxPrice).toLocaleString()}
              </span>
            </div>
          </div>

          {/* Estimates */}
          <div className="border-t border-slate-200 dark:border-white/10 pt-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Total Value</span>
              <span className="font-semibold text-slate-900 dark:text-white mono-numbers">
                ${estimatedValue.toLocaleString()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Est. Daily Fees</span>
              <span className="font-medium text-green-600 mono-numbers">
                ${estimatedFees.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Est. APR</span>
              <span className="font-medium text-green-600 mono-numbers">
                {config.pool.estimatedApr.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Settings */}
          <div className="border-t border-slate-200 dark:border-white/10 pt-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Auto-compound</span>
              <span className="font-medium text-slate-900 dark:text-white">
                {config.autoCompound ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Slippage</span>
              <span className="font-medium text-slate-900 dark:text-white mono-numbers">
                {config.slippage}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Important Notice */}
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
          <div>
            <h4 className="font-medium text-yellow-800 dark:text-yellow-300 mb-1">
              Important
            </h4>
            <ul className="text-sm text-yellow-700 dark:text-yellow-400 space-y-1">
              <li>• You will only earn fees when the price is within your specified range</li>
              <li>• This transaction will require multiple signatures for token approvals</li>
              <li>• Gas fees will be paid separately for the position creation</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <button
          onClick={onBack}
          className="flex-1 bg-slate-100 dark:bg-dark-600 text-slate-900 dark:text-white py-3 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-dark-500 transition-colors"
        >
          Back
        </button>
        <button
          onClick={onConfirm}
          className="flex-1 bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors"
        >
          Create Position
        </button>
      </div>
    </div>
  )
}

// Deploy Position Step
function DeployPositionStep({ 
  config, 
  onComplete 
}: {
  config: PositionConfig
  onComplete: () => void
}) {
  const [step, setStep] = useState<'approving' | 'creating' | 'success' | 'error'>('approving')
  const [txHash, setTxHash] = useState<string>('')

  useEffect(() => {
    // Simulate the deployment process
    const deployPosition = async () => {
      try {
        // Step 1: Token approvals
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        // Step 2: Create position
        setStep('creating')
        await new Promise(resolve => setTimeout(resolve, 3000))
        
        // Step 3: Success
        setTxHash('0x1234567890abcdef1234567890abcdef12345678')
        setStep('success')
      } catch (error) {
        setStep('error')
      }
    }

    deployPosition()
  }, [])

  if (step === 'success') {
    return (
      <div className="text-center space-y-8">
        <div className="w-20 h-20 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto">
          <CheckCircle className="w-10 h-10 text-green-600" />
        </div>
        
        <div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
            Position Created Successfully!
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Your Uniswap V4 liquidity position has been deployed
          </p>
        </div>

        <div className="bg-slate-50 dark:bg-dark-700 rounded-xl p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Transaction Hash</span>
              <div className="flex items-center space-x-2">
                <span className="font-mono text-sm text-slate-900 dark:text-white">
                  {txHash.slice(0, 10)}...{txHash.slice(-8)}
                </span>
                <button className="text-primary hover:text-primary/80">
                  <ExternalLink className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Pool</span>
              <span className="font-medium text-slate-900 dark:text-white">
                {config.pool.token0.symbol}/{config.pool.token1.symbol}
              </span>
            </div>
          </div>
        </div>

        <div className="flex space-x-4">
          <button
            onClick={() => window.open(`https://basescan.org/tx/${txHash}`, '_blank')}
            className="flex-1 bg-slate-100 dark:bg-dark-600 text-slate-900 dark:text-white py-3 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-dark-500 transition-colors"
          >
            View Transaction
          </button>
          <button
            onClick={onComplete}
            className="flex-1 bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors"
          >
            Go to Dashboard
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="text-center space-y-8">
      <div className="w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
      </div>
      
      <div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
          {step === 'approving' ? 'Approving Tokens' : 'Creating Position'}
        </h2>
        <p className="text-slate-600 dark:text-slate-400">
          {step === 'approving' 
            ? 'Please approve token spending in your wallet...'
            : 'Deploying your liquidity position to Uniswap V4...'
          }
        </p>
      </div>

      <div className="bg-slate-50 dark:bg-dark-700 rounded-xl p-6">
        <div className="space-y-4">
          <div className={`flex items-center space-x-3 ${step === 'approving' ? 'text-primary' : 'text-green-600'}`}>
            {step === 'approving' ? (
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <CheckCircle className="w-4 h-4" />
            )}
            <span className="font-medium">Token Approvals</span>
          </div>
          <div className={`flex items-center space-x-3 ${
            step === 'creating' ? 'text-primary' : step === 'approving' ? 'text-slate-400' : 'text-green-600'
          }`}>
            {step === 'creating' ? (
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            ) : step === 'approving' ? (
              <div className="w-4 h-4 border-2 border-slate-300 rounded-full"></div>
            ) : (
              <CheckCircle className="w-4 h-4" />
            )}
            <span className="font-medium">Position Creation</span>
          </div>
        </div>
      </div>
    </div>
  )
}