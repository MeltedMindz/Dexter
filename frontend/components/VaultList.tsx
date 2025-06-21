'use client'

import React, { useState, useEffect } from 'react'
import { useAccount, useReadContract } from 'wagmi'
import { formatUnits } from 'viem'
import { 
  Search, 
  Filter, 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  Zap,
  Eye,
  Star,
  StarOff,
  BarChart3,
  Users,
  Clock,
  ArrowUpRight
} from 'lucide-react'
import Link from 'next/link'

// Types
interface VaultInfo {
  address: string
  name: string
  symbol: string
  token0Symbol: string
  token1Symbol: string
  feeTier: number
  templateType: string
  creator: string
  createdAt: number
  totalValueLocked: string
  totalShares: string
  userShares?: string
  metrics: {
    apr: string
    sharpeRatio: string
    totalFees24h: string
    impermanentLoss: string
    successfulCompounds: number
    aiOptimizationCount: number
  }
  isAIEnabled: boolean
  riskLevel: 'Low' | 'Medium' | 'High'
}

const MOCK_VAULTS: VaultInfo[] = [
  {
    address: '0x123...abc',
    name: 'ETH/USDC Premium',
    symbol: 'dETH-USDC',
    token0Symbol: 'ETH',
    token1Symbol: 'USDC',
    feeTier: 3000,
    templateType: 'AI_OPTIMIZED',
    creator: '0xdef...456',
    createdAt: Date.now() - 86400000 * 30, // 30 days ago
    totalValueLocked: '2500000',
    totalShares: '1000000',
    userShares: '5000',
    metrics: {
      apr: '0.155',
      sharpeRatio: '1.24',
      totalFees24h: '12500',
      impermanentLoss: '0.02',
      successfulCompounds: 47,
      aiOptimizationCount: 15
    },
    isAIEnabled: true,
    riskLevel: 'Medium'
  },
  {
    address: '0x456...def',
    name: 'USDC/USDT Stable',
    symbol: 'dUSDC-USDT',
    token0Symbol: 'USDC',
    token1Symbol: 'USDT',
    feeTier: 100,
    templateType: 'BASIC',
    creator: '0x789...ghi',
    createdAt: Date.now() - 86400000 * 15, // 15 days ago
    totalValueLocked: '5000000',
    totalShares: '2000000',
    metrics: {
      apr: '0.08',
      sharpeRatio: '2.1',
      totalFees24h: '8000',
      impermanentLoss: '0.001',
      successfulCompounds: 28,
      aiOptimizationCount: 0
    },
    isAIEnabled: false,
    riskLevel: 'Low'
  },
  {
    address: '0x789...ghi',
    name: 'WBTC/ETH Gamma',
    symbol: 'dWBTC-ETH',
    token0Symbol: 'WBTC',
    token1Symbol: 'ETH',
    feeTier: 3000,
    templateType: 'GAMMA_STYLE',
    creator: '0xabc...123',
    createdAt: Date.now() - 86400000 * 7, // 7 days ago
    totalValueLocked: '1800000',
    totalShares: '750000',
    userShares: '2500',
    metrics: {
      apr: '0.185',
      sharpeRatio: '1.45',
      totalFees24h: '15000',
      impermanentLoss: '0.035',
      successfulCompounds: 35,
      aiOptimizationCount: 8
    },
    isAIEnabled: true,
    riskLevel: 'High'
  },
  {
    address: '0xabc...123',
    name: 'Institutional Multi',
    symbol: 'dINST',
    token0Symbol: 'ETH',
    token1Symbol: 'USDC',
    feeTier: 500,
    templateType: 'INSTITUTIONAL',
    creator: '0x321...cba',
    createdAt: Date.now() - 86400000 * 45, // 45 days ago
    totalValueLocked: '12000000',
    totalShares: '5000000',
    metrics: {
      apr: '0.12',
      sharpeRatio: '1.8',
      totalFees24h: '25000',
      impermanentLoss: '0.015',
      successfulCompounds: 92,
      aiOptimizationCount: 35
    },
    isAIEnabled: true,
    riskLevel: 'Low'
  }
]

export default function VaultList() {
  const { address } = useAccount()
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState('tvl')
  const [filterBy, setFilterBy] = useState('all')
  const [showOnlyMyVaults, setShowOnlyMyVaults] = useState(false)
  const [favoriteVaults, setFavoriteVaults] = useState<Set<string>>(new Set())

  // Filter and sort vaults
  const filteredVaults = MOCK_VAULTS
    .filter(vault => {
      // Search filter
      if (searchTerm) {
        const searchLower = searchTerm.toLowerCase()
        if (!vault.name.toLowerCase().includes(searchLower) &&
            !vault.token0Symbol.toLowerCase().includes(searchLower) &&
            !vault.token1Symbol.toLowerCase().includes(searchLower)) {
          return false
        }
      }

      // Template filter
      if (filterBy !== 'all' && vault.templateType !== filterBy.toUpperCase()) {
        return false
      }

      // My vaults filter
      if (showOnlyMyVaults && !vault.userShares) {
        return false
      }

      return true
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'tvl':
          return parseFloat(b.totalValueLocked) - parseFloat(a.totalValueLocked)
        case 'apr':
          return parseFloat(b.metrics.apr) - parseFloat(a.metrics.apr)
        case 'age':
          return b.createdAt - a.createdAt
        case 'volume':
          return parseFloat(b.metrics.totalFees24h) - parseFloat(a.metrics.totalFees24h)
        default:
          return 0
      }
    })

  const toggleFavorite = (vaultAddress: string) => {
    const newFavorites = new Set(favoriteVaults)
    if (newFavorites.has(vaultAddress)) {
      newFavorites.delete(vaultAddress)
    } else {
      newFavorites.add(vaultAddress)
    }
    setFavoriteVaults(newFavorites)
  }

  const formatCurrency = (amount: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(parseFloat(amount))
  }

  const formatPercentage = (value: string) => {
    return `${(parseFloat(value) * 100).toFixed(2)}%`
  }

  const formatTimeAgo = (timestamp: number) => {
    const days = Math.floor((Date.now() - timestamp) / (1000 * 60 * 60 * 24))
    return `${days} days ago`
  }

  const getTemplateDisplayName = (templateType: string) => {
    const names: { [key: string]: string } = {
      'BASIC': 'Basic',
      'GAMMA_STYLE': 'Gamma Style',
      'AI_OPTIMIZED': 'AI Optimized',
      'INSTITUTIONAL': 'Institutional'
    }
    return names[templateType] || templateType
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'Medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'High': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Vault Explorer</h1>
            <p className="text-gray-600 dark:text-gray-400">
              Discover and invest in automated liquidity management vaults
            </p>
          </div>
          <Link href="/vaults/create">
            <button className="inline-flex items-center bg-primary text-black px-4 py-2 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150">
              <DollarSign className="w-4 h-4 mr-2" />
              Create Vault
            </button>
          </Link>
        </div>

        {/* Filters and Search */}
        <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
                Search Vaults
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  placeholder="Search by name or tokens..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                />
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
                Filter by Template
              </label>
              <select 
                value={filterBy} 
                onChange={(e) => setFilterBy(e.target.value)}
                className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
              >
                <option value="all">All Templates</option>
                <option value="basic">Basic</option>
                <option value="gamma_style">Gamma Style</option>
                <option value="ai_optimized">AI Optimized</option>
                <option value="institutional">Institutional</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
                Sort by
              </label>
              <select 
                value={sortBy} 
                onChange={(e) => setSortBy(e.target.value)}
                className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
              >
                <option value="tvl">Total Value Locked</option>
                <option value="apr">APR</option>
                <option value="volume">24h Volume</option>
                <option value="age">Recently Created</option>
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="myVaults"
                checked={showOnlyMyVaults}
                onChange={(e) => setShowOnlyMyVaults(e.target.checked)}
                className="rounded border-gray-300"
              />
              <label htmlFor="myVaults" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                My Vaults Only
              </label>
            </div>
          </div>
        </div>

        {/* Vault Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Vaults</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{MOCK_VAULTS.length}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-600" />
            </div>
          </div>

          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total TVL</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatCurrency(MOCK_VAULTS.reduce((sum, v) => sum + parseFloat(v.totalValueLocked), 0).toString())}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-green-600" />
            </div>
          </div>

          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg APR</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatPercentage((MOCK_VAULTS.reduce((sum, v) => sum + parseFloat(v.metrics.apr), 0) / MOCK_VAULTS.length).toString())}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-600" />
            </div>
          </div>

          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">AI Vaults</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {MOCK_VAULTS.filter(v => v.isAIEnabled).length}
                </p>
              </div>
              <Zap className="w-8 h-8 text-orange-600" />
            </div>
          </div>
        </div>

        {/* Vault List */}
        <div className="space-y-4">
          {filteredVaults.length === 0 ? (
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-12 text-center">
              <Search className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No vaults found</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Try adjusting your search criteria or create a new vault.
              </p>
            </div>
          ) : (
            filteredVaults.map((vault) => (
              <div key={vault.address} className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150 p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {/* Vault Info */}
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{vault.name}</h3>
                        <span className="inline-flex items-center px-2 py-1 rounded border border-gray-500 text-gray-700 dark:text-gray-300 text-sm">
                          {vault.symbol}
                        </span>
                        <span className={`inline-flex items-center px-2 py-1 rounded text-sm ${getRiskColor(vault.riskLevel)}`}>
                          {vault.riskLevel} Risk
                        </span>
                        {vault.isAIEnabled && (
                          <span className="inline-flex items-center px-2 py-1 rounded border border-purple-500 text-purple-700 dark:text-purple-300 text-sm">
                            <Zap className="w-3 h-3 mr-1" />
                            AI
                          </span>
                        )}
                        {vault.userShares && (
                          <span className="inline-flex items-center px-2 py-1 rounded border border-blue-500 text-blue-700 dark:text-blue-300 text-sm">
                            <Users className="w-3 h-3 mr-1" />
                            My Vault
                          </span>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-6 text-sm text-gray-600 dark:text-gray-400">
                        <div className="flex items-center space-x-1">
                          <span className="font-medium">{vault.token0Symbol}/{vault.token1Symbol}</span>
                          <span>•</span>
                          <span>{vault.feeTier / 10000}% Fee</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <span>{getTemplateDisplayName(vault.templateType)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Clock className="w-3 h-3" />
                          <span>{formatTimeAgo(vault.createdAt)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">TVL</p>
                        <p className="text-lg font-semibold text-gray-900 dark:text-white">
                          {formatCurrency(vault.totalValueLocked)}
                        </p>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">APR</p>
                        <p className="text-lg font-semibold text-green-600">
                          {formatPercentage(vault.metrics.apr)}
                        </p>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">24h Fees</p>
                        <p className="text-lg font-semibold text-blue-600">
                          {formatCurrency(vault.metrics.totalFees24h)}
                        </p>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
                        <p className="text-lg font-semibold text-purple-600">
                          {parseFloat(vault.metrics.sharpeRatio).toFixed(2)}
                        </p>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => toggleFavorite(vault.address)}
                        className="p-2 border-2 border-black dark:border-white bg-white dark:bg-black hover:bg-gray-100 dark:hover:bg-gray-900 transition-colors"
                      >
                        {favoriteVaults.has(vault.address) ? (
                          <Star className="w-4 h-4 text-yellow-500 fill-current" />
                        ) : (
                          <StarOff className="w-4 h-4 text-gray-400" />
                        )}
                      </button>
                      
                      <Link href={`/vaults/${vault.address}`}>
                        <button className="inline-flex items-center bg-primary text-black px-4 py-2 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150">
                          <Eye className="w-4 h-4 mr-2" />
                          View
                        </button>
                      </Link>
                    </div>
                  </div>
                </div>

                {/* User Position (if applicable) */}
                {vault.userShares && (
                  <div className="mt-4 pt-4 border-t-2 border-black dark:border-white">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-4">
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Your Shares: </span>
                          <span className="font-medium text-gray-900 dark:text-white">{vault.userShares}</span>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Your Value: </span>
                          <span className="font-medium text-green-600">
                            {formatCurrency((parseFloat(vault.totalValueLocked) * parseFloat(vault.userShares) / parseFloat(vault.totalShares)).toString())}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Share: </span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {(parseFloat(vault.userShares) / parseFloat(vault.totalShares) * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                      <Link href={`/vaults/${vault.address}`}>
                        <button className="inline-flex items-center bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150">
                          <ArrowUpRight className="w-4 h-4 mr-1" />
                          Manage
                        </button>
                      </Link>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Load More */}
        {filteredVaults.length >= 10 && (
          <div className="text-center">
            <button className="bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150">
              Load More Vaults
            </button>
          </div>
        )}
      </div>
    </div>
  )
}