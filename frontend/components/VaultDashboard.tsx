'use client'

import React, { useState, useEffect } from 'react'
import { useAccount, useReadContract, useWriteContract } from 'wagmi'
import { formatUnits, parseUnits } from 'viem'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Zap, 
  Settings, 
  Eye,
  PieChart,
  BarChart3,
  AlertTriangle,
  CheckCircle
} from 'lucide-react'

// Types
interface VaultConfig {
  mode: 'MANUAL' | 'AI_ASSISTED' | 'FULLY_AUTOMATED'
  positionType: 'SINGLE_RANGE' | 'DUAL_POSITION' | 'MULTI_RANGE' | 'AI_OPTIMIZED'
  aiOptimizationEnabled: boolean
  autoCompoundEnabled: boolean
  rebalanceThreshold: number
  maxSlippageBps: number
}

interface VaultMetrics {
  totalValueLocked: string
  totalFees24h: string
  impermanentLoss: string
  apr: string
  sharpeRatio: string
  maxDrawdown: string
  successfulCompounds: number
  aiOptimizationCount: number
}

interface PositionRange {
  id: number
  tickLower: number
  tickUpper: number
  allocation: number
  isActive: boolean
  liquidity: string
  name: string
}

interface StrategyRecommendation {
  strategyType: string
  confidenceScore: number
  expectedAPR: number
  expectedRisk: number
  reasoning: string
  positionRanges: Array<[number, number, number]>
}

interface VaultInfo {
  address: string
  name: string
  symbol: string
  token0: string
  token1: string
  fee: number
  totalShares: string
  userShares: string
}

export default function VaultDashboard({ vaultAddress }: { vaultAddress: string }) {
  const { address } = useAccount()
  const [selectedTab, setSelectedTab] = useState('overview')
  const [depositAmount, setDepositAmount] = useState('')
  const [withdrawAmount, setWithdrawAmount] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // Mock data for demonstration
  const mockVaultInfo: VaultInfo = {
    address: vaultAddress,
    name: 'ETH/USDC Vault',
    symbol: 'dETH-USDC',
    token0: 'ETH',
    token1: 'USDC',
    fee: 3000,
    totalShares: '1000000',
    userShares: '5000'
  }

  const mockMetrics: VaultMetrics = {
    totalValueLocked: '2500000',
    totalFees24h: '12500',
    impermanentLoss: '0.02',
    apr: '0.155',
    sharpeRatio: '1.24',
    maxDrawdown: '0.08',
    successfulCompounds: 47,
    aiOptimizationCount: 15
  }

  const mockRanges: PositionRange[] = [
    {
      id: 0,
      tickLower: -200,
      tickUpper: 200,
      allocation: 8000, // 80%
      isActive: true,
      liquidity: '500000',
      name: 'Base Position'
    },
    {
      id: 1, 
      tickLower: -100,
      tickUpper: 100,
      allocation: 2000, // 20%
      isActive: true,
      liquidity: '125000',
      name: 'Limit Position'
    }
  ]

  const mockRecommendation: StrategyRecommendation = {
    strategyType: 'AI_BALANCED',
    confidenceScore: 0.87,
    expectedAPR: 0.18,
    expectedRisk: 0.12,
    reasoning: 'Current market conditions favor balanced multi-range strategy with moderate risk exposure',
    positionRanges: [[-250, 250, 0.7], [-150, 150, 0.3]]
  }

  const handleDeposit = async () => {
    if (!depositAmount || !address) return
    setIsLoading(true)
    try {
      // Deposit logic would go here
      console.log('Depositing:', depositAmount)
    } catch (error) {
      console.error('Deposit failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleWithdraw = async () => {
    if (!withdrawAmount || !address) return
    setIsLoading(true)
    try {
      // Withdraw logic would go here
      console.log('Withdrawing:', withdrawAmount)
    } catch (error) {
      console.error('Withdraw failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleCompound = async () => {
    setIsLoading(true)
    try {
      // Compound logic would go here
      console.log('Compounding')
    } catch (error) {
      console.error('Compound failed:', error)
    } finally {
      setIsLoading(false)
    }
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

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'positions', label: 'Positions' },
    { id: 'strategy', label: 'Strategy' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'manage', label: 'Manage' }
  ]

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{mockVaultInfo.name}</h1>
            <p className="text-gray-600 dark:text-gray-400">
              {mockVaultInfo.token0}/{mockVaultInfo.token1} • {mockVaultInfo.fee/10000}% Fee
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
              <CheckCircle className="w-4 h-4 mr-1" />
              Active
            </span>
            <button className="inline-flex items-center px-4 py-2 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150">
              <Settings className="w-4 h-4 mr-2" />
              Configure
            </button>
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Value Locked</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {formatCurrency(mockMetrics.totalValueLocked)}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-blue-600" />
            </div>
          </div>

          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Current APR</p>
                <p className="text-2xl font-bold text-green-600">
                  {formatPercentage(mockMetrics.apr)}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
            <div className="mt-2">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                24h Fees: {formatCurrency(mockMetrics.totalFees24h)}
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-purple-600">
                  {parseFloat(mockMetrics.sharpeRatio).toFixed(2)}
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-600" />
            </div>
            <div className="mt-2">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Max Drawdown: {formatPercentage(mockMetrics.maxDrawdown)}
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">AI Optimizations</p>
                <p className="text-2xl font-bold text-orange-600">
                  {mockMetrics.aiOptimizationCount}
                </p>
              </div>
              <Zap className="w-8 h-8 text-orange-600" />
            </div>
            <div className="mt-2">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Compounds: {mockMetrics.successfulCompounds}
              </p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="space-y-6">
          <div className="border-b-2 border-black dark:border-white">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSelectedTab(tab.id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    selectedTab === tab.id
                      ? 'border-black dark:border-white text-black dark:text-white'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          {selectedTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Portfolio Composition */}
              <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center">
                  <PieChart className="w-5 h-5 mr-2" />
                  Portfolio Composition
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Your Shares</span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {(parseFloat(mockVaultInfo.userShares) / parseFloat(mockVaultInfo.totalShares) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 h-2">
                    <div 
                      className="bg-primary h-2" 
                      style={{ width: `${(parseFloat(mockVaultInfo.userShares) / parseFloat(mockVaultInfo.totalShares)) * 100}%` }}
                    ></div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Your Value</p>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {formatCurrency((parseFloat(mockMetrics.totalValueLocked) * parseFloat(mockVaultInfo.userShares) / parseFloat(mockVaultInfo.totalShares)).toString())}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Share Count</p>
                      <p className="font-medium text-gray-900 dark:text-white">{mockVaultInfo.userShares}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* AI Recommendation */}
              <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center">
                  <Zap className="w-5 h-5 mr-2" />
                  AI Recommendation
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center px-2 py-1 rounded border border-blue-500 text-blue-700 text-sm">
                      {mockRecommendation.strategyType.replace('_', ' ')}
                    </span>
                    <div className="text-right">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Confidence</p>
                      <p className="font-medium text-blue-600">
                        {(mockRecommendation.confidenceScore * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Expected APR</p>
                      <p className="font-medium text-green-600">
                        {formatPercentage(mockRecommendation.expectedAPR.toString())}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-600 dark:text-gray-400">Risk Level</p>
                      <p className="font-medium text-orange-600">
                        {formatPercentage(mockRecommendation.expectedRisk.toString())}
                      </p>
                    </div>
                  </div>
                  
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {mockRecommendation.reasoning}
                  </p>
                  
                  <button className="w-full bg-primary text-black px-4 py-2 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150">
                    Apply Recommendation
                  </button>
                </div>
              </div>
            </div>
          )}

          {selectedTab === 'manage' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Deposit */}
              <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Deposit</h3>
                <div className="space-y-4">
                  <input
                    type="number"
                    placeholder="Amount to deposit"
                    value={depositAmount}
                    onChange={(e) => setDepositAmount(e.target.value)}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  />
                  <button 
                    onClick={handleDeposit} 
                    disabled={isLoading || !depositAmount}
                    className="w-full bg-primary text-black px-4 py-2 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150 disabled:opacity-50"
                  >
                    {isLoading ? 'Depositing...' : 'Deposit'}
                  </button>
                </div>
              </div>

              {/* Withdraw */}
              <div className="bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Withdraw</h3>
                <div className="space-y-4">
                  <input
                    type="number"
                    placeholder="Amount to withdraw"
                    value={withdrawAmount}
                    onChange={(e) => setWithdrawAmount(e.target.value)}
                    className="w-full px-3 py-2 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white"
                  />
                  <button 
                    onClick={handleWithdraw} 
                    disabled={isLoading || !withdrawAmount}
                    className="w-full bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150 disabled:opacity-50"
                  >
                    {isLoading ? 'Withdrawing...' : 'Withdraw'}
                  </button>
                </div>
              </div>

              {/* Actions */}
              <div className="lg:col-span-2 bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal p-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Vault Actions</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <button 
                    onClick={handleCompound}
                    disabled={isLoading}
                    className="flex items-center justify-center bg-primary text-black px-4 py-2 border-2 border-black font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150 disabled:opacity-50"
                  >
                    <Zap className="w-4 h-4 mr-2" />
                    Compound
                  </button>
                  <button 
                    disabled={isLoading}
                    className="flex items-center justify-center bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150 disabled:opacity-50"
                  >
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Rebalance
                  </button>
                  <button 
                    className="flex items-center justify-center bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150"
                  >
                    <Eye className="w-4 h-4 mr-2" />
                    View Details
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}