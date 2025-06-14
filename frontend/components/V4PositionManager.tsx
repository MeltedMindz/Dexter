'use client'

import React, { useState } from 'react'
import { useAccount } from 'wagmi'
import { 
  Plus,
  Minus,
  DollarSign,
  Trash2,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Settings,
  AlertCircle,
  CheckCircle,
  Clock,
  ExternalLink,
  Zap,
  Target
} from 'lucide-react'

interface Position {
  id: string
  tokenId: number
  pool: {
    token0: { symbol: string; logoURI: string }
    token1: { symbol: string; logoURI: string }
    feeTier: number
  }
  liquidity: string
  amount0: string
  amount1: string
  currentValue: number
  priceRange: {
    min: number
    max: number
    current: number
  }
  inRange: boolean
  feesEarned: {
    amount0: string
    amount1: string
    usdValue: number
  }
  performance: {
    apy: number
    change24h: number
    totalReturn: number
  }
  autoCompound: boolean
  createdAt: string
}

// Mock positions data
const MOCK_POSITIONS: Position[] = [
  {
    id: 'pos-1',
    tokenId: 12345,
    pool: {
      token0: { symbol: 'ETH', logoURI: 'https://ethereum-optimism.github.io/data/ETH/logo.png' },
      token1: { symbol: 'USDC', logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png' },
      feeTier: 0.05
    },
    liquidity: '5000000000000000000',
    amount0: '2.5',
    amount1: '6250.00',
    currentValue: 12500,
    priceRange: { min: 2000, max: 3000, current: 2500 },
    inRange: true,
    feesEarned: {
      amount0: '0.125',
      amount1: '312.50',
      usdValue: 625
    },
    performance: {
      apy: 28.5,
      change24h: 2.3,
      totalReturn: 1250
    },
    autoCompound: true,
    createdAt: '2024-01-15'
  },
  {
    id: 'pos-2',
    tokenId: 12346,
    pool: {
      token0: { symbol: 'USDC', logoURI: 'https://ethereum-optimism.github.io/data/USDC/logo.png' },
      token1: { symbol: 'DAI', logoURI: 'https://ethereum-optimism.github.io/data/DAI/logo.png' },
      feeTier: 0.01
    },
    liquidity: '10000000000',
    amount0: '5000.00',
    amount1: '5000.00',
    currentValue: 10000,
    priceRange: { min: 0.998, max: 1.002, current: 1.001 },
    inRange: true,
    feesEarned: {
      amount0: '25.00',
      amount1: '25.00',
      usdValue: 50
    },
    performance: {
      apy: 18.2,
      change24h: 0.1,
      totalReturn: 425
    },
    autoCompound: false,
    createdAt: '2024-01-20'
  }
]

type ManageAction = 'increase' | 'decrease' | 'collect' | 'burn' | null

export function V4PositionManager() {
  const { isConnected } = useAccount()
  const [positions] = useState<Position[]>(MOCK_POSITIONS)
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null)
  const [manageAction, setManageAction] = useState<ManageAction>(null)
  const [showActionModal, setShowActionModal] = useState(false)

  if (!isConnected) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="text-center py-12">
          <Target className="w-16 h-16 text-slate-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
            Connect Wallet to View Positions
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Connect your wallet to view and manage your Uniswap V4 positions
          </p>
        </div>
      </div>
    )
  }

  const totalValue = positions.reduce((sum, pos) => sum + pos.currentValue, 0)
  const totalFees = positions.reduce((sum, pos) => sum + pos.feesEarned.usdValue, 0)
  const avgApy = positions.length > 0 
    ? positions.reduce((sum, pos) => sum + pos.performance.apy, 0) / positions.length 
    : 0

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
          Your V4 Positions
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Manage your Uniswap V4 liquidity positions and track performance
        </p>
      </div>

      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
              <DollarSign className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">Total Value</p>
              <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                ${totalValue.toLocaleString()}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">Total Fees Earned</p>
              <p className="text-2xl font-bold text-green-600 mono-numbers">
                ${totalFees.toLocaleString()}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">Avg APY</p>
              <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                {avgApy.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl p-6 border border-slate-200 dark:border-white/10">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/20 rounded-lg flex items-center justify-center">
              <Target className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">Active Positions</p>
              <p className="text-2xl font-bold text-slate-900 dark:text-white mono-numbers">
                {positions.length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Positions List */}
      <div className="space-y-6">
        {positions.map((position) => (
          <PositionCard
            key={position.id}
            position={position}
            onManage={(action) => {
              setSelectedPosition(position)
              setManageAction(action)
              setShowActionModal(true)
            }}
          />
        ))}
      </div>

      {/* Management Modal */}
      {showActionModal && selectedPosition && manageAction && (
        <ManagePositionModal
          position={selectedPosition}
          action={manageAction}
          onClose={() => {
            setShowActionModal(false)
            setSelectedPosition(null)
            setManageAction(null)
          }}
        />
      )}
    </div>
  )
}

function PositionCard({ 
  position, 
  onManage 
}: { 
  position: Position
  onManage: (action: ManageAction) => void 
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 overflow-hidden">
      {/* Header */}
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center -space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary-600 rounded-full border-2 border-white dark:border-dark-700"></div>
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full border-2 border-white dark:border-dark-700"></div>
            </div>
            <div>
              <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                {position.pool.token0.symbol}/{position.pool.token1.symbol}
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {position.pool.feeTier}% Fee Tier • Token ID #{position.tokenId}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {position.inRange ? (
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300 rounded-full text-sm font-medium">
                In Range
              </span>
            ) : (
              <span className="px-3 py-1 bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300 rounded-full text-sm font-medium">
                Out of Range
              </span>
            )}
            {position.autoCompound && (
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 rounded-full text-sm font-medium">
                Auto-compound
              </span>
            )}
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Position Value</p>
            <p className="text-xl font-bold text-slate-900 dark:text-white mono-numbers">
              ${position.currentValue.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">Fees Earned</p>
            <p className="text-xl font-bold text-green-600 mono-numbers">
              ${position.feesEarned.usdValue.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">APY</p>
            <p className="text-xl font-bold text-slate-900 dark:text-white mono-numbers">
              {position.performance.apy.toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">24h Change</p>
            <p className={`text-xl font-bold mono-numbers ${
              position.performance.change24h >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {position.performance.change24h >= 0 ? '+' : ''}{position.performance.change24h.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Price Range */}
        <div className="bg-slate-50 dark:bg-dark-600 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Price Range</span>
            <span className="text-sm text-slate-600 dark:text-slate-400">
              Current: ${position.priceRange.current.toLocaleString()}
            </span>
          </div>
          <div className="relative">
            <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full">
              <div 
                className={`h-2 rounded-full ${position.inRange ? 'bg-green-500' : 'bg-red-500'}`}
                style={{ 
                  width: `${Math.min(
                    Math.max(
                      ((position.priceRange.current - position.priceRange.min) / 
                       (position.priceRange.max - position.priceRange.min)) * 100, 
                      0
                    ), 
                    100
                  )}%` 
                }}
              />
            </div>
            <div className="flex justify-between mt-1 text-xs text-slate-600 dark:text-slate-400">
              <span>${position.priceRange.min.toLocaleString()}</span>
              <span>${position.priceRange.max.toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-3">
          <button
            onClick={() => onManage('increase')}
            className="flex-1 flex items-center justify-center space-x-2 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300 py-2 rounded-lg font-medium hover:bg-green-200 dark:hover:bg-green-900/30 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Increase</span>
          </button>
          <button
            onClick={() => onManage('decrease')}
            className="flex-1 flex items-center justify-center space-x-2 bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-300 py-2 rounded-lg font-medium hover:bg-yellow-200 dark:hover:bg-yellow-900/30 transition-colors"
          >
            <Minus className="w-4 h-4" />
            <span>Decrease</span>
          </button>
          <button
            onClick={() => onManage('collect')}
            className="flex-1 flex items-center justify-center space-x-2 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 py-2 rounded-lg font-medium hover:bg-blue-200 dark:hover:bg-blue-900/30 transition-colors"
          >
            <DollarSign className="w-4 h-4" />
            <span>Collect</span>
          </button>
          <button
            onClick={() => onManage('burn')}
            className="flex-1 flex items-center justify-center space-x-2 bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300 py-2 rounded-lg font-medium hover:bg-red-200 dark:hover:bg-red-900/30 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            <span>Close</span>
          </button>
        </div>
      </div>

      {/* Expandable Details */}
      {expanded && (
        <div className="border-t border-slate-200 dark:border-white/10 p-6 bg-slate-50 dark:bg-dark-600">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-white mb-3">Position Details</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Liquidity</span>
                  <span className="text-sm font-medium text-slate-900 dark:text-white mono-numbers">
                    {parseFloat(position.liquidity).toExponential(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">{position.pool.token0.symbol} Amount</span>
                  <span className="text-sm font-medium text-slate-900 dark:text-white mono-numbers">
                    {position.amount0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">{position.pool.token1.symbol} Amount</span>
                  <span className="text-sm font-medium text-slate-900 dark:text-white mono-numbers">
                    {position.amount1}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Created</span>
                  <span className="text-sm font-medium text-slate-900 dark:text-white">
                    {new Date(position.createdAt).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-white mb-3">Uncollected Fees</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">{position.pool.token0.symbol} Fees</span>
                  <span className="text-sm font-medium text-green-600 mono-numbers">
                    {position.feesEarned.amount0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">{position.pool.token1.symbol} Fees</span>
                  <span className="text-sm font-medium text-green-600 mono-numbers">
                    {position.feesEarned.amount1}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Total Value</span>
                  <span className="text-sm font-bold text-green-600 mono-numbers">
                    ${position.feesEarned.usdValue.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Toggle Expand */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full py-3 border-t border-slate-200 dark:border-white/10 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
      >
        {expanded ? 'Show Less' : 'Show More Details'}
      </button>
    </div>
  )
}

function ManagePositionModal({
  position,
  action,
  onClose
}: {
  position: Position
  action: NonNullable<ManageAction>
  onClose: () => void
}) {
  const [amount, setAmount] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [isComplete, setIsComplete] = useState(false)

  const handleSubmit = async () => {
    setIsProcessing(true)
    // Simulate transaction
    await new Promise(resolve => setTimeout(resolve, 3000))
    setIsProcessing(false)
    setIsComplete(true)
  }

  const getActionConfig = () => {
    switch (action) {
      case 'increase':
        return {
          title: 'Increase Liquidity',
          description: 'Add more tokens to your existing position',
          buttonText: 'Increase Position',
          color: 'green'
        }
      case 'decrease':
        return {
          title: 'Decrease Liquidity',
          description: 'Remove tokens from your position',
          buttonText: 'Decrease Position',
          color: 'yellow'
        }
      case 'collect':
        return {
          title: 'Collect Fees',
          description: 'Claim your earned trading fees',
          buttonText: 'Collect Fees',
          color: 'blue'
        }
      case 'burn':
        return {
          title: 'Close Position',
          description: 'Remove all liquidity and close this position',
          buttonText: 'Close Position',
          color: 'red'
        }
    }
  }

  const config = getActionConfig()

  if (isComplete) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white dark:bg-dark-700 rounded-xl p-8 max-w-md w-full mx-4 text-center">
          <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
            Transaction Complete!
          </h3>
          <p className="text-slate-600 dark:text-slate-400 mb-6">
            Your {action} transaction has been successfully processed.
          </p>
          <button
            onClick={onClose}
            className="w-full bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-dark-700 rounded-xl p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-slate-900 dark:text-white">
            {config.title}
          </h3>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
          >
            ✕
          </button>
        </div>

        <p className="text-slate-600 dark:text-slate-400 mb-6">
          {config.description}
        </p>

        {/* Position Info */}
        <div className="bg-slate-50 dark:bg-dark-600 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-3 mb-2">
            <div className="flex items-center -space-x-1">
              <div className="w-6 h-6 bg-gradient-to-br from-primary to-primary-600 rounded-full"></div>
              <div className="w-6 h-6 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full"></div>
            </div>
            <span className="font-medium text-slate-900 dark:text-white">
              {position.pool.token0.symbol}/{position.pool.token1.symbol}
            </span>
          </div>
          {action === 'collect' && (
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600 dark:text-slate-400">Available Fees</span>
                <span className="font-medium text-green-600">
                  ${position.feesEarned.usdValue.toLocaleString()}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Amount Input (for increase/decrease) */}
        {(action === 'increase' || action === 'decrease') && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Amount
            </label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
              className="w-full px-4 py-3 border border-slate-200 dark:border-white/10 rounded-lg bg-white dark:bg-dark-600 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary focus:border-transparent mono-numbers"
            />
          </div>
        )}

        {/* Warning for burn action */}
        {action === 'burn' && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
              <div>
                <h4 className="font-medium text-red-800 dark:text-red-300 mb-1">
                  Warning
                </h4>
                <p className="text-sm text-red-700 dark:text-red-400">
                  This will permanently close your position and withdraw all liquidity. This action cannot be undone.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex space-x-3">
          <button
            onClick={onClose}
            disabled={isProcessing}
            className="flex-1 bg-slate-100 dark:bg-dark-600 text-slate-900 dark:text-white py-3 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-dark-500 disabled:opacity-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isProcessing || (action !== 'collect' && action !== 'burn' && !amount)}
            className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
              config.color === 'green' ? 'bg-green-600 hover:bg-green-700' :
              config.color === 'yellow' ? 'bg-yellow-600 hover:bg-yellow-700' :
              config.color === 'blue' ? 'bg-blue-600 hover:bg-blue-700' :
              'bg-red-600 hover:bg-red-700'
            } text-white disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isProcessing ? (
              <div className="flex items-center justify-center space-x-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Processing...</span>
              </div>
            ) : (
              config.buttonText
            )}
          </button>
        </div>
      </div>
    </div>
  )
}