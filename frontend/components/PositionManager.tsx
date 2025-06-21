'use client'

import { useState, useEffect } from 'react'
import { useAccount, useReadContract, useWriteContract } from 'wagmi'
import { 
  TrendingUp, 
  TrendingDown, 
  Zap, 
  Settings, 
  Eye, 
  EyeOff, 
  Bot, 
  User,
  Clock,
  DollarSign,
  Percent,
  Activity
} from 'lucide-react'

interface Position {
  tokenId: string
  token0: string
  token1: string
  fee: number
  liquidity: string
  tickLower: number
  tickUpper: number
  tokensOwed0: string
  tokensOwed1: string
  isAIManaged: boolean
  lastCompoundTime: number
  compoundCount: number
  currentPrice: number
  priceRange: {
    min: number
    max: number
  }
  performance: {
    totalFees: number
    totalRewards: number
    apr: number
    impermanentLoss: number
  }
}

interface CompoundModalProps {
  position: Position
  isOpen: boolean
  onClose: () => void
  onCompound: (params: CompoundParams) => void
}

interface CompoundParams {
  tokenId: string
  rewardConversion: 'NONE' | 'TOKEN_0' | 'TOKEN_1' | 'AI_OPTIMIZED'
  withdrawReward: boolean
  doSwap: boolean
  useAIOptimization: boolean
}

function CompoundModal({ position, isOpen, onClose, onCompound }: CompoundModalProps) {
  const [params, setParams] = useState<CompoundParams>({
    tokenId: position.tokenId,
    rewardConversion: 'NONE',
    withdrawReward: false,
    doSwap: false,
    useAIOptimization: position.isAIManaged
  })

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
        <h3 className="text-lg font-semibold mb-4">Compound Position</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Reward Conversion</label>
            <select
              value={params.rewardConversion}
              onChange={(e) => setParams({...params, rewardConversion: e.target.value as any})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
            >
              <option value="NONE">No Conversion</option>
              <option value="TOKEN_0">Convert to {position.token0}</option>
              <option value="TOKEN_1">Convert to {position.token1}</option>
              {position.isAIManaged && <option value="AI_OPTIMIZED">AI Optimized</option>}
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={params.withdrawReward}
              onChange={(e) => setParams({...params, withdrawReward: e.target.checked})}
              className="rounded"
            />
            <label className="text-sm">Withdraw reward immediately</label>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={params.doSwap}
              onChange={(e) => setParams({...params, doSwap: e.target.checked})}
              className="rounded"
            />
            <label className="text-sm">Optimize with swapping (higher gas)</label>
          </div>

          {position.isAIManaged && (
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={params.useAIOptimization}
                onChange={(e) => setParams({...params, useAIOptimization: e.target.checked})}
                className="rounded"
              />
              <label className="text-sm flex items-center space-x-1">
                <Bot className="w-4 h-4" />
                <span>Use AI optimization</span>
              </label>
            </div>
          )}

          <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
            <p className="text-sm text-blue-700 dark:text-blue-300">
              Expected compound: {position.tokensOwed0} {position.token0} + {position.tokensOwed1} {position.token1}
            </p>
          </div>
        </div>

        <div className="flex space-x-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Cancel
          </button>
          <button
            onClick={() => onCompound(params)}
            className="flex-1 px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600"
          >
            Compound
          </button>
        </div>
      </div>
    </div>
  )
}

function PositionCard({ position, onCompound }: { position: Position; onCompound: (position: Position) => void }) {
  const [showDetails, setShowDetails] = useState(false)
  
  const isInRange = position.currentPrice >= position.priceRange.min && position.currentPrice <= position.priceRange.max
  const hasFeesToCompound = parseFloat(position.tokensOwed0) > 0 || parseFloat(position.tokensOwed1) > 0

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1">
            <span className="font-semibold">{position.token0}/{position.token1}</span>
            <span className="text-sm text-gray-500">{position.fee / 10000}%</span>
          </div>
          {position.isAIManaged && (
            <div className="flex items-center space-x-1 bg-purple-100 dark:bg-purple-900/30 px-2 py-1 rounded-full">
              <Bot className="w-3 h-3 text-purple-600 dark:text-purple-400" />
              <span className="text-xs text-purple-600 dark:text-purple-400">AI Managed</span>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <div className={`px-2 py-1 rounded-full text-xs ${
            isInRange 
              ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
              : 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
          }`}>
            {isInRange ? 'In Range' : 'Out of Range'}
          </div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
          >
            {showDetails ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-sm text-gray-500">Liquidity</p>
          <p className="font-semibold">${(parseFloat(position.liquidity) / 1e18 * position.currentPrice).toLocaleString()}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">APR</p>
          <div className="flex items-center space-x-1">
            {position.performance.apr > 0 ? (
              <TrendingUp className="w-4 h-4 text-green-500" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500" />
            )}
            <span className={`font-semibold ${position.performance.apr > 0 ? 'text-green-500' : 'text-red-500'}`}>
              {position.performance.apr.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      {/* Fees Available */}
      {hasFeesToCompound && (
        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-700 dark:text-green-300">Fees Available to Compound</p>
              <p className="text-xs text-green-600 dark:text-green-400">
                {position.tokensOwed0} {position.token0} + {position.tokensOwed1} {position.token1}
              </p>
            </div>
            <button
              onClick={() => onCompound(position)}
              className="flex items-center space-x-1 bg-green-500 text-white px-3 py-1 rounded-md hover:bg-green-600 text-sm"
            >
              <Zap className="w-3 h-3" />
              <span>Compound</span>
            </button>
          </div>
        </div>
      )}

      {/* Detailed Stats */}
      {showDetails && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4 space-y-3">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Token ID</p>
              <p className="font-medium">#{position.tokenId}</p>
            </div>
            <div>
              <p className="text-gray-500">Compounds</p>
              <p className="font-medium">{position.compoundCount}</p>
            </div>
            <div>
              <p className="text-gray-500">Last Compound</p>
              <p className="font-medium">
                {position.lastCompoundTime > 0 
                  ? new Date(position.lastCompoundTime * 1000).toLocaleDateString()
                  : 'Never'
                }
              </p>
            </div>
            <div>
              <p className="text-gray-500">Total Fees</p>
              <p className="font-medium">${position.performance.totalFees.toFixed(2)}</p>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Price Range</span>
              <span className="font-medium">
                ${position.priceRange.min.toFixed(4)} - ${position.priceRange.max.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Current Price</span>
              <span className="font-medium">${position.currentPrice.toFixed(4)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Impermanent Loss</span>
              <span className={`font-medium ${position.performance.impermanentLoss < 0 ? 'text-red-500' : 'text-green-500'}`}>
                {position.performance.impermanentLoss.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export function PositionManager() {
  const { address, isConnected } = useAccount()
  const [positions, setPositions] = useState<Position[]>([])
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null)
  const [showCompoundModal, setShowCompoundModal] = useState(false)
  const [filter, setFilter] = useState<'all' | 'ai-managed' | 'manual'>('all')

  // Mock data for now - replace with actual contract calls
  useEffect(() => {
    if (isConnected) {
      // TODO: Fetch positions from contract
      const mockPositions: Position[] = [
        {
          tokenId: '12345',
          token0: 'WETH',
          token1: 'USDC',
          fee: 3000,
          liquidity: '1000000000000000000',
          tickLower: -276320,
          tickUpper: -276300,
          tokensOwed0: '0.0045',
          tokensOwed1: '12.34',
          isAIManaged: true,
          lastCompoundTime: Date.now() / 1000 - 86400,
          compoundCount: 15,
          currentPrice: 2456.78,
          priceRange: { min: 2400, max: 2500 },
          performance: {
            totalFees: 234.56,
            totalRewards: 45.67,
            apr: 18.45,
            impermanentLoss: -2.3
          }
        },
        {
          tokenId: '67890',
          token0: 'USDC',
          token1: 'USDT',
          fee: 500,
          liquidity: '5000000000000000000',
          tickLower: -1,
          tickUpper: 1,
          tokensOwed0: '0.0012',
          tokensOwed1: '0.0008',
          isAIManaged: false,
          lastCompoundTime: 0,
          compoundCount: 0,
          currentPrice: 1.0001,
          priceRange: { min: 0.9995, max: 1.0005 },
          performance: {
            totalFees: 45.23,
            totalRewards: 0,
            apr: 5.67,
            impermanentLoss: -0.1
          }
        }
      ]
      setPositions(mockPositions)
    }
  }, [isConnected])

  const filteredPositions = positions.filter(position => {
    if (filter === 'ai-managed') return position.isAIManaged
    if (filter === 'manual') return !position.isAIManaged
    return true
  })

  const handleCompound = (position: Position) => {
    setSelectedPosition(position)
    setShowCompoundModal(true)
  }

  const executeCompound = async (params: CompoundParams) => {
    // TODO: Execute compound transaction
    console.log('Compounding with params:', params)
    setShowCompoundModal(false)
    setSelectedPosition(null)
  }

  if (!isConnected) {
    return (
      <div className="text-center py-12">
        <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-600 dark:text-gray-400 mb-2">
          Connect Your Wallet
        </h3>
        <p className="text-gray-500">
          Connect your wallet to view and manage your liquidity positions
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Position Manager</h2>
          <p className="text-gray-600 dark:text-gray-400">Manage your Uniswap V3 positions with AI optimization</p>
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
          >
            <option value="all">All Positions</option>
            <option value="ai-managed">AI Managed</option>
            <option value="manual">Manual</option>
          </select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-blue-500" />
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Positions</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{positions.length}</p>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <Bot className="w-5 h-5 text-purple-500" />
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">AI Managed</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {positions.filter(p => p.isAIManaged).length}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <DollarSign className="w-5 h-5 text-green-500" />
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Fees</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            ${positions.reduce((sum, p) => sum + p.performance.totalFees, 0).toFixed(2)}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <Percent className="w-5 h-5 text-yellow-500" />
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg APR</span>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {(positions.reduce((sum, p) => sum + p.performance.apr, 0) / positions.length || 0).toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Positions Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredPositions.map((position) => (
          <PositionCard
            key={position.tokenId}
            position={position}
            onCompound={handleCompound}
          />
        ))}
      </div>

      {filteredPositions.length === 0 && (
        <div className="text-center py-12">
          <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-600 dark:text-gray-400 mb-2">
            No Positions Found
          </h3>
          <p className="text-gray-500">
            {filter === 'all' 
              ? 'You don\'t have any liquidity positions yet.'
              : `No ${filter.replace('-', ' ')} positions found.`
            }
          </p>
        </div>
      )}

      {/* Compound Modal */}
      {selectedPosition && (
        <CompoundModal
          position={selectedPosition}
          isOpen={showCompoundModal}
          onClose={() => {
            setShowCompoundModal(false)
            setSelectedPosition(null)
          }}
          onCompound={executeCompound}
        />
      )}
    </div>
  )
}