'use client'

import React, { useState, useEffect } from 'react'
import { useAccount, useReadContract, useWriteContract } from 'wagmi'
import { formatUnits, parseUnits } from 'viem'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
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

  // Contract read hooks
  const { data: vaultInfo } = useReadContract({
    address: vaultAddress as `0x${string}`,
    abi: [
      {
        name: 'getVaultConfig',
        type: 'function',
        stateMutability: 'view',
        inputs: [],
        outputs: [{ type: 'tuple', components: [] }]
      }
    ],
    functionName: 'getVaultConfig'
  })

  const { data: vaultMetrics } = useReadContract({
    address: vaultAddress as `0x${string}`,
    abi: [
      {
        name: 'getVaultMetrics',
        type: 'function', 
        stateMutability: 'view',
        inputs: [],
        outputs: [{ type: 'tuple', components: [] }]
      }
    ],
    functionName: 'getVaultMetrics'
  })

  const { data: positionRanges } = useReadContract({
    address: vaultAddress as `0x${string}`,
    abi: [
      {
        name: 'getPositionRanges',
        type: 'function',
        stateMutability: 'view', 
        inputs: [],
        outputs: [{ type: 'array', components: [] }]
      }
    ],
    functionName: 'getPositionRanges'
  })

  const { data: aiRecommendation } = useReadContract({
    address: vaultAddress as `0x${string}`,
    abi: [
      {
        name: 'getAIRecommendation',
        type: 'function',
        stateMutability: 'view',
        inputs: [],
        outputs: [{ type: 'tuple', components: [] }]
      }
    ],
    functionName: 'getAIRecommendation'
  })

  // Contract write hooks
  const { writeContract: deposit } = useWriteContract()
  const { writeContract: withdraw } = useWriteContract()
  const { writeContract: compound } = useWriteContract()
  const { writeContract: rebalance } = useWriteContract()

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
      await deposit({
        address: vaultAddress as `0x${string}`,
        abi: [],
        functionName: 'deposit',
        args: [parseUnits(depositAmount, 18), address]
      })
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
      await withdraw({
        address: vaultAddress as `0x${string}`,
        abi: [],
        functionName: 'withdraw',
        args: [parseUnits(withdrawAmount, 18), address, address]
      })
    } catch (error) {
      console.error('Withdraw failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleCompound = async () => {
    setIsLoading(true)
    try {
      await compound({
        address: vaultAddress as `0x${string}`,
        abi: [],
        functionName: 'compound',
        args: []
      })
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

  return (
    <div className=\"min-h-screen bg-gray-50 p-6\">
      <div className=\"max-w-7xl mx-auto space-y-6\">
        {/* Header */}
        <div className=\"flex items-center justify-between\">
          <div>
            <h1 className=\"text-3xl font-bold text-gray-900\">{mockVaultInfo.name}</h1>
            <p className=\"text-gray-600\">
              {mockVaultInfo.token0}/{mockVaultInfo.token1} • {mockVaultInfo.fee/10000}% Fee
            </p>
          </div>
          <div className=\"flex items-center space-x-3\">
            <Badge variant=\"outline\" className=\"bg-green-50 text-green-700\">
              <CheckCircle className=\"w-4 h-4 mr-1\" />
              Active
            </Badge>
            <Button size=\"sm\" variant=\"outline\">
              <Settings className=\"w-4 h-4 mr-2\" />
              Configure
            </Button>
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6\">
          <Card>
            <CardContent className=\"p-6\">
              <div className=\"flex items-center justify-between\">
                <div>
                  <p className=\"text-sm font-medium text-gray-600\">Total Value Locked</p>
                  <p className=\"text-2xl font-bold text-gray-900\">
                    {formatCurrency(mockMetrics.totalValueLocked)}
                  </p>
                </div>
                <DollarSign className=\"w-8 h-8 text-blue-600\" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className=\"p-6\">
              <div className=\"flex items-center justify-between\">
                <div>
                  <p className=\"text-sm font-medium text-gray-600\">Current APR</p>
                  <p className=\"text-2xl font-bold text-green-600\">
                    {formatPercentage(mockMetrics.apr)}
                  </p>
                </div>
                <TrendingUp className=\"w-8 h-8 text-green-600\" />
              </div>
              <div className=\"mt-2\">
                <p className=\"text-xs text-gray-500\">
                  24h Fees: {formatCurrency(mockMetrics.totalFees24h)}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className=\"p-6\">
              <div className=\"flex items-center justify-between\">
                <div>
                  <p className=\"text-sm font-medium text-gray-600\">Sharpe Ratio</p>
                  <p className=\"text-2xl font-bold text-purple-600\">
                    {parseFloat(mockMetrics.sharpeRatio).toFixed(2)}
                  </p>
                </div>
                <BarChart3 className=\"w-8 h-8 text-purple-600\" />
              </div>
              <div className=\"mt-2\">
                <p className=\"text-xs text-gray-500\">
                  Max Drawdown: {formatPercentage(mockMetrics.maxDrawdown)}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className=\"p-6\">
              <div className=\"flex items-center justify-between\">
                <div>
                  <p className=\"text-sm font-medium text-gray-600\">AI Optimizations</p>
                  <p className=\"text-2xl font-bold text-orange-600\">
                    {mockMetrics.aiOptimizationCount}
                  </p>
                </div>
                <Zap className=\"w-8 h-8 text-orange-600\" />
              </div>
              <div className=\"mt-2\">
                <p className=\"text-xs text-gray-500\">
                  Compounds: {mockMetrics.successfulCompounds}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className=\"space-y-6\">
          <TabsList className=\"grid w-full grid-cols-5\">
            <TabsTrigger value=\"overview\">Overview</TabsTrigger>
            <TabsTrigger value=\"positions\">Positions</TabsTrigger>
            <TabsTrigger value=\"strategy\">Strategy</TabsTrigger>
            <TabsTrigger value=\"analytics\">Analytics</TabsTrigger>
            <TabsTrigger value=\"manage\">Manage</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value=\"overview\" className=\"space-y-6\">
            <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
              {/* Portfolio Composition */}
              <Card>
                <CardHeader>
                  <CardTitle className=\"flex items-center\">
                    <PieChart className=\"w-5 h-5 mr-2\" />
                    Portfolio Composition
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Your Shares</span>
                      <span className=\"text-sm text-gray-600\">
                        {(parseFloat(mockVaultInfo.userShares) / parseFloat(mockVaultInfo.totalShares) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <Progress 
                      value={(parseFloat(mockVaultInfo.userShares) / parseFloat(mockVaultInfo.totalShares)) * 100} 
                      className=\"h-2\"
                    />
                    <div className=\"grid grid-cols-2 gap-4 text-sm\">
                      <div>
                        <p className=\"text-gray-600\">Your Value</p>
                        <p className=\"font-medium\">
                          {formatCurrency((parseFloat(mockMetrics.totalValueLocked) * parseFloat(mockVaultInfo.userShares) / parseFloat(mockVaultInfo.totalShares)).toString())}
                        </p>
                      </div>
                      <div>
                        <p className=\"text-gray-600\">Share Count</p>
                        <p className=\"font-medium\">{mockVaultInfo.userShares}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* AI Recommendation */}
              <Card>
                <CardHeader>
                  <CardTitle className=\"flex items-center\">
                    <Zap className=\"w-5 h-5 mr-2\" />
                    AI Recommendation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <div className=\"flex items-center justify-between\">
                      <Badge variant=\"outline\" className=\"bg-blue-50 text-blue-700\">
                        {mockRecommendation.strategyType.replace('_', ' ')}
                      </Badge>
                      <div className=\"text-right\">
                        <p className=\"text-sm text-gray-600\">Confidence</p>
                        <p className=\"font-medium text-blue-600\">
                          {(mockRecommendation.confidenceScore * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                    
                    <div className=\"grid grid-cols-2 gap-4 text-sm\">
                      <div>
                        <p className=\"text-gray-600\">Expected APR</p>
                        <p className=\"font-medium text-green-600\">
                          {formatPercentage(mockRecommendation.expectedAPR.toString())}
                        </p>
                      </div>
                      <div>
                        <p className=\"text-gray-600\">Risk Level</p>
                        <p className=\"font-medium text-orange-600\">
                          {formatPercentage(mockRecommendation.expectedRisk.toString())}
                        </p>
                      </div>
                    </div>
                    
                    <p className=\"text-xs text-gray-600\">
                      {mockRecommendation.reasoning}
                    </p>
                    
                    <Button size=\"sm\" className=\"w-full\">
                      Apply Recommendation
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Positions Tab */}
          <TabsContent value=\"positions\" className=\"space-y-6\">
            <Card>
              <CardHeader>
                <CardTitle>Active Position Ranges</CardTitle>
              </CardHeader>
              <CardContent>
                <div className=\"space-y-4\">
                  {mockRanges.map((range, index) => (
                    <div key={range.id} className=\"border rounded-lg p-4\">
                      <div className=\"flex items-center justify-between mb-3\">
                        <h4 className=\"font-medium\">{range.name}</h4>
                        <Badge variant={range.isActive ? \"default\" : \"secondary\"}>
                          {range.isActive ? \"Active\" : \"Inactive\"}
                        </Badge>
                      </div>
                      
                      <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4 text-sm\">
                        <div>
                          <p className=\"text-gray-600\">Tick Range</p>
                          <p className=\"font-medium\">{range.tickLower} to {range.tickUpper}</p>
                        </div>
                        <div>
                          <p className=\"text-gray-600\">Allocation</p>
                          <p className=\"font-medium\">{range.allocation / 100}%</p>
                        </div>
                        <div>
                          <p className=\"text-gray-600\">Liquidity</p>
                          <p className=\"font-medium\">{formatCurrency(range.liquidity)}</p>
                        </div>
                        <div>
                          <p className=\"text-gray-600\">Status</p>
                          <p className=\"font-medium text-green-600\">In Range</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Strategy Tab */}
          <TabsContent value=\"strategy\" className=\"space-y-6\">
            <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
              <Card>
                <CardHeader>
                  <CardTitle>Current Strategy</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Strategy Mode</span>
                      <Badge>AI Assisted</Badge>
                    </div>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Position Type</span>
                      <Badge variant=\"outline\">Dual Position</Badge>
                    </div>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Auto Compound</span>
                      <Badge variant=\"outline\" className=\"bg-green-50 text-green-700\">
                        Enabled
                      </Badge>
                    </div>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Rebalance Threshold</span>
                      <span className=\"text-sm text-gray-600\">10%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Strategy Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Win Rate</span>
                      <span className=\"text-sm font-medium text-green-600\">76%</span>
                    </div>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Avg Return per Rebalance</span>
                      <span className=\"text-sm font-medium text-green-600\">0.8%</span>
                    </div>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">Capital Efficiency</span>
                      <span className=\"text-sm font-medium text-blue-600\">85%</span>
                    </div>
                    <div className=\"flex items-center justify-between\">
                      <span className=\"text-sm font-medium\">vs Benchmark</span>
                      <span className=\"text-sm font-medium text-green-600\">+4.2%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value=\"analytics\" className=\"space-y-6\">
            <Card>
              <CardHeader>
                <CardTitle>Performance Analytics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className=\"grid grid-cols-1 md:grid-cols-3 gap-6\">
                  <div className=\"text-center\">
                    <p className=\"text-2xl font-bold text-green-600\">
                      {formatPercentage(mockMetrics.apr)}
                    </p>
                    <p className=\"text-sm text-gray-600\">Current APR</p>
                  </div>
                  <div className=\"text-center\">
                    <p className=\"text-2xl font-bold text-purple-600\">
                      {parseFloat(mockMetrics.sharpeRatio).toFixed(2)}
                    </p>
                    <p className=\"text-sm text-gray-600\">Sharpe Ratio</p>
                  </div>
                  <div className=\"text-center\">
                    <p className=\"text-2xl font-bold text-red-600\">
                      {formatPercentage(mockMetrics.impermanentLoss)}
                    </p>
                    <p className=\"text-sm text-gray-600\">Impermanent Loss</p>
                  </div>
                </div>
                
                <div className=\"mt-6 text-center\">
                  <p className=\"text-gray-600 mb-4\">Detailed analytics charts would be displayed here</p>
                  <div className=\"h-64 bg-gray-100 rounded-lg flex items-center justify-center\">
                    <p className=\"text-gray-500\">Performance Chart Placeholder</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Manage Tab */}
          <TabsContent value=\"manage\" className=\"space-y-6\">
            <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
              {/* Deposit */}
              <Card>
                <CardHeader>
                  <CardTitle>Deposit</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <Input
                      type=\"number\"
                      placeholder=\"Amount to deposit\"
                      value={depositAmount}
                      onChange={(e) => setDepositAmount(e.target.value)}
                    />
                    <Button 
                      onClick={handleDeposit} 
                      disabled={isLoading || !depositAmount}
                      className=\"w-full\"
                    >
                      {isLoading ? 'Depositing...' : 'Deposit'}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Withdraw */}
              <Card>
                <CardHeader>
                  <CardTitle>Withdraw</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"space-y-4\">
                    <Input
                      type=\"number\"
                      placeholder=\"Amount to withdraw\"
                      value={withdrawAmount}
                      onChange={(e) => setWithdrawAmount(e.target.value)}
                    />
                    <Button 
                      onClick={handleWithdraw} 
                      disabled={isLoading || !withdrawAmount}
                      variant=\"outline\"
                      className=\"w-full\"
                    >
                      {isLoading ? 'Withdrawing...' : 'Withdraw'}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Actions */}
              <Card className=\"lg:col-span-2\">
                <CardHeader>
                  <CardTitle>Vault Actions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className=\"grid grid-cols-1 md:grid-cols-3 gap-4\">
                    <Button 
                      onClick={handleCompound}
                      disabled={isLoading}
                      className=\"flex items-center justify-center\"
                    >
                      <Zap className=\"w-4 h-4 mr-2\" />
                      Compound
                    </Button>
                    <Button 
                      onClick={() => rebalance({})}
                      disabled={isLoading}
                      variant=\"outline\"
                      className=\"flex items-center justify-center\"
                    >
                      <BarChart3 className=\"w-4 h-4 mr-2\" />
                      Rebalance
                    </Button>
                    <Button 
                      variant=\"outline\"
                      className=\"flex items-center justify-center\"
                    >
                      <Eye className=\"w-4 h-4 mr-2\" />
                      View Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}