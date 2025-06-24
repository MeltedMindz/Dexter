'use client'

import { useState, useEffect } from 'react'
import { ArrowLeft, Brain, TrendingUp, AlertTriangle, Target, Coins, Activity, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import { useBalance } from 'wagmi'

interface TokenPosition {
  token: string
  symbol: string
  balance: string
  valueUSD: number
  type: 'idle' | 'lp' | 'staked'
}

interface PoolRecommendation {
  poolAddress: string
  tokenPair: string
  fee: string
  expectedAPR: number
  ilRisk: 'low' | 'medium' | 'high'
  strategy: string
  reasoning: string
  confidence: number
}

interface WalletAnalysisProps {
  address: string
  onBack: () => void
}

export function WalletAnalysis({ address, onBack }: WalletAnalysisProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(true)
  const [analysis, setAnalysis] = useState<{
    totalValue: number
    positions: TokenPosition[]
    recommendations: PoolRecommendation[]
    riskProfile: 'conservative' | 'moderate' | 'aggressive'
  } | null>(null)

  const { data: ethBalance } = useBalance({ address: address as `0x${string}` })

  useEffect(() => {
    // Simulate AI analysis
    const analyzeWallet = async () => {
      setIsAnalyzing(true)
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 3000))
      
      // Mock analysis results
      const mockAnalysis = {
        totalValue: 12450,
        positions: [
          {
            token: '0xETH',
            symbol: 'ETH',
            balance: ethBalance?.formatted || '2.5',
            valueUSD: 8750,
            type: 'idle' as const
          },
          {
            token: '0xUSDC',
            symbol: 'USDC',
            balance: '2500',
            valueUSD: 2500,
            type: 'idle' as const
          },
          {
            token: '0xUNI-V3-ETH-USDC',
            symbol: 'UNI-V3-ETH-USDC',
            balance: '1',
            valueUSD: 1200,
            type: 'lp' as const
          }
        ],
        recommendations: [
          {
            poolAddress: '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640',
            tokenPair: 'ETH/USDC',
            fee: '0.3%',
            expectedAPR: 18.5,
            ilRisk: 'medium' as const,
            strategy: 'Tight Range (95% concentration)',
            reasoning: 'Based on your ETH/USDC holdings and current market volatility, a tight range strategy could maximize fee generation',
            confidence: 87
          },
          {
            poolAddress: '0x4e68Ccd3E89f51C3074ca5072bbAC773960dFa36',
            tokenPair: 'ETH/USDT',
            fee: '0.3%',
            expectedAPR: 16.2,
            ilRisk: 'medium' as const,
            strategy: 'Wide Range (Conservative)',
            reasoning: 'Alternative stable pairing with lower maintenance requirements',
            confidence: 72
          }
        ],
        riskProfile: 'moderate' as const
      }
      
      setAnalysis(mockAnalysis)
      setIsAnalyzing(false)
    }

    analyzeWallet()
  }, [address, ethBalance])

  if (isAnalyzing) {
    return (
      <div className="min-h-screen bg-white dark:bg-black font-sans flex items-center justify-center">
        <div className="text-center max-w-2xl mx-auto px-6">
          <Brain className="w-24 h-24 text-primary mx-auto mb-8 animate-pulse" />
          <h1 className="text-4xl font-bold text-black dark:text-white mb-6 text-brutal">
            AI ANALYZING YOUR PORTFOLIO
          </h1>
          <div className="space-y-4 font-mono">
            <div className="bg-gray-100 dark:bg-gray-900 p-4 border-2 border-black dark:border-white">
              <p className="text-green-600">✓ Scanning wallet balances...</p>
            </div>
            <div className="bg-gray-100 dark:bg-gray-900 p-4 border-2 border-black dark:border-white">
              <p className="text-blue-600">⏳ Detecting LP positions...</p>
            </div>
            <div className="bg-gray-100 dark:bg-gray-900 p-4 border-2 border-black dark:border-white">
              <p className="text-yellow-600">🔍 Analyzing market opportunities...</p>
            </div>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mt-6 font-mono">
            This usually takes 10-15 seconds
          </p>
        </div>
      </div>
    )
  }

  if (!analysis) return null

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'low': return '🟢'
      case 'medium': return '🟡'
      case 'high': return '🔴'
      default: return '⚪'
    }
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black font-sans">
      {/* Header */}
      <section className="border-b-2 border-black dark:border-white bg-gradient-to-r from-primary to-accent-cyan">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-black font-bold hover:opacity-80"
            >
              <ArrowLeft className="w-5 h-5" />
              BACK TO HOME
            </button>
            <div className="text-center">
              <h1 className="text-3xl font-bold text-black text-brutal">
                AI PORTFOLIO ANALYSIS COMPLETE
              </h1>
              <p className="text-black font-mono">
                Address: {address.slice(0, 6)}...{address.slice(-4)}
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-black text-brutal">
                ${analysis.totalValue.toLocaleString()}
              </div>
              <div className="text-sm text-black font-mono">
                Total Portfolio Value
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Portfolio Breakdown */}
      <section className="py-12 border-b-2 border-black dark:border-white">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-black dark:text-white mb-8 text-brutal">
            DETECTED ASSETS & POSITIONS
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            {analysis.positions.map((position, index) => (
              <div key={index} className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <Coins className="w-8 h-8 text-accent-cyan" />
                    <div>
                      <h3 className="font-bold text-black dark:text-white text-brutal">
                        {position.symbol}
                      </h3>
                      <span className={`text-xs px-2 py-1 rounded ${
                        position.type === 'idle' ? 'bg-yellow-100 text-yellow-800' :
                        position.type === 'lp' ? 'bg-green-100 text-green-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        {position.type.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-black dark:text-white">
                      ${position.valueUSD.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                      {parseFloat(position.balance).toFixed(2)} {position.symbol}
                    </div>
                  </div>
                </div>
                
                {position.type === 'idle' && (
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 border border-yellow-200 dark:border-yellow-700">
                    <p className="text-yellow-800 dark:text-yellow-200 text-sm font-mono">
                      💡 Idle asset - earning no yield
                    </p>
                  </div>
                )}
                
                {position.type === 'lp' && (
                  <div className="bg-green-50 dark:bg-green-900/20 p-3 border border-green-200 dark:border-green-700">
                    <p className="text-green-800 dark:text-green-200 text-sm font-mono">
                      ✅ Active LP position
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Recommendations */}
      <section className="py-12 border-b-2 border-black dark:border-white">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex items-center gap-4 mb-8">
            <Brain className="w-10 h-10 text-primary" />
            <h2 className="text-3xl font-bold text-black dark:text-white text-brutal">
              PERSONALIZED RECOMMENDATIONS
            </h2>
          </div>
          
          <div className="space-y-6">
            {analysis.recommendations.map((rec, index) => (
              <div key={index} className="bg-white dark:bg-black border-2 border-black dark:border-white p-8">
                <div className="grid md:grid-cols-3 gap-6">
                  <div>
                    <div className="flex items-center gap-3 mb-3">
                      <Target className="w-6 h-6 text-accent-cyan" />
                      <h3 className="text-xl font-bold text-black dark:text-white text-brutal">
                        {rec.tokenPair} Pool
                      </h3>
                    </div>
                    <div className="space-y-2 font-mono text-sm">
                      <div>Fee Tier: <span className="font-bold">{rec.fee}</span></div>
                      <div>Strategy: <span className="font-bold">{rec.strategy}</span></div>
                      <div>Pool: {rec.poolAddress.slice(0, 8)}...</div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Expected APR:</span>
                        <span className="text-2xl font-bold text-green-500">
                          {rec.expectedAPR}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-600 dark:text-gray-400">IL Risk:</span>
                        <span className="flex items-center gap-2">
                          {getRiskIcon(rec.ilRisk)}
                          <span className="font-bold capitalize">{rec.ilRisk}</span>
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
                        <span className="font-bold">{rec.confidence}%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 border border-gray-200 dark:border-gray-700 mb-4">
                      <h4 className="font-bold text-black dark:text-white mb-2">AI Reasoning:</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                        {rec.reasoning}
                      </p>
                    </div>
                    
                    <div className="flex gap-2">
                      <Link
                        href={`/vaults/create?pool=${rec.poolAddress}`}
                        className="bg-primary text-black px-4 py-2 border border-black dark:border-white text-sm font-bold hover:opacity-80 flex-1 text-center"
                      >
                        CREATE MANUAL VAULT
                      </Link>
                      <Link
                        href={`/vaults/create?pool=${rec.poolAddress}&ai=true`}
                        className="bg-accent-cyan text-black px-4 py-2 border border-black dark:border-white text-sm font-bold hover:opacity-80 flex-1 text-center"
                      >
                        AI-MANAGED VAULT
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Action Selection */}
      <section className="py-12">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-black dark:text-white text-center mb-8 text-brutal">
            CHOOSE YOUR PATH
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* Manual Vault */}
            <div className="bg-white dark:bg-black border-2 border-green-500 p-8 text-center">
              <Target className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">
                MANUAL VAULT
              </h3>
              <p className="text-gray-600 dark:text-gray-400 font-mono mb-6">
                Use AI insights to build your own vault with full control over parameters
              </p>
              <Link
                href="/vaults/create"
                className="bg-green-500 text-white px-6 py-3 font-bold hover:opacity-80 inline-flex items-center gap-2"
              >
                CREATE MANUAL VAULT
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>

            {/* AI-Managed Vault */}
            <div className="bg-white dark:bg-black border-2 border-primary p-8 text-center">
              <Brain className="w-16 h-16 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">
                AI-MANAGED VAULT
              </h3>
              <p className="text-gray-600 dark:text-gray-400 font-mono mb-6">
                Let AI handle rebalancing and optimization with optional oversight
              </p>
              <Link
                href="/vaults/create?ai=true"
                className="bg-primary text-black px-6 py-3 font-bold hover:opacity-80 inline-flex items-center gap-2"
              >
                CREATE AI VAULT
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>

            {/* Learn More */}
            <div className="bg-white dark:bg-black border-2 border-accent-cyan p-8 text-center">
              <Activity className="w-16 h-16 text-accent-cyan mx-auto mb-4" />
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">
                LEARN MORE
              </h3>
              <p className="text-gray-600 dark:text-gray-400 font-mono mb-6">
                Chat with AI about pools, risks, and strategies before deciding
              </p>
              <button className="bg-accent-cyan text-black px-6 py-3 font-bold hover:opacity-80 inline-flex items-center gap-2">
                OPEN AI CHAT
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Risk Profile Display */}
          <div className="mt-12 bg-gray-100 dark:bg-gray-900 border-2 border-black dark:border-white p-6">
            <div className="text-center">
              <h3 className="text-lg font-bold text-black dark:text-white mb-2 text-brutal">
                AI-DETECTED RISK PROFILE: {analysis.riskProfile.toUpperCase()}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 font-mono">
                Based on your portfolio composition and asset allocation
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}