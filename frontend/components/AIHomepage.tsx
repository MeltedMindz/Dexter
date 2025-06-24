'use client'

import { useState, useEffect } from 'react'
import { ArrowRight, Wallet, Brain, MessageCircle, FileText, Zap, Target, TrendingUp, Shield } from 'lucide-react'
import Link from 'next/link'
import { useAccount, useConnect } from 'wagmi'
import { AIChat } from './AIChat'
import { WalletAnalysis } from './WalletAnalysis'

export function AIHomepage() {
  const { address, isConnected } = useAccount()
  const { connect, connectors } = useConnect()
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [isChatOpen, setIsChatOpen] = useState(false)

  useEffect(() => {
    if (isConnected && address && !showAnalysis) {
      // Small delay to allow wallet connection to complete
      setTimeout(() => setShowAnalysis(true), 1000)
    }
  }, [isConnected, address, showAnalysis])

  if (isConnected && showAnalysis) {
    return <WalletAnalysis address={address} onBack={() => setShowAnalysis(false)} />
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black font-sans">
      {/* Hero Section - AI-Powered DeFi */}
      <section className="border-b-2 border-black dark:border-white bg-grid">
        <div className="max-w-6xl mx-auto px-6 py-16">
          <div className="text-center space-y-8">
            <div className="flex items-center justify-center gap-4 mb-6">
              <Brain className="w-16 h-16 text-primary animate-pulse" />
              <h1 className="text-6xl md:text-7xl font-bold text-black dark:text-white text-brutal">
                AI-POWERED DEFI
              </h1>
              <Zap className="w-16 h-16 text-accent-cyan animate-pulse" />
            </div>
            
            <p className="text-xl text-black dark:text-white max-w-4xl mx-auto font-mono">
              CONNECT YOUR WALLET FOR INSTANT AI PORTFOLIO ANALYSIS • PERSONALIZED LIQUIDITY STRATEGIES • REAL-TIME OPTIMIZATION
            </p>

            {/* Wallet Connection CTA */}
            <div className="flex flex-col items-center gap-6 mt-12">
              {!isConnected ? (
                <div className="space-y-4">
                  <button
                    onClick={() => connect({ connector: connectors[0] })}
                    className="bg-primary text-black px-12 py-4 border-2 border-black dark:border-white shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal inline-flex items-center gap-3 text-lg"
                  >
                    <Wallet className="w-6 h-6" />
                    CONNECT WALLET & GET AI ANALYSIS
                    <ArrowRight className="w-6 h-6" />
                  </button>
                  <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                    Instant analysis of your positions • No transaction required
                  </p>
                </div>
              ) : (
                <div className="bg-green-100 dark:bg-green-900 border-2 border-green-500 p-4 rounded">
                  <p className="text-green-800 dark:text-green-200 font-bold">
                    ✅ Wallet Connected! Analyzing your portfolio...
                  </p>
                </div>
              )}

              {/* Live AI Chat Preview */}
              <button
                onClick={() => setIsChatOpen(true)}
                className="bg-white dark:bg-black text-black dark:text-white px-8 py-3 border-2 border-black dark:border-white shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal inline-flex items-center gap-3"
              >
                <MessageCircle className="w-5 h-5" />
                TRY AI CHAT (NO WALLET NEEDED)
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works: 4-Step Visual Flow */}
      <section className="py-16 border-b-2 border-black dark:border-white">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white text-center mb-16 text-brutal">
            AI-INTEGRATED USER FLOW
          </h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            {/* Step 1: Wallet Connection */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 text-center relative">
              <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-black font-bold text-xl">1</span>
              </div>
              <Wallet className="w-12 h-12 text-accent-cyan mx-auto mb-4" />
              <h3 className="text-lg font-bold text-black dark:text-white mb-2 text-brutal">
                CONNECT WALLET
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                MetaMask, WalletConnect, or Coinbase Wallet
              </p>
              {/* Arrow */}
              <div className="hidden md:block absolute -right-6 top-1/2 transform -translate-y-1/2">
                <ArrowRight className="w-8 h-8 text-black dark:text-white" />
              </div>
            </div>

            {/* Step 2: AI Analysis */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 text-center relative">
              <div className="w-16 h-16 bg-accent-cyan rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-black font-bold text-xl">2</span>
              </div>
              <Brain className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-lg font-bold text-black dark:text-white mb-2 text-brutal">
                AI PORTFOLIO SCAN
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                Instant analysis of tokens, LP positions, and opportunities
              </p>
              <div className="hidden md:block absolute -right-6 top-1/2 transform -translate-y-1/2">
                <ArrowRight className="w-8 h-8 text-black dark:text-white" />
              </div>
            </div>

            {/* Step 3: Personalized Recommendations */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 text-center relative">
              <div className="w-16 h-16 bg-accent-magenta rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white font-bold text-xl">3</span>
              </div>
              <Target className="w-12 h-12 text-accent-yellow mx-auto mb-4" />
              <h3 className="text-lg font-bold text-black dark:text-white mb-2 text-brutal">
                SMART RECOMMENDATIONS
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                Best pools, strategies, and risk assessments for your assets
              </p>
              <div className="hidden md:block absolute -right-6 top-1/2 transform -translate-y-1/2">
                <ArrowRight className="w-8 h-8 text-black dark:text-white" />
              </div>
            </div>

            {/* Step 4: Choose Your Path */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 text-center">
              <div className="w-16 h-16 bg-accent-yellow rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-black font-bold text-xl">4</span>
              </div>
              <TrendingUp className="w-12 h-12 text-green-500 mx-auto mb-4" />
              <h3 className="text-lg font-bold text-black dark:text-white mb-2 text-brutal">
                CHOOSE YOUR PATH
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                Manual vaults or AI-managed with optional control
              </p>
            </div>
          </div>

          {/* Strategic Positioning */}
          <div className="mt-12 bg-gradient-to-r from-primary to-accent-cyan p-8 border-2 border-black dark:border-white">
            <div className="text-center">
              <h3 className="text-2xl font-bold text-black mb-4 text-brutal">
                STRATEGIC MARKET POSITIONING
              </h3>
              <p className="text-black font-mono italic text-lg max-w-4xl mx-auto">
                "Building early brand trust around AI in DeFi is critical—long before users fully trust AI to manage their liquidity. 
                We see this as a progressive adoption curve, and positioning Dexter now ensures we become the default when the ecosystem matures."
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* AI Features Preview */}
      <section className="py-16 border-b-2 border-black dark:border-white">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white text-center mb-16 text-brutal">
            AI-POWERED FEATURES
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* Portfolio Analysis */}
            <div className="bg-white dark:bg-black border-2 border-primary p-8">
              <Brain className="w-16 h-16 text-primary mb-6" />
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">
                PORTFOLIO ANALYSIS
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 font-mono">
                <li>• Detect LP tokens & staked positions</li>
                <li>• Analyze idle assets & opportunities</li>
                <li>• Risk assessment & recommendations</li>
                <li>• Real-time market conditions</li>
              </ul>
            </div>

            {/* Smart Recommendations */}
            <div className="bg-white dark:bg-black border-2 border-accent-cyan p-8">
              <Target className="w-16 h-16 text-accent-cyan mb-6" />
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">
                SMART RECOMMENDATIONS
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 font-mono">
                <li>• Best pools for your tokens</li>
                <li>• Personalized strategies (tight/wide range)</li>
                <li>• Expected APR & IL risk analysis</li>
                <li>• Gas optimization suggestions</li>
              </ul>
            </div>

            {/* Executive Summaries */}
            <div className="bg-white dark:bg-black border-2 border-accent-yellow p-8">
              <FileText className="w-16 h-16 text-accent-yellow mb-6" />
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">
                EXECUTIVE SUMMARIES
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 font-mono">
                <li>• Daily/weekly performance reports</li>
                <li>• Market condition insights</li>
                <li>• Portfolio optimization suggestions</li>
                <li>• Base ecosystem developments</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits & Trust Building */}
      <section className="py-16">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl font-bold text-black dark:text-white mb-8 text-brutal">
                OPTIONAL CONTROL • MAXIMUM BENEFIT
              </h2>
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <Shield className="w-8 h-8 text-green-500 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="text-lg font-bold text-black dark:text-white text-brutal">
                      NO FORCED AI MANAGEMENT
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 font-mono">
                      Use AI insights to build manual vaults or opt into AI-managed strategies
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Brain className="w-8 h-8 text-primary flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="text-lg font-bold text-black dark:text-white text-brutal">
                      ALWAYS-ON AI ASSISTANCE
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 font-mono">
                      Chat with AI about pools, risks, and strategies regardless of vault type
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <TrendingUp className="w-8 h-8 text-accent-cyan flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="text-lg font-bold text-black dark:text-white text-brutal">
                      PROGRESSIVE ADOPTION
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 font-mono">
                      Start manual, graduate to AI-assisted, evolve to fully automated
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-8">
                <Link 
                  href="/vaults"
                  className="bg-primary text-black px-8 py-4 border-2 border-black dark:border-white shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal inline-flex items-center gap-3"
                >
                  EXPLORE VAULTS
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </div>
            </div>

            <div className="bg-gray-100 dark:bg-gray-900 border-2 border-black dark:border-white p-8">
              <h3 className="text-xl font-bold text-black dark:text-white mb-6 text-brutal">
                TRUST THROUGH TRANSPARENCY
              </h3>
              <div className="space-y-4 font-mono text-sm">
                <div className="bg-white dark:bg-black p-4 border border-gray-300 dark:border-gray-700">
                  <span className="text-green-600">AI Recommendation:</span><br />
                  "ETH/USDC 0.3% pool shows 18.5% APR with moderate IL risk based on current volatility patterns"
                </div>
                <div className="bg-white dark:bg-black p-4 border border-gray-300 dark:border-gray-700">
                  <span className="text-blue-600">Your Choice:</span><br />
                  ✓ Create manual vault with these parameters<br />
                  ✓ Let AI manage and rebalance automatically<br />
                  ✓ Ask questions before deciding
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* AI Chat Component */}
      {isChatOpen && (
        <AIChat 
          isOpen={isChatOpen} 
          onClose={() => setIsChatOpen(false)} 
          walletAddress={address}
        />
      )}
    </div>
  )
}