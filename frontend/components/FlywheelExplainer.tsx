'use client'

import { ArrowRight, Users, Zap, TrendingUp, Coins, Shield, Target, BarChart3 } from 'lucide-react'
import Link from 'next/link'

export function FlywheelExplainer() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center space-y-6 mb-16">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white">
          AI-Powered Liquidity Management
        </h1>
        <p className="text-xl text-slate-600 dark:text-slate-400 max-w-3xl mx-auto">
          Auto-compound your Uniswap V3 positions with performance-based fees. 
          Only pay when you profit with advanced AI optimization.
        </p>
        <div className="flex flex-wrap justify-center gap-4 mt-8">
          <Link 
            href="/create"
            className="bg-primary hover:bg-primary-600 text-white px-8 py-3 rounded-lg font-semibold transition-colors inline-block"
          >
            Get Started
          </Link>
          <Link 
            href="/about"
            className="border border-slate-300 dark:border-white/20 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/5 px-8 py-3 rounded-lg font-semibold transition-colors inline-block"
          >
            Learn More
          </Link>
        </div>
      </div>

      {/* How It Works */}
      <div className="bg-white dark:bg-dark-700 rounded-3xl border border-slate-200 dark:border-white/10 shadow-lg dark:shadow-xl p-8 mb-16">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white text-center mb-12">
          How Dexter Works
        </h2>
        
        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* Step 1: Auto-Compound */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-primary/10 dark:bg-primary/20 rounded-full flex items-center justify-center mx-auto">
              <Zap className="w-8 h-8 text-primary" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Auto-Compound</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                AI optimizes compounding timing for maximum returns
              </p>
            </div>
          </div>

          {/* Step 2: Performance Fees */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-success/10 dark:bg-success/20 rounded-full flex items-center justify-center mx-auto">
              <TrendingUp className="w-8 h-8 text-success" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Performance Based</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Only pay 8% fee on profits - no fees on losses
              </p>
            </div>
          </div>

          {/* Step 3: Staker Rewards */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-warning/10 dark:bg-warning/20 rounded-full flex items-center justify-center mx-auto">
              <Coins className="w-8 h-8 text-warning" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Staker Rewards</h3>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                100% of protocol fees go to $DEX token stakers
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 shadow-sm dark:shadow-xl">
          <div className="w-12 h-12 bg-primary/10 dark:bg-primary/20 rounded-lg flex items-center justify-center mb-4">
            <Zap className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">AI Optimization</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm">
            Advanced algorithms optimize compounding timing for maximum returns
          </p>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 shadow-sm dark:shadow-xl">
          <div className="w-12 h-12 bg-success/10 dark:bg-success/20 rounded-lg flex items-center justify-center mb-4">
            <Shield className="w-6 h-6 text-success" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">Performance Based</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm">
            Only pay fees when you profit - no losses, no fees
          </p>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 shadow-sm dark:shadow-xl">
          <div className="w-12 h-12 bg-warning/10 dark:bg-warning/20 rounded-lg flex items-center justify-center mb-4">
            <BarChart3 className="w-6 h-6 text-warning" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">Real-time Analytics</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm">
            Track performance with detailed charts and metrics
          </p>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 shadow-sm dark:shadow-xl">
          <div className="w-12 h-12 bg-primary/10 dark:bg-primary/20 rounded-lg flex items-center justify-center mb-4">
            <Shield className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">Any V3 Position</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm">
            Works with any Uniswap V3 position across all supported pairs
          </p>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 shadow-sm dark:shadow-xl">
          <div className="w-12 h-12 bg-success/10 dark:bg-success/20 rounded-lg flex items-center justify-center mb-4">
            <Target className="w-6 h-6 text-success" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">Gas Optimized</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm">
            Efficient batch operations minimize gas costs for all users
          </p>
        </div>

        <div className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 shadow-sm dark:shadow-xl">
          <div className="w-12 h-12 bg-warning/10 dark:bg-warning/20 rounded-lg flex items-center justify-center mb-4">
            <Users className="w-6 h-6 text-warning" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">Community Driven</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm">
            100% of fees go to $DEX stakers - truly community owned
          </p>
        </div>
      </div>
    </div>
  )
}