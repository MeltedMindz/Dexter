'use client'

import { ArrowRight, Users, Zap, TrendingUp, Coins, Shield, Target, BarChart3 } from 'lucide-react'
import Link from 'next/link'
import { BrainWindow } from './BrainWindow'

export function FlywheelExplainer() {
  return (
    <div className="min-h-screen bg-white dark:bg-black font-sans">
      {/* Hero Section */}
      <section className="border-b-2 border-black dark:border-white bg-grid">
        <div className="max-w-6xl mx-auto px-6 py-16">
          <div className="text-center space-y-8">
            <h1 className="text-6xl md:text-7xl font-bold text-black dark:text-white text-brutal">
              AI-POWERED LIQUIDITY MANAGEMENT
            </h1>
            <p className="text-xl text-black dark:text-white max-w-4xl mx-auto font-mono">
              AUTO-COMPOUND YOUR UNISWAP V3 POSITIONS WITH PERFORMANCE-BASED FEES. ONLY PAY WHEN YOU PROFIT WITH ADVANCED AI OPTIMIZATION.
            </p>
            <div className="flex flex-wrap justify-center gap-6 mt-12">
              <Link 
                href="/create"
                className="bg-primary text-black px-12 py-4 border-2 border-black dark:border-white shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal inline-block"
              >
                GET STARTED
              </Link>
              <Link 
                href="/about"
                className="bg-white dark:bg-black text-black dark:text-white px-12 py-4 border-2 border-black dark:border-white shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal inline-block"
              >
                LEARN MORE
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Brain Window - Prominently placed after hero */}
      <BrainWindow />

      {/* How It Works */}
      <section className="py-16 border-b-2 border-black dark:border-white">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white text-center mb-16 text-brutal">
            HOW DEXTER WORKS
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* Step 1: Auto-Compound */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-8 text-center">
              <div className="w-16 h-16 bg-primary border-2 border-black dark:border-white flex items-center justify-center mx-auto mb-6">
                <Zap className="w-8 h-8 text-black" />
              </div>
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">AUTO-COMPOUND</h3>
              <p className="text-black dark:text-white font-mono">
                AI OPTIMIZES COMPOUNDING TIMING FOR MAXIMUM RETURNS
              </p>
            </div>

            {/* Step 2: Performance Fees */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-8 text-center">
              <div className="w-16 h-16 bg-accent-yellow border-2 border-black dark:border-white flex items-center justify-center mx-auto mb-6">
                <TrendingUp className="w-8 h-8 text-black" />
              </div>
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">PERFORMANCE BASED</h3>
              <p className="text-black dark:text-white font-mono">
                ONLY PAY 8% FEE ON PROFITS - NO FEES ON LOSSES
              </p>
            </div>

            {/* Step 3: Staker Rewards */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-8 text-center">
              <div className="w-16 h-16 bg-accent-cyan border-2 border-black dark:border-white flex items-center justify-center mx-auto mb-6">
                <Coins className="w-8 h-8 text-black" />
              </div>
              <h3 className="text-xl font-bold text-black dark:text-white mb-4 text-brutal">STAKER REWARDS</h3>
              <p className="text-black dark:text-white font-mono">
                100% OF PROTOCOL FEES GO TO $DEX TOKEN STAKERS
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white text-center mb-16 text-brutal">
            FEATURES
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
              <div className="w-12 h-12 bg-primary border-2 border-black dark:border-white flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-black" />
              </div>
              <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">AI OPTIMIZATION</h3>
              <p className="text-black dark:text-white text-sm font-mono">
                ADVANCED ALGORITHMS OPTIMIZE COMPOUNDING TIMING FOR MAXIMUM RETURNS
              </p>
            </div>

            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
              <div className="w-12 h-12 bg-accent-yellow border-2 border-black dark:border-white flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-black" />
              </div>
              <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">PERFORMANCE BASED</h3>
              <p className="text-black dark:text-white text-sm font-mono">
                ONLY PAY FEES WHEN YOU PROFIT - NO LOSSES, NO FEES
              </p>
            </div>

            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
              <div className="w-12 h-12 bg-accent-cyan border-2 border-black dark:border-white flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-black" />
              </div>
              <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">REAL-TIME ANALYTICS</h3>
              <p className="text-black dark:text-white text-sm font-mono">
                TRACK PERFORMANCE WITH DETAILED CHARTS AND METRICS
              </p>
            </div>

            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
              <div className="w-12 h-12 bg-accent-magenta border-2 border-black dark:border-white flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-black" />
              </div>
              <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">ANY V3 POSITION</h3>
              <p className="text-black dark:text-white text-sm font-mono">
                WORKS WITH ANY UNISWAP V3 POSITION ACROSS ALL SUPPORTED PAIRS
              </p>
            </div>

            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
              <div className="w-12 h-12 bg-primary border-2 border-black dark:border-white flex items-center justify-center mb-4">
                <Target className="w-6 h-6 text-black" />
              </div>
              <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">GAS OPTIMIZED</h3>
              <p className="text-black dark:text-white text-sm font-mono">
                EFFICIENT BATCH OPERATIONS MINIMIZE GAS COSTS FOR ALL USERS
              </p>
            </div>

            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6">
              <div className="w-12 h-12 bg-accent-yellow border-2 border-black dark:border-white flex items-center justify-center mb-4">
                <Users className="w-6 h-6 text-black" />
              </div>
              <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">COMMUNITY DRIVEN</h3>
              <p className="text-black dark:text-white text-sm font-mono">
                100% OF FEES GO TO $DEX STAKERS - TRULY COMMUNITY OWNED
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}