'use client'

import { ArrowRight, Code, Zap, TrendingUp, Shield, Users, Target } from 'lucide-react'
import Link from 'next/link'

export function AboutPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-black font-mono">
      {/* Hero Section */}
      <section className="border-b-2 border-black dark:border-white bg-grid">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-6xl font-bold text-black dark:text-white mb-6 text-brutal">
                DEXTER
              </h1>
              <p className="text-2xl text-gray-700 dark:text-gray-300 mb-8 font-display">
                AI-POWERED LIQUIDITY MANAGEMENT PROTOCOL
              </p>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
                Auto-compound your Uniswap V3 positions with performance-based fees. 
                Only pay when you profit with advanced AI optimization.
              </p>
              <Link 
                href="/create"
                className="inline-flex items-center gap-3 bg-primary text-black px-8 py-4 border-2 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal"
              >
                START BUILDING
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
            
            <div className="bg-black dark:bg-white text-white dark:text-black p-8 border-2 border-black dark:border-white">
              <div className="font-mono space-y-4">
                <div className="text-primary font-bold">STATUS: ACTIVE</div>
                <div>PROTOCOL: DEXTER V1.0</div>
                <div>NETWORK: BASE, ETHEREUM, ARBITRUM</div>
                <div>TVL: $2,451,872</div>
                <div>POSITIONS: 1,247</div>
                <div>USERS: 892</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 border-b-2 border-black dark:border-white">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white mb-12 text-brutal text-center">
            FEATURES
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: Zap,
                title: "AUTO-COMPOUND",
                description: "AI optimizes compounding timing for maximum returns",
                color: "primary"
              },
              {
                icon: TrendingUp,
                title: "PERFORMANCE FEES",
                description: "Only pay 8% fee on profits - no fees on losses",
                color: "accent-yellow"
              },
              {
                icon: Shield,
                title: "SECURE PROTOCOL",
                description: "Audited smart contracts with proven security",
                color: "accent-cyan"
              },
              {
                icon: Users,
                title: "COMMUNITY OWNED",
                description: "100% of fees go to $DEX token stakers",
                color: "accent-magenta"
              },
              {
                icon: Target,
                title: "GAS OPTIMIZED",
                description: "Efficient batch operations minimize costs",
                color: "primary"
              },
              {
                icon: Code,
                title: "OPEN SOURCE",
                description: "Transparent, verifiable, community-driven",
                color: "accent-yellow"
              }
            ].map((feature, index) => (
              <div 
                key={index}
                className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 hover:shadow-brutal transition-all duration-100"
              >
                <feature.icon className="w-8 h-8 text-black dark:text-white mb-4" />
                <h3 className="text-xl font-bold text-black dark:text-white mb-3 text-brutal">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Protocol Stats */}
      <section className="py-20 border-b-2 border-black dark:border-white bg-gray-100 dark:bg-gray-900">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white mb-12 text-brutal text-center">
            PROTOCOL METRICS
          </h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { label: "TOTAL VALUE LOCKED", value: "$2.4M", change: "+12.3%" },
              { label: "ACTIVE POSITIONS", value: "1,247", change: "+8.9%" },
              { label: "TOTAL USERS", value: "892", change: "+15.2%" },
              { label: "FEES EARNED", value: "$48.2K", change: "+24.1%" }
            ].map((stat, index) => (
              <div 
                key={index}
                className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 text-center"
              >
                <div className="text-3xl font-bold text-black dark:text-white mb-2 font-mono">
                  {stat.value}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-2 text-brutal">
                  {stat.label}
                </div>
                <div className="text-primary font-bold font-mono">
                  {stat.change}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white mb-12 text-brutal text-center">
            TECHNOLOGY STACK
          </h2>
          
          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-black dark:bg-white text-white dark:text-black p-8 border-2 border-black dark:border-white">
              <h3 className="text-2xl font-bold mb-6 text-brutal">
                SMART CONTRACTS
              </h3>
              <div className="space-y-3 font-mono">
                <div className="flex justify-between">
                  <span>SOLIDITY</span>
                  <span className="text-primary">v0.8.19</span>
                </div>
                <div className="flex justify-between">
                  <span>UNISWAP V3</span>
                  <span className="text-primary">INTEGRATED</span>
                </div>
                <div className="flex justify-between">
                  <span>OPENZEPPELIN</span>
                  <span className="text-primary">v4.9.0</span>
                </div>
                <div className="flex justify-between">
                  <span>CHAINLINK</span>
                  <span className="text-primary">ORACLES</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-8">
              <h3 className="text-2xl font-bold text-black dark:text-white mb-6 text-brutal">
                FRONTEND
              </h3>
              <div className="space-y-3 font-mono text-black dark:text-white">
                <div className="flex justify-between">
                  <span>NEXT.JS</span>
                  <span className="text-primary">v15.3.3</span>
                </div>
                <div className="flex justify-between">
                  <span>TYPESCRIPT</span>
                  <span className="text-primary">v5.0.0</span>
                </div>
                <div className="flex justify-between">
                  <span>WAGMI</span>
                  <span className="text-primary">v2.0.0</span>
                </div>
                <div className="flex justify-between">
                  <span>TAILWIND CSS</span>
                  <span className="text-primary">v3.4.0</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary border-t-2 border-black dark:border-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold text-black mb-6 text-brutal">
            READY TO OPTIMIZE?
          </h2>
          <p className="text-xl text-black mb-8">
            Start auto-compounding your Uniswap V3 positions with Dexter
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link 
              href="/create"
              className="bg-black text-white px-8 py-4 border-2 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal"
            >
              CREATE POSITION
            </Link>
            <a
              href="https://github.com/MeltedMindz/Dexter"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-white text-black px-8 py-4 border-2 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal"
            >
              VIEW CODE
            </a>
          </div>
        </div>
      </section>
    </div>
  )
}