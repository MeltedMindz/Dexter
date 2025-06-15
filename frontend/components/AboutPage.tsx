'use client'

import { ArrowRight, Code, Zap, TrendingUp, Shield, Users, Target, Globe, Network, Coins } from 'lucide-react'
import Link from 'next/link'

export function AboutPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-black font-mono">
      {/* Hero Section */}
      <section className="border-b-2 border-black dark:border-white bg-grid">
        <div className="max-w-6xl mx-auto px-6 py-16">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-6xl font-bold text-black dark:text-white mb-6 text-brutal">
                DEXTER
              </h1>
              <p className="text-2xl text-gray-700 dark:text-gray-300 mb-4 font-display">
                TWO-PART DEFI REVOLUTION
              </p>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-6 leading-relaxed">
                <span className="font-bold text-black dark:text-white">$DEX Platform:</span> Stake tokens, earn revenue share from our liquidity management protocol.
              </p>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
                <span className="font-bold text-black dark:text-white">Open Source Network:</span> Free agent framework with shared intelligence brain for the entire DeFi ecosystem.
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

      {/* Dual Architecture Overview */}
      <section className="py-20 border-b-2 border-black dark:border-white bg-gray-50 dark:bg-gray-950">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-black dark:text-white mb-12 text-brutal text-center">
            THE DEXTER ECOSYSTEM
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-12">
            {/* $DEX Token Platform */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-8">
              <div className="flex items-center gap-4 mb-6">
                <Coins className="w-10 h-10 text-primary" />
                <h3 className="text-2xl font-bold text-black dark:text-white text-brutal">
                  $DEX TOKEN PLATFORM
                </h3>
              </div>
              
              <div className="space-y-4 mb-6">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-bold text-black dark:text-white">Stake $DEX tokens</span> to earn revenue share from our liquidity management protocol
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-bold text-black dark:text-white">8% performance fees</span> from all managed positions flow to stakers
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-bold text-black dark:text-white">Sustainable yield</span> directly tied to protocol usage and success
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-600 dark:text-gray-400">
                    <span className="font-bold text-black dark:text-white">Governance rights</span> over protocol parameters and upgrades
                  </p>
                </div>
              </div>
              
              <Link 
                href="/stake"
                className="inline-flex items-center gap-3 bg-primary text-black px-6 py-3 border-2 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal text-sm"
              >
                STAKE $DEX
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>

            {/* Open Source Network */}
            <div className="bg-black dark:bg-white text-white dark:text-black border-2 border-black dark:border-white p-8">
              <div className="flex items-center gap-4 mb-6">
                <Globe className="w-10 h-10 text-primary" />
                <h3 className="text-2xl font-bold text-brutal">
                  OPEN SOURCE NETWORK
                </h3>
              </div>
              
              <div className="space-y-4 mb-6">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-300 dark:text-gray-600">
                    <span className="font-bold text-white dark:text-black">Free agent framework</span> - anyone can deploy liquidity management bots
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-300 dark:text-gray-600">
                    <span className="font-bold text-white dark:text-black">Shared intelligence brain</span> - all agents contribute and benefit from collective data
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-300 dark:text-gray-600">
                    <span className="font-bold text-white dark:text-black">Cross-chain compatible</span> - supports Ethereum, Base, Solana, and more
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-primary mt-3 flex-shrink-0"></div>
                  <p className="text-gray-300 dark:text-gray-600">
                    <span className="font-bold text-white dark:text-black">Public good</span> - improves DeFi efficiency for the entire ecosystem
                  </p>
                </div>
              </div>
              
              <a
                href="https://github.com/MeltedMindz/Dexter"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-3 bg-white dark:bg-black text-black dark:text-white px-6 py-3 border-2 border-white dark:border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal text-sm"
              >
                VIEW SOURCE
                <Code className="w-4 h-4" />
              </a>
            </div>
          </div>
          
          {/* Intelligence Network Diagram */}
          <div className="mt-16 bg-white dark:bg-black border-2 border-black dark:border-white p-8">
            <h3 className="text-2xl font-bold text-black dark:text-white mb-8 text-brutal text-center">
              GLOBAL INTELLIGENCE NETWORK
            </h3>
            
            <div className="grid md:grid-cols-3 gap-8 items-center">
              <div className="text-center">
                <div className="bg-primary p-6 border-2 border-black dark:border-white mb-4 mx-auto w-fit">
                  <Network className="w-8 h-8 text-black" />
                </div>
                <h4 className="font-bold text-black dark:text-white mb-2">DEPLOY AGENTS</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Anyone deploys Dexter agents using open source code
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-accent-cyan p-6 border-2 border-black dark:border-white mb-4 mx-auto w-fit">
                  <Target className="w-8 h-8 text-black" />
                </div>
                <h4 className="font-bold text-black dark:text-white mb-2">SHARE DATA</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  All agents feed performance data to the shared brain
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-accent-magenta p-6 border-2 border-black dark:border-white mb-4 mx-auto w-fit">
                  <TrendingUp className="w-8 h-8 text-black" />
                </div>
                <h4 className="font-bold text-black dark:text-white mb-2">IMPROVE TOGETHER</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Collective intelligence makes everyone better
                </p>
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
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
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

            <div className="bg-primary text-black p-8 border-2 border-black dark:border-white">
              <h3 className="text-2xl font-bold mb-6 text-brutal">
                INTELLIGENCE NETWORK
              </h3>
              <div className="space-y-3 font-mono">
                <div className="flex justify-between">
                  <span>PYTHON</span>
                  <span className="text-black">v3.11+</span>
                </div>
                <div className="flex justify-between">
                  <span>FLASK API</span>
                  <span className="text-black">REST</span>
                </div>
                <div className="flex justify-between">
                  <span>ASYNCIO</span>
                  <span className="text-black">ASYNC</span>
                </div>
                <div className="flex justify-between">
                  <span>POSTGRESQL</span>
                  <span className="text-black">DATABASE</span>
                </div>
                <div className="flex justify-between">
                  <span>ML MODELS</span>
                  <span className="text-black">SKLEARN</span>
                </div>
              </div>
            </div>
          </div>

          {/* API Integration Example */}
          <div className="mt-12 bg-gray-900 text-green-400 p-8 border-2 border-black dark:border-white">
            <h3 className="text-xl font-bold mb-6 text-brutal text-white">
              QUICK START: DEPLOY YOUR AGENT
            </h3>
            <div className="font-mono text-sm space-y-2">
              <div><span className="text-gray-500"># 1. Register for API key</span></div>
              <div>curl -X POST https://api.dexteragent.com/register \</div>
              <div className="ml-4">-d {`'{"agent_id": "my-agent", "metadata": {}}'`}</div>
              <div><span className="text-gray-500"># 2. Clone open source repo</span></div>
              <div>git clone https://github.com/MeltedMindz/Dexter.git</div>
              <div><span className="text-gray-500"># 3. Configure your agent</span></div>
              <div>export DEXBRAIN_API_KEY={`"dx_your_key_here"`}</div>
              <div><span className="text-gray-500"># 4. Start contributing to the network!</span></div>
              <div>python dexter-liquidity/main.py</div>
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