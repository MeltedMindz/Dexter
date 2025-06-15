'use client'

import { useState, useEffect } from 'react'
import { Brain, Activity, Zap, Eye } from 'lucide-react'

export function BrainWindow() {
  const [isVisible, setIsVisible] = useState(false)
  const [iframeLoaded, setIframeLoaded] = useState(false)
  const [showFallback, setShowFallback] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 500)
    
    // Show fallback after 3 seconds if iframe hasn't loaded
    const fallbackTimer = setTimeout(() => {
      if (!iframeLoaded) {
        setShowFallback(true)
      }
    }, 3000)
    
    return () => {
      clearTimeout(timer)
      clearTimeout(fallbackTimer)
    }
  }, [iframeLoaded])

  return (
    <section className="py-16 border-b-2 border-black dark:border-white bg-gradient-to-br from-black via-gray-900 to-black">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-4 mb-6">
            <Brain className="w-12 h-12 text-primary animate-pulse" />
            <h2 className="text-4xl md:text-5xl font-bold text-white text-brutal">
              WINDOW INTO THE BRAIN
            </h2>
            <Eye className="w-12 h-12 text-accent-cyan animate-pulse" />
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto font-mono">
            WATCH THE DEXBRAIN INTELLIGENCE NETWORK IN REAL-TIME • LIVE AGENT ACTIVITY • GLOBAL DATA SHARING
          </p>
        </div>

        {/* Main Brain Window */}
        <div className={`relative transition-all duration-1000 ${isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
          {/* Glowing Frame */}
          <div className="relative bg-black border-4 border-primary shadow-[0_0_50px_rgba(255,255,0,0.3)] rounded-lg overflow-hidden">
            
            {/* Header Bar */}
            <div className="bg-gradient-to-r from-primary via-accent-yellow to-primary p-4 border-b-2 border-black">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full border border-black"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full border border-black"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full border border-black"></div>
                  </div>
                  <span className="text-black font-bold text-sm font-mono">
                    DEXBRAIN_INTELLIGENCE_NETWORK.exe
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-black animate-pulse" />
                  <span className="text-black font-bold text-xs font-mono">LIVE</span>
                </div>
              </div>
            </div>

            {/* Status Bar */}
            <div className="bg-gradient-to-r from-gray-900 to-black p-3 border-b border-gray-700">
              <div className="flex items-center justify-center gap-6 text-sm font-mono">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-green-400">NETWORK: ONLINE</span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="w-3 h-3 text-yellow-400" />
                  <span className="text-yellow-400">157.90.230.148</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                  <span className="text-cyan-400">AGENTS: ACTIVE</span>
                </div>
              </div>
            </div>

            {/* Terminal Window */}
            <div className="relative bg-black min-h-[500px]">
              {!showFallback ? (
                <iframe 
                  src="http://157.90.230.148/live-monitor"
                  className="w-full h-[500px] border-0"
                  title="DexBrain Intelligence Network Live Monitor"
                  onLoad={() => setIframeLoaded(true)}
                  style={{
                    background: 'transparent',
                    overflow: 'hidden'
                  }}
                />
              ) : (
                <div className="w-full h-[500px] bg-black text-green-400 font-mono text-sm p-4 overflow-hidden">
                  <SimulatedBrainActivity />
                </div>
              )}
              
              {/* Loading overlay */}
              {!iframeLoaded && !showFallback && (
                <div className="absolute inset-0 bg-black flex items-center justify-center">
                  <div className="text-primary text-center">
                    <Brain className="w-12 h-12 mx-auto mb-4 animate-pulse" />
                    <div className="text-lg font-mono">CONNECTING TO DEXBRAIN...</div>
                    <div className="text-sm text-gray-400 mt-2">Loading intelligence network...</div>
                  </div>
                </div>
              )}
              
              {/* Overlay effects */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Scanlines effect */}
                <div 
                  className="absolute inset-0 opacity-10"
                  style={{
                    background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,0,0.1) 2px, rgba(0,255,0,0.1) 4px)'
                  }}
                ></div>
                
                {/* Corner glow effects */}
                <div className="absolute top-0 left-0 w-32 h-32 bg-primary opacity-20 blur-3xl"></div>
                <div className="absolute top-0 right-0 w-32 h-32 bg-accent-cyan opacity-20 blur-3xl"></div>
                <div className="absolute bottom-0 left-0 w-32 h-32 bg-accent-magenta opacity-20 blur-3xl"></div>
                <div className="absolute bottom-0 right-0 w-32 h-32 bg-accent-yellow opacity-20 blur-3xl"></div>
              </div>
            </div>

            {/* Footer Info */}
            <div className="bg-gradient-to-r from-gray-900 to-black p-4 border-t border-gray-700">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
                <div className="text-center">
                  <div className="text-primary font-bold">GLOBAL INTELLIGENCE</div>
                  <div className="text-gray-400">Real-time data sharing across agents</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-cyan font-bold">PERFORMANCE TRACKING</div>
                  <div className="text-gray-400">Live P&L and APR monitoring</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-yellow font-bold">MULTI-CHAIN</div>
                  <div className="text-gray-400">Base • Ethereum • Arbitrum • Solana</div>
                </div>
              </div>
            </div>
          </div>

          {/* Side Indicators */}
          <div className="absolute -left-8 top-1/2 transform -translate-y-1/2 space-y-4">
            <div className="w-4 h-4 bg-green-400 rounded-full animate-pulse shadow-lg"></div>
            <div className="w-4 h-4 bg-yellow-400 rounded-full animate-pulse shadow-lg" style={{animationDelay: '0.5s'}}></div>
            <div className="w-4 h-4 bg-cyan-400 rounded-full animate-pulse shadow-lg" style={{animationDelay: '1s'}}></div>
          </div>
          
          <div className="absolute -right-8 top-1/2 transform -translate-y-1/2 space-y-4">
            <div className="w-4 h-4 bg-accent-magenta rounded-full animate-pulse shadow-lg"></div>
            <div className="w-4 h-4 bg-red-400 rounded-full animate-pulse shadow-lg" style={{animationDelay: '0.5s'}}></div>
            <div className="w-4 h-4 bg-blue-400 rounded-full animate-pulse shadow-lg" style={{animationDelay: '1s'}}></div>
          </div>
        </div>

        {/* Bottom Description */}
        <div className="mt-12 text-center">
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-gray-900 border-2 border-primary p-6 rounded">
              <h3 className="text-xl font-bold text-primary mb-3 text-brutal">OPEN SOURCE INTELLIGENCE</h3>
              <p className="text-gray-300 text-sm font-mono">
                The DexBrain network operates as a public good, allowing any agent to contribute data and access collective intelligence for better trading decisions.
              </p>
            </div>
            <div className="bg-gray-900 border-2 border-accent-cyan p-6 rounded">
              <h3 className="text-xl font-bold text-accent-cyan mb-3 text-brutal">REAL-TIME MONITORING</h3>
              <p className="text-gray-300 text-sm font-mono">
                Watch live agent registrations, data submissions, performance updates, and cross-chain activity as the network processes millions in liquidity.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// Fallback component that simulates brain activity
function SimulatedBrainActivity() {
  const [logs, setLogs] = useState<string[]>([])

  useEffect(() => {
    const activities = [
      '🧠 DEXBRAIN INTELLIGENCE NETWORK - STATUS: ONLINE',
      '🤖 Active Agents: 12 | Requests/min: 34',
      '📊 Processing: Market Data • Liquidity Metrics • Performance Analytics',
      '🔍 Intelligence Queries: 47 | Quality Score: 92%',
      '🌐 Network Activity: Base • 18 pools monitored',
      '✅ NEW AGENT REGISTERED: agent_7834 | Type: Aggressive',
      '📈 DATA SUBMISSION: 0xa7c4e2f... • P&L: +18.3% • APR: 22.1%',
      '🔍 INTELLIGENCE QUERY: Best Pools | Response: 83 insights',
      '📊 MARKET ANALYSIS: ETH/USDC • Volatility 12.4% • TVL: $275K',
      '🏆 PERFORMANCE UPDATE: agent_9241 • Score: 89% | Rank: #14',
      '⛓️  BLOCKCHAIN: Base • Gas: 15 gwei • Block: #19847429',
      '🌐 API GET: /api/intelligence | Total: 1,389',
      '🔄 INTELLIGENCE SHARED: agent_5627 → Performance Data (Quality: 94.2%)',
    ]

    const addLog = () => {
      const timestamp = new Date().toTimeString().slice(0, 8)
      const activity = activities[Math.floor(Math.random() * activities.length)]
      const newLog = `[${timestamp}] ${activity}`
      
      setLogs(prev => {
        const newLogs = [...prev, newLog]
        return newLogs.slice(-20) // Keep only last 20 logs
      })
    }

    // Add initial logs
    addLog()
    addLog()
    addLog()

    // Continue adding logs
    const interval = setInterval(addLog, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="h-full overflow-y-auto space-y-1">
      <div className="text-primary font-bold text-center mb-4">
        🚀 DEXBRAIN INTELLIGENCE NETWORK SIMULATOR 🚀
      </div>
      {logs.map((log, index) => (
        <div key={index} className="animate-fadeIn">
          {log}
        </div>
      ))}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateX(-10px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-in;
        }
      `}</style>
    </div>
  )
}