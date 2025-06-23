'use client'

import { useState, useEffect, useRef } from 'react'
import { Brain, Activity, Zap, Eye } from 'lucide-react'

interface LogEntry {
  type: 'log' | 'error' | 'history' | 'heartbeat' | 'vault_strategy' | 'vault_optimization' | 'compound_success' | 'compound_opportunities' | 'vault_intelligence' | 'intelligence_feed';
  data?: string;
  message?: string;
  timestamp: Date;
  metadata?: any;
  level?: string;
  module?: string;
}

export function BrainWindow() {
  const [isVisible, setIsVisible] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const pollRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 500)
    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    let pollInterval: NodeJS.Timeout;
    
    // Fetch logs from DexBrain API
    const fetchLogs = async () => {
      try {
        // Use internal API (proxied through Vercel to avoid CORS/HTTPS issues)
        const response = await fetch('/api/logs', {
          method: 'GET',
          cache: 'no-store'
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log('API Response:', data); // Debug log
        
        if (data.success && data.logs && Array.isArray(data.logs)) {
          console.log('✅ Fetched logs from Dexter agent:', data.logs.length, 'entries');
          setIsConnected(true);
          
          // Transform logs to expected format
          const transformedLogs = data.logs.map((log: any) => ({
            type: log.type || 'log',
            data: log.data || log.message || 'No data',
            timestamp: new Date(log.timestamp || new Date())
          }));
          
          setLogs(transformedLogs.slice(-50)); // Keep last 50 entries
        } else {
          console.warn('❌ Failed to fetch logs or invalid format:', data.error || 'Unknown error');
          setIsConnected(false);
          
          // Use fallback logs if provided
          if (data.logs && Array.isArray(data.logs) && data.logs.length > 0) {
            console.log('📋 Using fallback logs:', data.logs.length, 'entries');
            const fallbackLogs = data.logs.map((log: any) => ({
              type: log.type || 'log',
              data: log.data || log.message || 'No data',
              timestamp: new Date(log.timestamp || new Date())
            }));
            setLogs(fallbackLogs.slice(-50));
          } else {
            // Show structured demo logs with vault intelligence
            console.log('📝 No real data available, showing demo vault intelligence logs');
            setLogs([
              {
                type: 'vault_strategy',
                data: '[VaultMLEngine] Strategy prediction complete | Recommended: gamma_balanced | Confidence: 87% | Expected APR: 18.5% | Ranges: 2',
                timestamp: new Date(Date.now() - 5 * 60 * 1000)
              },
              {
                type: 'compound_success',
                data: '[CompoundService] COMPOUND_SUCCESS | Token ID: 12345 | TX Hash: 0xabc...def | Gas Used: 150,000 | Net Profit: $45.67',
                timestamp: new Date(Date.now() - 3 * 60 * 1000)
              },
              {
                type: 'vault_intelligence',
                data: '[DexBrain] Generated vault intelligence for 0x123...abc | Strategy: gamma_balanced | Confidence: 87% | Compound Opportunities: 5',
                timestamp: new Date(Date.now() - 2 * 60 * 1000)
              },
              {
                type: 'compound_opportunities',
                data: '[CompoundService] Found 15 compound opportunities | Total Profit Potential: $1,234.56 | Top Priority Score: 0.89',
                timestamp: new Date(Date.now() - 7 * 60 * 1000)
              },
              {
                type: 'vault_optimization',
                data: '[GammaStyleOptimizer] Dual position optimization complete | Base Range: [98000, 102000] (75%) | Limit Range: [99500, 100500] (25%)',
                timestamp: new Date(Date.now() - 8 * 60 * 1000)
              },
              {
                type: 'intelligence_feed',
                data: '[DexBrain] Intelligence served to agent dexter_agent_1 | Insights: 45 | Predictions: 3 | Quality Score: 92%',
                timestamp: new Date(Date.now() - 6 * 60 * 1000)
              },
              {
                type: 'error',
                data: '[Network] Unable to connect to live DexBrain API - showing demo data',
                timestamp: new Date()
              },
              {
                type: 'log',
                data: '[System] Vault infrastructure integration active • AI strategies enabled',
                timestamp: new Date(Date.now() - 3000)
              }
            ]);
          }
        }
      } catch (error) {
        console.error('🚨 Failed to fetch logs:', error);
        setIsConnected(false);
        
        // Show error state with connection info only
        setLogs([
          {
            type: 'error',
            data: '[Network] Connection to DexBrain API failed - retrying...',
            timestamp: new Date()
          },
          {
            type: 'log',
            data: '[System] Attempting to reconnect to vault intelligence network',
            timestamp: new Date(Date.now() - 3000)
          }
        ]);
      }
    };

    // Initial fetch
    fetchLogs();
    
    // Set up polling every 3 seconds
    pollInterval = setInterval(fetchLogs, 3000);

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [])

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      // Use scrollIntoView with block: 'nearest' to prevent full page scroll
      logsEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'nearest',
        inline: 'nearest'
      })
    }
  }, [logs, autoScroll])

  const formatLogLine = (log: LogEntry) => {
    const line = log.data || log.message || 'No data'
    // Remove ANSI color codes if present
    const cleanLine = line.replace(/\x1b\[[0-9;]*m/g, '')
    
    // Get type-specific styling
    const getLogStyle = (type: string, level?: string) => {
      if (type === 'error' || level === 'ERROR') {
        return 'text-red-400'
      } else if (type === 'vault_strategy') {
        return 'text-cyan-400'
      } else if (type === 'compound_success') {
        return 'text-green-400'
      } else if (type === 'compound_opportunities') {
        return 'text-yellow-400'
      } else if (type === 'vault_intelligence') {
        return 'text-purple-400'
      } else if (type === 'vault_optimization') {
        return 'text-blue-400'
      } else if (type === 'intelligence_feed') {
        return 'text-orange-400'
      } else if (cleanLine.includes('WARN') || cleanLine.includes('warning')) {
        return 'text-yellow-400'
      } else if (cleanLine.includes('INFO') || cleanLine.includes('Starting')) {
        return 'text-cyan-400'
      } else if (cleanLine.includes('Success') || cleanLine.includes('✓')) {
        return 'text-green-400'
      }
      return 'text-green-400'
    }
    
    const styleClass = getLogStyle(log.type, log.level)
    
    // Add module prefix if available
    const modulePrefix = log.module ? `[${log.module}] ` : ''
    
    return (
      <span className={styleClass}>
        {modulePrefix}{cleanLine}
      </span>
    )
  }

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
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => setAutoScroll(!autoScroll)}
                    className={`flex items-center gap-2 px-2 py-1 rounded border ${
                      autoScroll 
                        ? 'bg-green-500 border-green-700 text-black' 
                        : 'bg-gray-500 border-gray-700 text-white'
                    } font-bold text-xs font-mono transition-colors`}
                    title={autoScroll ? 'Auto-scroll is ON' : 'Auto-scroll is OFF'}
                  >
                    {autoScroll ? '▼ AUTO' : '|| MANUAL'}
                  </button>
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-black animate-pulse" />
                    <span className="text-black font-bold text-xs font-mono">LIVE</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Status Bar */}
            <div className="bg-gradient-to-r from-gray-900 to-black p-3 border-b border-gray-700">
              <div className="flex items-center justify-center gap-6 text-sm font-mono">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
                  <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
                    NETWORK: {isConnected ? 'ONLINE' : 'CONNECTING...'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="w-3 h-3 text-yellow-400" />
                  <span className="text-yellow-400">5.78.71.231</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                  <span className="text-cyan-400">DEXTER: ACTIVE</span>
                </div>
              </div>
            </div>

            {/* Terminal Window */}
            <div className="relative bg-black h-[500px] overflow-hidden">
              <div 
                className="h-full overflow-y-auto p-4 font-mono text-xs"
                onWheel={(e) => e.stopPropagation()}
                onScroll={(e) => e.stopPropagation()}
                style={{ overscrollBehavior: 'contain' }}>
                {logs.length === 0 ? (
                  <div className="text-center py-20">
                    <Brain className="w-12 h-12 text-primary mx-auto mb-4 animate-pulse" />
                    <div className="text-primary">CONNECTING TO DEXTER AGENT...</div>
                    <div className="text-gray-400 text-xs mt-2">Establishing secure connection to AI brain...</div>
                  </div>
                ) : (
                  <>
                    {logs.map((log, index) => (
                      <div key={index} className="mb-1">
                        <div className="leading-relaxed flex items-start gap-2">
                          <span className="text-gray-500 text-xs min-w-[60px]">
                            {log.timestamp.toLocaleTimeString('en-US', { 
                              hour12: false, 
                              hour: '2-digit', 
                              minute: '2-digit', 
                              second: '2-digit' 
                            })}
                          </span>
                          <div className="flex-1">
                            {formatLogLine(log)}
                          </div>
                        </div>
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </>
                )}
              </div>
              
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
                  <div className="text-primary font-bold">VAULT INTELLIGENCE</div>
                  <div className="text-gray-400">AI-powered strategy optimization</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-cyan font-bold">COMPOUND TRACKING</div>
                  <div className="text-gray-400">Live compound opportunities & execution</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-yellow font-bold">MULTI-RANGE</div>
                  <div className="text-gray-400">Gamma-style dual position management</div>
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