import { BrainWindow } from '@/components/BrainWindow'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: 'DexBrain Intelligence Network | Live AI Activity',
  description: 'Watch the DexBrain intelligence network in real-time. Live agent activity, data sharing, and AI-powered liquidity management insights.',
  keywords: ['DexBrain', 'AI activity', 'live data', 'intelligence network', 'real-time analytics']
})

export default function BrainPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-black">
      {/* Page Header */}
      <section className="border-b-2 border-black dark:border-white bg-gradient-to-br from-black via-gray-900 to-black py-12">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-white text-brutal mb-4">
            DEXBRAIN INTELLIGENCE NETWORK
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto font-mono">
            LIVE VIEW INTO THE AI BRAIN • REAL-TIME AGENT ACTIVITY • GLOBAL DATA SHARING NETWORK
          </p>
        </div>
      </section>

      {/* Brain Window Component */}
      <BrainWindow />
      
      {/* Additional Info */}
      <section className="py-12 border-t-2 border-black dark:border-white bg-gray-50 dark:bg-gray-900">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-black border-2 border-primary p-6">
              <h3 className="text-xl font-bold text-black dark:text-white mb-3 text-brutal">
                WHAT YOU'RE SEEING
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 font-mono text-sm">
                <li>• Live AI service logs from production VPS (5.78.71.231)</li>
                <li>• Real-time vault strategy optimization decisions</li>
                <li>• ML model training and prediction cycles</li>
                <li>• Position compounding and fee collection events</li>
                <li>• Market analysis and regime detection updates</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-black border-2 border-accent-cyan p-6">
              <h3 className="text-xl font-bold text-black dark:text-white mb-3 text-brutal">
                AI SERVICES ACTIVE
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 font-mono text-sm">
                <li>• <span className="text-green-400">DexBrain</span> - Central intelligence coordination</li>
                <li>• <span className="text-cyan-400">Position Harvester</span> - Automated compounding</li>
                <li>• <span className="text-purple-400">Vault Processor</span> - ERC4626 management</li>
                <li>• <span className="text-yellow-400">Market Analyzer</span> - Real-time analysis</li>
                <li>• <span className="text-orange-400">ML Pipeline</span> - Continuous learning</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}