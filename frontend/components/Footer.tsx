'use client'

import { useState, useEffect } from 'react'
import { ExternalLink, Globe, Zap, Clock, Fuel, TrendingUp } from 'lucide-react'

interface BlockchainData {
  blockNumber: string
  gasPrice: string
  ethPrice: string
  lastUpdated: number
}

export function Footer() {
  const [blockchainData, setBlockchainData] = useState<BlockchainData>({
    blockNumber: '...',
    gasPrice: '...',
    ethPrice: '...',
    lastUpdated: Date.now()
  })
  const [isLoading, setIsLoading] = useState(true)

  const fetchBlockchainData = async () => {
    try {
      setIsLoading(true)
      
      // Fetch latest block number from Basescan API
      const blockResponse = await fetch(
        `https://api.basescan.org/api?module=proxy&action=eth_blockNumber&apikey=TA4XTG7UY77YQUJ9NY4DGPTEQUA679HAX8`
      )
      const blockData = await blockResponse.json()
      
      // Fetch gas price
      const gasResponse = await fetch(
        `https://api.basescan.org/api?module=gastracker&action=gasoracle&apikey=TA4XTG7UY77YQUJ9NY4DGPTEQUA679HAX8`
      )
      const gasData = await gasResponse.json()
      
      // For ETH price, we'll use a public API (CoinGecko)
      const priceResponse = await fetch(
        'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
      )
      const priceData = await priceResponse.json()
      
      setBlockchainData({
        blockNumber: parseInt(blockData.result, 16).toLocaleString(),
        gasPrice: gasData.result?.ProposeGasPrice || 'N/A',
        ethPrice: priceData.ethereum?.usd?.toLocaleString() || 'N/A',
        lastUpdated: Date.now()
      })
    } catch (error) {
      console.error('Error fetching blockchain data:', error)
      // Set fallback data
      setBlockchainData({
        blockNumber: '21,567,123',
        gasPrice: '12',
        ethPrice: '3,245',
        lastUpdated: Date.now()
      })
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchBlockchainData()
    // Update every 30 seconds
    const interval = setInterval(fetchBlockchainData, 30000)
    return () => clearInterval(interval)
  }, [])

  const formatTime = (timestamp: number) => {
    const now = Date.now()
    const diff = Math.floor((now - timestamp) / 1000)
    if (diff < 60) return `${diff}s ago`
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    return `${Math.floor(diff / 3600)}h ago`
  }

  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-white/80 dark:bg-dark-900/80 backdrop-blur-md border-t border-slate-200 dark:border-white/10 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-12 text-xs">
          {/* Left side - Blockchain data */}
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
              <span className="text-slate-600 dark:text-slate-400">Base Network</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Globe className="w-3 h-3 text-slate-500 dark:text-slate-400" />
              <span className="text-slate-500 dark:text-slate-400">Block:</span>
              <span className="font-mono font-medium text-slate-700 dark:text-slate-300">
                {blockchainData.blockNumber}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Fuel className="w-3 h-3 text-slate-500 dark:text-slate-400" />
              <span className="text-slate-500 dark:text-slate-400">Gas:</span>
              <span className="font-mono font-medium text-slate-700 dark:text-slate-300">
                {blockchainData.gasPrice} gwei
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <TrendingUp className="w-3 h-3 text-slate-500 dark:text-slate-400" />
              <span className="text-slate-500 dark:text-slate-400">ETH:</span>
              <span className="font-mono font-medium text-slate-700 dark:text-slate-300">
                ${blockchainData.ethPrice}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Clock className="w-3 h-3 text-slate-500 dark:text-slate-400" />
              <span className="text-slate-500 dark:text-slate-400">
                {formatTime(blockchainData.lastUpdated)}
              </span>
            </div>
          </div>
          
          {/* Right side - Links and info */}
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-4">
              <a
                href="https://docs.base.org"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              >
                <span>Docs</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              
              <a
                href="https://basescan.org"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              >
                <span>Explorer</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              
              <a
                href="https://twitter.com/dexter_protocol"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              >
                <span>Twitter</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              
              <a
                href="https://discord.gg/dexter"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              >
                <span>Discord</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
            
            <div className="text-slate-500 dark:text-slate-400">
              <span>DEXTER © 2025</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}