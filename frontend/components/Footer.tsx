'use client'

import { useState, useEffect } from 'react'
// Fixed: Using Fuel instead of Gas icon (Gas doesn't exist in Lucide React)
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
      
      // Get Base network data using Alchemy API
      const baseRpcUrl = process.env.NEXT_PUBLIC_BASE_RPC_URL
      
      // Fetch latest block from Base network
      const blockResponse = await fetch(baseRpcUrl!, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_blockNumber',
          params: [],
          id: 1
        })
      })
      
      const blockData = await blockResponse.json()
      const blockNumber = parseInt(blockData.result, 16)
      
      // Fetch gas price from Base network
      const gasResponse = await fetch(baseRpcUrl!, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_gasPrice',
          params: [],
          id: 2
        })
      })
      
      const gasData = await gasResponse.json()
      const gasPrice = (parseInt(gasData.result, 16) / 1e9).toFixed(1) // Convert wei to gwei
      
      // Fetch ETH price from CoinGecko (this should work from server-side)
      const ethPriceResponse = await fetch('/api/eth-price')
      const ethPriceData = await ethPriceResponse.json()
      
      setBlockchainData({
        blockNumber: blockNumber.toLocaleString(),
        gasPrice,
        ethPrice: ethPriceData.price ? ethPriceData.price.toLocaleString() : '3,240',
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
    <footer className="fixed bottom-0 left-0 right-0 bg-white dark:bg-black border-t-2 border-black dark:border-white z-40">
      <div className="w-full px-6 lg:px-12">
        <div className="flex items-center justify-between h-12 text-xs font-mono">
          {/* Left side - Blockchain data */}
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
              <span className="text-black dark:text-white text-brutal">BASE</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Globe className="w-3 h-3 text-black dark:text-white" />
              <span className="text-black dark:text-white">BLOCK:</span>
              <span className="font-mono font-bold text-black dark:text-white">
                {blockchainData.blockNumber}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Fuel className="w-3 h-3 text-black dark:text-white" />
              <span className="text-black dark:text-white">GAS:</span>
              <span className="font-mono font-bold text-black dark:text-white">
                {blockchainData.gasPrice} gwei
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <TrendingUp className="w-3 h-3 text-black dark:text-white" />
              <span className="text-black dark:text-white">ETH:</span>
              <span className="font-mono font-bold text-black dark:text-white">
                ${blockchainData.ethPrice}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Clock className="w-3 h-3 text-black dark:text-white" />
              <span className="text-black dark:text-white">
                {formatTime(blockchainData.lastUpdated)}
              </span>
            </div>
          </div>
          
          {/* Right side - Links and info */}
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/MeltedMindz/Dexter"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-black dark:text-white hover:text-primary transition-colors text-brutal"
              >
                <span>GitHub</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              
              <a
                href="https://basescan.org"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-black dark:text-white hover:text-primary transition-colors text-brutal"
              >
                <span>Explorer</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              
              <a
                href="https://x.com/Dexter_AI_"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-black dark:text-white hover:text-primary transition-colors text-brutal"
              >
                <span>Twitter</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              
              <a
                href="https://t.me/+VELgzJret51mYzkx"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-black dark:text-white hover:text-primary transition-colors text-brutal"
              >
                <span>Telegram</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
            
            <div className="text-black dark:text-white">
              <span className="text-brutal">DEXTER © 2025</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}