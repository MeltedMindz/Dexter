'use client'

import { useAccount, useConnect, useDisconnect } from 'wagmi'
import { useState } from 'react'
import { ChevronDown, Wallet, LogOut, Copy, ExternalLink } from 'lucide-react'

export function ConnectButton() {
  const { address, isConnected } = useAccount()
  const { connect, connectors } = useConnect()
  const { disconnect } = useDisconnect()
  const [showDropdown, setShowDropdown] = useState(false)
  const [showConnectors, setShowConnectors] = useState(false)

  const formatAddress = (addr: string) => 
    `${addr.slice(0, 6)}...${addr.slice(-4)}`

  const copyAddress = () => {
    if (address) {
      navigator.clipboard.writeText(address)
    }
  }

  if (!isConnected) {
    return (
      <div className="relative">
        <button
          onClick={() => setShowConnectors(!showConnectors)}
          className="bg-primary text-black px-4 py-2 border-2 border-black shadow-brutal font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] transition-all duration-150 flex items-center space-x-2 uppercase tracking-wider"
        >
          <Wallet className="w-5 h-5" />
          <span>Connect Wallet</span>
        </button>
        
        {showConnectors && (
          <div className="absolute right-0 mt-2 w-72 bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal z-[90]">
            <div className="p-3">
              <div className="px-3 py-2 border-b-2 border-black dark:border-white mb-3">
                <span className="text-sm font-bold text-black dark:text-white uppercase tracking-wider">
                  Select Wallet
                </span>
              </div>
              {connectors.map((connector) => (
                <button
                  key={connector.uid}
                  onClick={() => {
                    connect({ connector })
                    setShowConnectors(false)
                  }}
                  className="w-full text-left px-3 py-3 mb-2 border-2 border-black dark:border-white hover:bg-primary hover:border-black dark:hover:bg-primary dark:hover:border-black transition-all duration-150 flex items-center space-x-3 bg-white dark:bg-gray-900 group"
                >
                  <div className="w-10 h-10 bg-black dark:bg-white flex items-center justify-center group-hover:bg-white dark:group-hover:bg-black transition-colors duration-150">
                    <Wallet className="w-5 h-5 text-white dark:text-black group-hover:text-black dark:group-hover:text-white transition-colors duration-150" />
                  </div>
                  <div className="flex-1">
                    <span className="text-sm font-bold text-black dark:text-white group-hover:text-black">{connector.name}</span>
                    <div className="text-xs text-gray-600 dark:text-gray-400 group-hover:text-black font-medium">
                      {connector.name === 'Injected' && 'Browser Wallet'}
                      {connector.name === 'WalletConnect' && 'Mobile & Hardware'}
                      {connector.name === 'Coinbase Wallet' && 'Coinbase Extension'}
                      {connector.name === 'MetaMask' && 'MetaMask Extension'}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="bg-white dark:bg-black text-black dark:text-white px-4 py-2 border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal font-bold hover:translate-x-[-2px] hover:translate-y-[-2px] hover:shadow-[6px_6px_0px_0px_#000000] dark:hover:shadow-[6px_6px_0px_0px_#FFFFFF] transition-all duration-150 flex items-center space-x-2"
      >
        <div className="w-6 h-6 bg-primary border-2 border-black"></div>
        <span className="font-mono">{formatAddress(address!)}</span>
        <ChevronDown className="w-4 h-4" />
      </button>
      
      {showDropdown && (
        <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-black border-2 border-black dark:border-white shadow-brutal dark:shadow-brutal z-[90]">
          <div className="p-3">
            <div className="px-3 py-2 border-b-2 border-black dark:border-white mb-3">
              <span className="text-sm font-bold text-black dark:text-white uppercase tracking-wider">
                Wallet Actions
              </span>
            </div>
            <button
              onClick={copyAddress}
              className="w-full text-left px-3 py-3 mb-2 border-2 border-black dark:border-white hover:bg-accent-cyan hover:border-black dark:hover:bg-accent-cyan dark:hover:border-black transition-all duration-150 flex items-center space-x-3 bg-white dark:bg-gray-900 group"
            >
              <Copy className="w-4 h-4 text-black dark:text-white group-hover:text-black" />
              <span className="text-sm font-bold text-black dark:text-white group-hover:text-black">Copy Address</span>
            </button>
            <button
              onClick={() => window.open(`https://basescan.org/address/${address}`, '_blank')}
              className="w-full text-left px-3 py-3 mb-2 border-2 border-black dark:border-white hover:bg-accent-yellow hover:border-black dark:hover:bg-accent-yellow dark:hover:border-black transition-all duration-150 flex items-center space-x-3 bg-white dark:bg-gray-900 group"
            >
              <ExternalLink className="w-4 h-4 text-black dark:text-white group-hover:text-black" />
              <span className="text-sm font-bold text-black dark:text-white group-hover:text-black">View on Explorer</span>
            </button>
            <div className="border-t-2 border-black dark:border-white mt-3 pt-3">
              <button
                onClick={() => disconnect()}
                className="w-full text-left px-3 py-3 border-2 border-error hover:bg-error hover:border-black dark:hover:border-black transition-all duration-150 flex items-center space-x-3 bg-white dark:bg-gray-900 group"
              >
                <LogOut className="w-4 h-4 text-error group-hover:text-white" />
                <span className="text-sm font-bold text-error group-hover:text-white">Disconnect</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}