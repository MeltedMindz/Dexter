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
          className="bg-primary text-white px-4 py-2 rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center space-x-2"
        >
          <Wallet className="w-4 h-4" />
          <span>Connect Wallet</span>
        </button>
        
        {showConnectors && (
          <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-dark-700 border border-slate-200 dark:border-white/10 rounded-lg shadow-lg dark:shadow-xl z-50">
            <div className="p-2">
              <div className="px-3 py-2 border-b border-slate-200 dark:border-white/10 mb-2">
                <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
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
                  className="w-full text-left px-3 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-dark-600 transition-colors flex items-center space-x-3 text-slate-900 dark:text-white"
                >
                  <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary-600 rounded-lg flex items-center justify-center shadow-sm">
                    <Wallet className="w-4 h-4 text-white" />
                  </div>
                  <div className="flex-1">
                    <span className="text-sm font-medium">{connector.name}</span>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
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
        className="bg-slate-100 dark:bg-dark-600 text-slate-900 dark:text-white px-4 py-2 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-dark-500 transition-colors flex items-center space-x-2"
      >
        <div className="w-6 h-6 bg-gradient-to-br from-primary to-primary-600 rounded-full"></div>
        <span className="mono-numbers">{formatAddress(address!)}</span>
        <ChevronDown className="w-4 h-4" />
      </button>
      
      {showDropdown && (
        <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-dark-700 border border-slate-200 dark:border-white/10 rounded-lg shadow-lg dark:shadow-xl z-50">
          <div className="p-2">
            <div className="px-3 py-2 border-b border-slate-200 dark:border-white/10 mb-2">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">
                Wallet Actions
              </span>
            </div>
            <button
              onClick={copyAddress}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-dark-600 transition-colors flex items-center space-x-3 text-slate-900 dark:text-white"
            >
              <Copy className="w-4 h-4" />
              <span className="text-sm">Copy Address</span>
            </button>
            <button
              onClick={() => window.open(`https://basescan.org/address/${address}`, '_blank')}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-dark-600 transition-colors flex items-center space-x-3 text-slate-900 dark:text-white"
            >
              <ExternalLink className="w-4 h-4" />
              <span className="text-sm">View on Explorer</span>
            </button>
            <hr className="my-2 border-slate-200 dark:border-white/10" />
            <button
              onClick={() => disconnect()}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20 text-red-600 dark:text-red-400 transition-colors flex items-center space-x-3"
            >
              <LogOut className="w-4 h-4" />
              <span className="text-sm">Disconnect</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}