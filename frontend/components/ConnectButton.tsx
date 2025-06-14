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
          <div className="absolute right-0 mt-2 w-56 bg-white border border-slate-200 rounded-lg shadow-lg z-50">
            <div className="p-2">
              {connectors.map((connector) => (
                <button
                  key={connector.uid}
                  onClick={() => {
                    connect({ connector })
                    setShowConnectors(false)
                  }}
                  className="w-full text-left px-3 py-2 rounded-md hover:bg-slate-100 transition-colors flex items-center space-x-3"
                >
                  <div className="w-6 h-6 bg-slate-100 rounded-full flex items-center justify-center">
                    <Wallet className="w-3 h-3" />
                  </div>
                  <span className="text-sm font-medium">{connector.name}</span>
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
        className="bg-slate-100 text-slate-900 px-4 py-2 rounded-lg font-medium hover:bg-slate-200 transition-colors flex items-center space-x-2"
      >
        <div className="w-6 h-6 bg-primary rounded-full"></div>
        <span className="mono-numbers">{formatAddress(address!)}</span>
        <ChevronDown className="w-4 h-4" />
      </button>
      
      {showDropdown && (
        <div className="absolute right-0 mt-2 w-56 bg-white border border-slate-200 rounded-lg shadow-lg z-50">
          <div className="p-2">
            <button
              onClick={copyAddress}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-slate-100 transition-colors flex items-center space-x-3"
            >
              <Copy className="w-4 h-4" />
              <span className="text-sm">Copy Address</span>
            </button>
            <button
              onClick={() => window.open(`https://basescan.org/address/${address}`, '_blank')}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-slate-100 transition-colors flex items-center space-x-3"
            >
              <ExternalLink className="w-4 h-4" />
              <span className="text-sm">View on Explorer</span>
            </button>
            <hr className="my-2" />
            <button
              onClick={() => disconnect()}
              className="w-full text-left px-3 py-2 rounded-md hover:bg-red-50 text-red-600 transition-colors flex items-center space-x-3"
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