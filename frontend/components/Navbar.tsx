'use client'

import { useState } from 'react'
import { useAccount } from 'wagmi'
import { ConnectButton } from './ConnectButton'
import Link from 'next/link'

export function Navbar() {
  const { isConnected } = useAccount()
  const [activeTab, setActiveTab] = useState('portfolio')

  return (
    <nav className="bg-white shadow-sm border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-8">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">D</span>
              </div>
              <span className="text-xl font-bold text-slate-900">Dexter</span>
            </Link>
            
            {/* Navigation Links */}
            {isConnected && (
              <div className="hidden md:flex space-x-6">
                {[
                  { id: 'portfolio', label: 'Portfolio', href: '/' },
                  { id: 'positions', label: 'Positions', href: '/' },
                  { id: 'stake', label: 'Stake', href: '/stake' },
                  { id: 'about', label: 'About', href: '/about' },
                ].map((tab) => (
                  <Link
                    key={tab.id}
                    href={tab.href}
                    className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                      activeTab === tab.id
                        ? 'text-primary bg-primary/10'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                    }`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </Link>
                ))}
              </div>
            )}
          </div>

          {/* Connect Wallet Button */}
          <ConnectButton />
        </div>
      </div>
    </nav>
  )
}