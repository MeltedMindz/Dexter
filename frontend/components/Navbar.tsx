'use client'

import { useState } from 'react'
import { useAccount } from 'wagmi'
import { ConnectButton } from './ConnectButton'
import { ThemeToggle } from './ThemeToggle'
import Link from 'next/link'

export function Navbar() {
  const { isConnected } = useAccount()
  const [activeTab, setActiveTab] = useState('portfolio')

  return (
    <nav className="bg-white dark:bg-black border-b-2 border-black dark:border-white transition-colors">
      <div className="w-full px-6 lg:px-12">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-8">
            <Link href="/" className="flex items-center space-x-2 group">
              <div className="w-8 h-8 bg-primary border-2 border-black dark:border-white flex items-center justify-center">
                <span className="text-black font-bold text-lg font-mono">D</span>
              </div>
              <span className="text-xl font-bold text-black dark:text-white text-brutal">DEXTER</span>
            </Link>
            
            {/* Navigation Links */}
            {isConnected && (
              <div className="hidden md:flex space-x-2">
                {[
                  { id: 'home', label: 'HOME', href: '/' },
                  { id: 'create', label: 'CREATE', href: '/create' },
                  { id: 'dashboard', label: 'DASHBOARD', href: '/dashboard' },
                  { id: 'positions', label: 'POSITIONS', href: '/about' },
                ].map((tab) => (
                  <Link
                    key={tab.id}
                    href={tab.href}
                    className={`px-4 py-2 text-xs border-2 transition-all duration-100 text-brutal ${
                      activeTab === tab.id
                        ? 'text-black bg-primary border-black dark:border-white shadow-brutal'
                        : 'text-black dark:text-white border-black dark:border-white hover:bg-primary hover:text-black'
                    }`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </Link>
                ))}
              </div>
            )}
          </div>

          {/* Right Side Actions */}
          <div className="flex items-center space-x-3">
            <ThemeToggle />
            <ConnectButton />
          </div>
        </div>
      </div>
    </nav>
  )
}