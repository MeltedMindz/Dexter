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
    <nav className="bg-white/80 dark:bg-dark-900/80 backdrop-blur-md border-b border-slate-200 dark:border-white/10 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-8">
            <Link href="/" className="flex items-center space-x-2 group">
              <div className="relative w-8 h-8 bg-gradient-to-br from-primary to-primary-600 rounded-lg flex items-center justify-center transition-transform group-hover:scale-105">
                <span className="text-white font-bold text-lg">D</span>
                <div className="absolute inset-0 rounded-lg bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <span className="text-xl font-bold text-slate-900 dark:text-white transition-colors">Dexter</span>
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
                    className={`px-4 py-2 text-xs font-bold tracking-wider rounded-lg transition-all duration-200 ${
                      activeTab === tab.id
                        ? 'text-white bg-primary shadow-lg shadow-primary/25'
                        : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-white/5'
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