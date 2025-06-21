'use client'

import { useState, useEffect } from 'react'
import { useAccount } from 'wagmi'
import { ConnectButton } from './ConnectButton'
import { ThemeToggle } from './ThemeToggle'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Menu, X } from 'lucide-react'

export function Navbar() {
  const { isConnected } = useAccount()
  const pathname = usePathname()
  const [activeTab, setActiveTab] = useState('home')
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    // Update active tab based on current pathname
    if (pathname === '/') {
      setActiveTab('home')
    } else if (pathname === '/create') {
      setActiveTab('create')
    } else if (pathname.startsWith('/vaults')) {
      setActiveTab('vaults')
    } else if (pathname === '/positions') {
      setActiveTab('positions')
    } else if (pathname === '/dashboard') {
      setActiveTab('dashboard')
    } else if (pathname === '/docs') {
      setActiveTab('docs')
    }
  }, [pathname])




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
            <div className="hidden md:flex space-x-2">
              {[
                { id: 'home', label: 'HOME', href: '/' },
                { id: 'vaults', label: 'VAULTS', href: '/vaults' },
                { id: 'create', label: 'CREATE', href: '/create' },
                { id: 'positions', label: 'POSITIONS', href: '/positions' },
                { id: 'dashboard', label: 'DASHBOARD', href: '/dashboard' },
                { id: 'docs', label: 'API DOCS', href: '/docs' },
              ].map((tab) => (
                <Link
                  key={tab.id}
                  href={tab.href}
                  className={`px-4 py-2 text-xs border-2 transition-all duration-100 text-brutal ${
                    activeTab === tab.id
                      ? 'text-black bg-primary border-black dark:border-white shadow-brutal'
                      : 'text-black dark:text-white border-black dark:border-white hover:bg-primary hover:text-black'
                  }`}
                >
                  {tab.label}
                </Link>
              ))}
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 text-black dark:text-white border-2 border-black dark:border-white hover:bg-primary hover:text-black transition-colors"
            >
              {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

          {/* Right Side Actions */}
          <div className="flex items-center space-x-3">
            <ThemeToggle />
            <ConnectButton />
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t-2 border-black dark:border-white bg-white dark:bg-black">
            <div className="px-6 py-4 space-y-2">
              {[
                { id: 'home', label: 'HOME', href: '/' },
                { id: 'vaults', label: 'VAULTS', href: '/vaults' },
                { id: 'create', label: 'CREATE', href: '/create' },
                { id: 'positions', label: 'POSITIONS', href: '/positions' },
                { id: 'dashboard', label: 'DASHBOARD', href: '/dashboard' },
                { id: 'docs', label: 'API DOCS', href: '/docs' },
              ].map((tab) => (
                <Link
                  key={tab.id}
                  href={tab.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`block px-4 py-3 text-sm border-2 transition-all duration-100 text-brutal ${
                    activeTab === tab.id
                      ? 'text-black bg-primary border-black dark:border-white shadow-brutal'
                      : 'text-black dark:text-white border-black dark:border-white hover:bg-primary hover:text-black'
                  }`}
                >
                  {tab.label}
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}