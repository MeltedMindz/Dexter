'use client'

import { useState, useEffect } from 'react'
import { useAccount } from 'wagmi'
import { ConnectButton } from './ConnectButton'
import { ThemeToggle } from './ThemeToggle'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { Menu, X } from 'lucide-react'

export function Navbar() {
  const { isConnected } = useAccount()
  const pathname = usePathname()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState('home')
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [debugMode, setDebugMode] = useState(false)

  useEffect(() => {
    // Update active tab based on current pathname
    if (pathname === '/') {
      setActiveTab('home')
    } else if (pathname === '/create') {
      setActiveTab('create')
    } else if (pathname === '/positions') {
      setActiveTab('positions')
    } else if (pathname === '/dashboard') {
      setActiveTab('dashboard')
    }
  }, [pathname])

  // Manual navigation handler as fallback
  const handleNavigation = (href: string, tabId: string) => {
    console.log(`🚀 MANUAL NAV: Navigating to ${href} (${tabId})`)
    router.push(href)
  }

  // Debug function to log navbar clicks
  const handleNavClick = (href: string, label: string) => {
    console.log(`🔍 DEBUG: Navbar link clicked - ${label} (${href})`)
    console.log(`🔍 DEBUG: Current pathname: ${pathname}`)
    console.log(`🔍 DEBUG: Active tab: ${activeTab}`)
  }

  // Enable debug mode on create page
  useEffect(() => {
    if (pathname === '/create') {
      setDebugMode(true)
      console.log('🔍 DEBUG: Debug mode enabled for navbar on /create page')
    } else {
      setDebugMode(false)
    }
  }, [pathname])

  return (
    <nav className="bg-white dark:bg-black border-b-2 border-black dark:border-white transition-colors" 
         style={{ pointerEvents: 'auto', position: 'relative', zIndex: 101 }}>
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
                { id: 'create', label: 'CREATE V4', href: '/create' },
                { id: 'positions', label: 'POSITIONS', href: '/positions' },
                { id: 'dashboard', label: 'DASHBOARD', href: '/dashboard' },
              ].map((tab) => (
                <Link
                  key={tab.id}
                  href={tab.href}
                  onClick={(e) => {
                    handleNavClick(tab.href, tab.label)
                    // Force navigation on create page using router.push
                    if (pathname === '/create') {
                      e.preventDefault()
                      handleNavigation(tab.href, tab.id)
                    }
                  }}
                  className={`px-4 py-2 text-xs border-2 transition-all duration-100 text-brutal relative ${
                    activeTab === tab.id
                      ? 'text-black bg-primary border-black dark:border-white shadow-brutal'
                      : 'text-black dark:text-white border-black dark:border-white hover:bg-primary hover:text-black'
                  } ${debugMode ? 'debug-navbar-clickable' : ''}`}
                  style={{ 
                    pointerEvents: 'auto',
                    position: 'relative',
                    zIndex: 101
                  }}
                >
                  {tab.label}
                  {/* Debug indicator */}
                  {pathname === '/create' && (
                    <div className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" title="Debug: Create page active" />
                  )}
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
                { id: 'create', label: 'CREATE V4', href: '/create' },
                { id: 'positions', label: 'POSITIONS', href: '/positions' },
                { id: 'dashboard', label: 'DASHBOARD', href: '/dashboard' },
              ].map((tab) => (
                <Link
                  key={tab.id}
                  href={tab.href}
                  onClick={(e) => {
                    handleNavClick(tab.href, tab.label)
                    setIsMobileMenuOpen(false)
                    // Force navigation on create page using router.push
                    if (pathname === '/create') {
                      e.preventDefault()
                      handleNavigation(tab.href, tab.id)
                    }
                  }}
                  className={`block px-4 py-3 text-sm border-2 transition-all duration-100 text-brutal ${
                    activeTab === tab.id
                      ? 'text-black bg-primary border-black dark:border-white shadow-brutal'
                      : 'text-black dark:text-white border-black dark:border-white hover:bg-primary hover:text-black'
                  }`}
                  style={{ 
                    pointerEvents: 'auto',
                    position: 'relative',
                    zIndex: 101
                  }}
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