import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Dexter Protocol - AI-Powered Liquidity Management',
  description: 'Auto-compound your Uniswap V3 positions with performance-based fees',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gradient-to-br from-slate-50 to-slate-100 dark:from-dark-900 dark:to-dark-800 transition-colors`}>
        <Providers>
          <div className="flex flex-col h-full">
            {/* Fixed Header */}
            <div className="fixed top-0 left-0 right-0 z-50">
              <Navbar />
            </div>
            
            {/* Main Content with padding for fixed header/footer */}
            <main className="flex-1 pt-16 pb-12 overflow-y-auto">
              {children}
            </main>
            
            {/* Fixed Footer */}
            <Footer />
          </div>
        </Providers>
      </body>
    </html>
  )
}