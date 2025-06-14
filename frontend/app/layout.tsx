import type { Metadata } from 'next'
import { Space_Grotesk, IBM_Plex_Mono } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'

const spaceGrotesk = Space_Grotesk({ 
  subsets: ['latin'],
  variable: '--font-space-grotesk'
})

const ibmPlexMono = IBM_Plex_Mono({ 
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-ibm-plex-mono'
})

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
      <body className={`${spaceGrotesk.variable} ${ibmPlexMono.variable} font-sans h-full bg-white dark:bg-black transition-colors`}>
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