import type { Metadata } from 'next'
import { Space_Grotesk, IBM_Plex_Mono } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { Analytics } from '@vercel/analytics/next'
import { SpeedInsights } from '@vercel/speed-insights/next'
import { generateSEOMetadata, generateJsonLd } from '@/lib/seo'

const spaceGrotesk = Space_Grotesk({ 
  subsets: ['latin'],
  variable: '--font-space-grotesk'
})

const ibmPlexMono = IBM_Plex_Mono({ 
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-ibm-plex-mono'
})

export const metadata: Metadata = generateSEOMetadata()

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <head>
        {/* Additional Social Media Meta Tags */}
        <meta property="og:title" content="Dexter Protocol - AI-Powered Liquidity Management" />
        <meta property="og:description" content="AI-POWERED LIQUIDITY MANAGEMENT - Maximize your DeFi yields with automated Uniswap V3 position management on Base Network." />
        <meta property="og:image" content="https://via.placeholder.com/1200x630/000000/00FF88?text=DEXTER+PROTOCOL+%7C+AI-POWERED+LIQUIDITY+MANAGEMENT" />
        <meta property="og:url" content="https://www.dexteragent.com" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Dexter Protocol" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Dexter Protocol - AI-Powered Liquidity Management" />
        <meta name="twitter:description" content="AI-POWERED LIQUIDITY MANAGEMENT - Maximize your DeFi yields with automated Uniswap V3 position management on Base Network." />
        <meta name="twitter:image" content="https://via.placeholder.com/1200x630/000000/00FF88?text=DEXTER+PROTOCOL+%7C+AI-POWERED+LIQUIDITY+MANAGEMENT" />
        
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(generateJsonLd('WebSite'))
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(generateJsonLd('Organization'))
          }}
        />
      </head>
      <body className={`${spaceGrotesk.variable} ${ibmPlexMono.variable} font-sans h-full bg-white dark:bg-black transition-colors`}>
        <Providers>
          <div className="flex flex-col h-full">
            {/* Fixed Header */}
            <div className="fixed top-0 left-0 right-0 z-[100]" style={{ pointerEvents: 'auto' }}>
              <Navbar />
            </div>
            
            {/* Main Content with padding for fixed header/footer */}
            <main className="flex-1 pt-20 pb-16 overflow-y-auto">
              {children}
            </main>
            
            {/* Fixed Footer */}
            <Footer />
          </div>
          
          {/* Vercel Analytics & Speed Insights */}
          <Analytics />
          <SpeedInsights />
        </Providers>
      </body>
    </html>
  )
}