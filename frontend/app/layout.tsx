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
        {/* Structured Data */}
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
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(generateJsonLd('Product'))
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