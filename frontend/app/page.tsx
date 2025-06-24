import { AIHomepage } from '@/components/AIHomepage'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: 'Dexter Protocol | AI-Powered DeFi Liquidity Management',
  description: 'Connect your wallet for instant AI portfolio analysis. Get personalized liquidity strategies, risk assessments, and vault recommendations powered by advanced machine learning.',
  keywords: ['AI DeFi', 'liquidity management', 'portfolio analysis', 'vault strategies', 'automated trading', 'Base network']
})

export default function Home() {
  return (
    <AIHomepage />
  )
}