import VaultList from '@/components/VaultList'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Vault Explorer",
  description: "Discover and invest in automated liquidity management vaults powered by AI optimization",
  keywords: ['DeFi vaults', 'automated liquidity management', 'vault explorer', 'yield farming', 'AI optimization']
})

export default function VaultsPage() {
  return <VaultList />
}