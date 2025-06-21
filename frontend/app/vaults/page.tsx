import VaultList from '@/components/VaultList'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Vault Explorer | Dexter Protocol",
  description: "Discover and invest in automated liquidity management vaults powered by AI optimization",
  path: "/vaults"
})

export default function VaultsPage() {
  return <VaultList />
}