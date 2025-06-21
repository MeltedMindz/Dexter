import VaultFactory from '@/components/VaultFactory'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Create Vault",
  description: "Deploy your own automated liquidity management vault with AI optimization",
  keywords: ['create vault', 'deploy vault', 'liquidity management', 'DeFi vault creation', 'AI optimization']
})

export default function CreateVaultPage() {
  return <VaultFactory />
}