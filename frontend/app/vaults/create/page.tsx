import VaultFactory from '@/components/VaultFactory'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Create Vault | Dexter Protocol",
  description: "Deploy your own automated liquidity management vault with AI optimization",
  path: "/vaults/create"
})

export default function CreateVaultPage() {
  return <VaultFactory />
}