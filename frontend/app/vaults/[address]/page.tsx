import VaultDashboard from '@/components/VaultDashboard'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Vault Dashboard | Dexter Protocol",
  description: "Manage your automated liquidity vault with real-time analytics and AI optimization",
  path: "/vaults/[address]"
})

interface VaultPageProps {
  params: {
    address: string
  }
}

export default function VaultPage({ params }: VaultPageProps) {
  return <VaultDashboard vaultAddress={params.address} />
}