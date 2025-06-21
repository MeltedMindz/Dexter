import VaultDashboard from '@/components/VaultDashboard'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Vault Dashboard",
  description: "Manage your automated liquidity vault with real-time analytics and AI optimization",
  keywords: ['vault dashboard', 'vault management', 'liquidity analytics', 'DeFi performance tracking', 'AI optimization']
})

interface VaultPageProps {
  params: Promise<{
    address: string
  }>
}

export default async function VaultPage({ params }: VaultPageProps) {
  const { address } = await params
  return <VaultDashboard vaultAddress={address} />
}