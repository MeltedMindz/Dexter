import VaultDashboard from '@/components/VaultDashboard'
import { generateSEOMetadata } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata({
  title: "Vault Dashboard",
  description: "Manage your automated liquidity vault with real-time analytics and AI optimization",
  keywords: ['vault dashboard', 'vault management', 'liquidity analytics', 'DeFi performance tracking', 'AI optimization']
})

type VaultPageProps = {
  params: Promise<{ address: string }>
}

export default async function VaultPage(props: VaultPageProps) {
  const params = await props.params
  return <VaultDashboard vaultAddress={params.address} />
}