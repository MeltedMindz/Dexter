import { Dashboard } from '@/components/Dashboard'
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata(pageSEO.dashboard)

export default function DashboardPage() {
  return (
    <Dashboard />
  )
}