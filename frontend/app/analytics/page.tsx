import { PoolAnalytics } from '@/components/PoolAnalytics'
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata(pageSEO.analytics)

export default function AnalyticsPage() {
  return (
    <PoolAnalytics />
  )
}