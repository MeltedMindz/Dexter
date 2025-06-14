import { V4PositionManager } from '@/components/V4PositionManager'
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata(pageSEO.positions)

export default function PositionsPage() {
  return (
    <V4PositionManager />
  )
}