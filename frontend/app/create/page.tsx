import { V4PositionCreator } from '@/components/V4PositionCreator'
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata(pageSEO.create)

export default function CreatePositionPage() {
  return (
    <V4PositionCreator />
  )
}