import { FlywheelExplainer } from '@/components/FlywheelExplainer'
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata(pageSEO.home)

export default function Home() {
  return (
    <FlywheelExplainer />
  )
}