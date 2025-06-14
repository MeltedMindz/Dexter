import { AboutPage } from '@/components/AboutPage'
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
import type { Metadata } from 'next'

export const metadata: Metadata = generateSEOMetadata(pageSEO.about)

export default function About() {
  return (
    <AboutPage />
  )
}