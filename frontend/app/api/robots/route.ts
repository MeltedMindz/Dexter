import { NextRequest } from 'next/server'

export function GET(request: NextRequest) {
  const host = request.headers.get('host') || 'www.dexteragent.com'
  const protocol = process.env.NODE_ENV === 'production' ? 'https' : 'http'
  const baseUrl = `${protocol}://${host}`

  const robotsTxt = `User-agent: *
Allow: /

# Block admin and API routes
Disallow: /api/
Disallow: /admin/
Disallow: /_next/
Disallow: /private/

# Allow specific API routes for SEO
Allow: /api/sitemap
Allow: /api/robots

# Sitemap location
Sitemap: ${baseUrl}/sitemap.xml

# Crawl-delay for responsible crawling
Crawl-delay: 1

# Specific instructions for different bots
User-agent: Googlebot
Allow: /

User-agent: Bingbot
Allow: /

User-agent: facebookexternalhit
Allow: /

User-agent: Twitterbot
Allow: /

User-agent: LinkedInBot
Allow: /

# Block aggressive crawlers
User-agent: AhrefsBot
Disallow: /

User-agent: MJ12bot
Disallow: /

User-agent: DotBot
Disallow: /`

  return new Response(robotsTxt, {
    headers: {
      'Content-Type': 'text/plain',
      'Cache-Control': 'public, max-age=86400', // Cache for 24 hours
    },
  })
}