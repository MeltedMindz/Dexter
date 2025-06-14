import { NextRequest } from 'next/server'

export function GET(request: NextRequest) {
  const host = request.headers.get('host') || 'www.dexteragent.com'
  const protocol = process.env.NODE_ENV === 'production' ? 'https' : 'http'
  const baseUrl = `${protocol}://${host}`

  // Define your site's routes with their priorities and update frequencies
  const routes = [
    {
      url: '',
      priority: '1.0',
      changefreq: 'weekly',
      lastmod: new Date().toISOString()
    },
    {
      url: '/dashboard',
      priority: '0.9',
      changefreq: 'daily',
      lastmod: new Date().toISOString()
    },
    {
      url: '/create',
      priority: '0.8',
      changefreq: 'weekly',
      lastmod: new Date().toISOString()
    },
    {
      url: '/positions',
      priority: '0.8',
      changefreq: 'daily',
      lastmod: new Date().toISOString()
    },
    {
      url: '/analytics',
      priority: '0.7',
      changefreq: 'daily',
      lastmod: new Date().toISOString()
    },
    {
      url: '/stake',
      priority: '0.6',
      changefreq: 'weekly',
      lastmod: new Date().toISOString()
    },
    {
      url: '/about',
      priority: '0.5',
      changefreq: 'monthly',
      lastmod: new Date().toISOString()
    }
  ]

  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9
        http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
${routes
  .map(
    route => `  <url>
    <loc>${baseUrl}${route.url}</loc>
    <lastmod>${route.lastmod}</lastmod>
    <changefreq>${route.changefreq}</changefreq>
    <priority>${route.priority}</priority>
  </url>`
  )
  .join('\n')}
</urlset>`

  return new Response(sitemap, {
    headers: {
      'Content-Type': 'application/xml',
      'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
    },
  })
}