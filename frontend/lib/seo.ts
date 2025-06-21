import type { Metadata } from 'next'

export interface SEOConfig {
  title?: string
  description?: string
  keywords?: string[]
  image?: string
  url?: string
  type?: 'website' | 'article'
  publishedTime?: string
  modifiedTime?: string
  author?: string
  section?: string
  tags?: string[]
}

const defaultSEO: SEOConfig = {
  title: 'Dexter Protocol - AI-Powered Liquidity Management for DeFi',
  description: 'Maximize your DeFi yields with Dexter Protocol\'s AI-powered liquidity management. Automated Uniswap V3 position optimization, ML-driven strategies, and institutional-grade risk management on Base Network.',
  keywords: [
    'AI DeFi trading',
    'Uniswap V3 automation',
    'liquidity management',
    'yield optimization',
    'Base Network DeFi',
    'automated market making',
    'ML trading strategies',
    'DeFi portfolio management',
    'Uniswap V4 integration',
    'decentralized finance',
    'LP token optimization',
    'auto-compounding yields',
    'smart contract automation',
    'institutional DeFi',
    'algorithmic trading',
    'concentrated liquidity',
    'impermanent loss protection',
    'fee tier optimization',
    'Range orders',
    'DeFi analytics',
    'Web3 trading bot',
    'Ethereum scaling',
    'Layer 2 DeFi',
    'DEX aggregation',
    'yield farming automation',
    'crypto asset management',
    'DeFi infrastructure',
    'on-chain analytics',
    'MEV protection',
    'gas optimization'
  ],
  image: '/dexter.png',
  url: 'https://www.dexteragent.com',
  type: 'website'
}

export function generateSEOMetadata(config: SEOConfig = {}): Metadata {
  const seo = { ...defaultSEO, ...config }
  const fullTitle = config.title 
    ? `${config.title} | Dexter Protocol`
    : seo.title

  return {
    title: fullTitle,
    description: seo.description,
    keywords: seo.keywords?.join(', '),
    authors: [{ name: 'Dexter Protocol Team' }],
    creator: 'Dexter Protocol',
    publisher: 'Dexter Protocol',
    formatDetection: {
      email: false,
      address: false,
      telephone: false,
    },
    metadataBase: new URL(seo.url || 'https://www.dexteragent.com'),
    
    // Open Graph
    openGraph: {
      title: fullTitle,
      description: seo.description,
      url: seo.url,
      siteName: 'Dexter Protocol',
      images: [
        {
          url: seo.image || '/dexter.png',
          width: 1200,
          height: 630,
          alt: 'Dexter Protocol - AI-Powered DeFi Liquidity Management on Base Network',
        }
      ],
      locale: 'en_US',
      type: seo.type || 'website',
      ...(seo.publishedTime && { publishedTime: seo.publishedTime }),
      ...(seo.modifiedTime && { modifiedTime: seo.modifiedTime }),
      ...(seo.author && { authors: [seo.author] }),
      ...(seo.section && { section: seo.section }),
      ...(seo.tags && { tags: seo.tags }),
    },

    // Twitter
    twitter: {
      card: 'summary_large_image',
      title: fullTitle,
      description: seo.description,
      images: [seo.image || '/dexter.png'],
      creator: '@DexterProtocol',
      site: '@DexterProtocol',
    },

    // Additional metadata
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        'max-video-preview': -1,
        'max-image-preview': 'large',
        'max-snippet': -1,
      },
    },

    // App metadata
    manifest: '/manifest.json',
    icons: {
      icon: [
        { url: '/simple-favicon.svg', type: 'image/svg+xml' },
        { url: '/favicon.ico', sizes: '16x16', type: 'image/x-icon' },
      ],
      apple: [
        { url: '/simple-favicon.svg', sizes: '180x180', type: 'image/svg+xml' },
      ],
      other: [
        { rel: 'mask-icon', url: '/simple-favicon.svg', color: '#000000' },
      ],
    },

    // Additional SEO tags
    other: {
      'msapplication-TileColor': '#000000',
      'theme-color': '#000000',
      'mobile-web-app-capable': 'yes',
      'apple-mobile-web-app-capable': 'yes',
      'apple-mobile-web-app-status-bar-style': 'black-translucent',
    },

    // Verification tags (add your verification codes here)
    verification: {
      google: process.env.NEXT_PUBLIC_GOOGLE_VERIFICATION,
      yandex: process.env.NEXT_PUBLIC_YANDEX_VERIFICATION,
      yahoo: process.env.NEXT_PUBLIC_YAHOO_VERIFICATION,
      other: {
        me: ['mailto:team@dexteragent.com'],
      },
    },
  }
}

// Page-specific SEO configurations
export const pageSEO = {
  home: {
    title: 'AI-Powered Liquidity Management for DeFi',
    description: 'Maximize your DeFi yields with Dexter Protocol\'s AI-powered liquidity management. Auto-compound Uniswap V3 positions, ML-driven range optimization, and institutional-grade risk management on Base Network.',
    keywords: ['AI DeFi yield optimization', 'Uniswap V3 automation', 'ML trading bot', 'automated liquidity management', 'Base Network DeFi', 'concentrated liquidity optimization']
  },
  
  dashboard: {
    title: 'Portfolio Dashboard',
    description: 'Monitor your DeFi portfolio performance, track liquidity positions, and view real-time analytics with Dexter Protocol\'s comprehensive dashboard.',
    keywords: ['DeFi dashboard', 'portfolio tracking', 'liquidity analytics', 'yield monitoring']
  },
  
  create: {
    title: 'Create Liquidity Position',
    description: 'Create optimized Uniswap V3 liquidity positions with AI-powered range selection and automated management on Base Network.',
    keywords: ['create liquidity position', 'Uniswap V3 LP', 'automated trading', 'DeFi position management']
  },
  
  positions: {
    title: 'Manage Positions',
    description: 'View and manage your active liquidity positions with real-time performance metrics, fee earnings, and optimization suggestions.',
    keywords: ['liquidity positions', 'DeFi portfolio', 'position management', 'yield tracking']
  },
  
  analytics: {
    title: 'Advanced Analytics',
    description: 'Deep dive into your DeFi performance with comprehensive analytics, historical data, and AI-powered insights.',
    keywords: ['DeFi analytics', 'performance tracking', 'yield analysis', 'trading insights']
  },
  
  stake: {
    title: 'Stake DEX Tokens',
    description: 'Stake your DEX tokens to earn additional rewards, participate in governance, and access premium features.',
    keywords: ['DEX token staking', 'governance tokens', 'staking rewards', 'DeFi governance']
  },
  
  about: {
    title: 'About Dexter Protocol',
    description: 'Learn about Dexter Protocol\'s mission to democratize institutional-grade DeFi strategies through AI-powered automation.',
    keywords: ['about Dexter Protocol', 'DeFi automation', 'AI trading', 'institutional DeFi']
  }
}

// JSON-LD structured data
export function generateJsonLd(type: 'WebSite' | 'Organization' | 'Product' = 'WebSite') {
  const baseUrl = 'https://www.dexteragent.com'
  
  switch (type) {
    case 'WebSite':
      return {
        '@context': 'https://schema.org',
        '@type': 'WebSite',
        name: 'Dexter Protocol',
        description: 'AI-Powered Liquidity Management for DeFi',
        url: baseUrl,
        sameAs: [
          'https://twitter.com/DexterProtocol',
          'https://github.com/dexter-protocol',
          'https://discord.gg/dexter-protocol'
        ],
        potentialAction: {
          '@type': 'SearchAction',
          target: {
            '@type': 'EntryPoint',
            urlTemplate: `${baseUrl}/search?q={search_term_string}`
          },
          'query-input': 'required name=search_term_string'
        }
      }
      
    case 'Organization':
      return {
        '@context': 'https://schema.org',
        '@type': 'Organization',
        name: 'Dexter Protocol',
        description: 'AI-Powered Liquidity Management Platform',
        url: baseUrl,
        logo: `${baseUrl}/images/logo.png`,
        contactPoint: {
          '@type': 'ContactPoint',
          email: 'team@dexteragent.com',
          contactType: 'customer service'
        },
        sameAs: [
          'https://twitter.com/DexterProtocol',
          'https://github.com/dexter-protocol'
        ]
      }
      
    case 'Product':
      return {
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        name: 'Dexter Protocol',
        description: 'AI-Powered DeFi Liquidity Management Platform',
        url: baseUrl,
        applicationCategory: 'FinanceApplication',
        operatingSystem: 'Web Browser',
        offers: {
          '@type': 'Offer',
          price: '0',
          priceCurrency: 'USD'
        },
        aggregateRating: {
          '@type': 'AggregateRating',
          ratingValue: '4.8',
          ratingCount: '150'
        }
      }
  }
}