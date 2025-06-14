# Dexter Protocol Frontend Deployment Guide

## 🚀 Vercel Deployment with Alchemy Integration

This guide covers deploying the Dexter Protocol frontend to Vercel with your Alchemy API integration.

### Prerequisites

- Vercel account
- GitHub repository connected to Vercel
- Alchemy API key: `ory0F2cLFNIXsovAmrtJj`

### Environment Variables

Add these environment variables in your Vercel dashboard:

#### Required Variables
```bash
NEXT_PUBLIC_ALCHEMY_API_KEY=ory0F2cLFNIXsovAmrtJj
NEXT_PUBLIC_BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/ory0F2cLFNIXsovAmrtJj
NEXT_PUBLIC_MAINNET_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/ory0F2cLFNIXsovAmrtJj
NEXT_PUBLIC_WC_PROJECT_ID=130fac92acf3681422903f821a58922a
```

#### App Configuration
```bash
NEXT_PUBLIC_APP_NAME=Dexter Protocol
NEXT_PUBLIC_APP_DESCRIPTION=AI-Powered Liquidity Management
NEXT_PUBLIC_APP_URL=https://www.dexteragent.com
```

#### Optional SEO Variables
```bash
NEXT_PUBLIC_GOOGLE_VERIFICATION=your-google-verification-code
NEXT_PUBLIC_YANDEX_VERIFICATION=your-yandex-verification-code
NEXT_PUBLIC_YAHOO_VERIFICATION=your-yahoo-verification-code
```

### Deployment Steps

1. **Connect Repository to Vercel**
   ```bash
   # Install Vercel CLI (optional)
   npm i -g vercel
   
   # Deploy from CLI
   vercel --prod
   ```

2. **Configure Environment Variables**
   - Go to Vercel Dashboard → Project → Settings → Environment Variables
   - Add all the variables listed above
   - Make sure to select "Production", "Preview", and "Development" for each variable

3. **Domain Configuration**
   - Add your custom domain in Vercel Dashboard → Project → Settings → Domains
   - Update `NEXT_PUBLIC_APP_URL` to match your domain

4. **Deploy**
   - Push changes to your main branch
   - Vercel will automatically deploy
   - Or manually trigger deployment in Vercel Dashboard

### SEO Features Included

✅ **Comprehensive Meta Tags**
- Open Graph tags for social sharing
- Twitter Card optimization
- JSON-LD structured data
- Mobile-friendly viewport settings

✅ **Dynamic Sitemap**
- Auto-generated XML sitemap at `/sitemap.xml`
- Updates automatically with new pages

✅ **Robots.txt**
- SEO-friendly robots.txt at `/robots.txt`
- Allows search engines while blocking sensitive paths

✅ **Performance Optimization**
- Security headers configured
- Caching strategies implemented
- Image optimization ready

✅ **PWA Ready**
- Web app manifest included
- Offline capability foundation
- Mobile app-like experience

### Alchemy Features Integrated

🔗 **Enhanced RPC Endpoints**
- Base Network via Alchemy
- Mainnet support via Alchemy
- Improved reliability and speed

📊 **Enhanced Data APIs**
- Token metadata with logos
- Enhanced balance fetching
- Transaction history access
- NFT support ready

⚡ **Performance Features**
- WebSocket connections for real-time updates
- Subgraph access enabled
- Enhanced gas estimation

### Security Features

🔒 **Headers Configured**
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection enabled
- Referrer-Policy configured
- Permissions-Policy set

### Social Media Integration

📱 **Quick Links**
- `/twitter` → Redirects to Twitter profile
- `/github` → Redirects to GitHub repository
- `/discord` → Redirects to Discord server

### Monitoring & Analytics

📈 **Built-in Analytics**
- Vercel Analytics integrated
- Speed Insights enabled
- Performance monitoring ready

### Post-Deployment Checklist

- [ ] Verify all environment variables are set
- [ ] Test Alchemy API connectivity
- [ ] Check `/sitemap.xml` loads correctly
- [ ] Verify `/robots.txt` is accessible
- [ ] Test social media sharing (Open Graph)
- [ ] Confirm mobile responsiveness
- [ ] Verify all pages have proper SEO meta tags
- [ ] Test wallet connection with Base Network
- [ ] Confirm real-time data loading

### Troubleshooting

**Common Issues:**

1. **RPC Connection Failed**
   - Verify `NEXT_PUBLIC_ALCHEMY_API_KEY` is set correctly
   - Check Alchemy dashboard for API usage

2. **SEO Meta Tags Not Showing**
   - Ensure pages are server-side rendered
   - Check meta tags in browser dev tools

3. **Environment Variables Not Loading**
   - Verify variables are prefixed with `NEXT_PUBLIC_`
   - Redeploy after adding new variables

### Support

For deployment issues:
- Check Vercel deployment logs
- Verify all environment variables
- Test locally with production build: `npm run build && npm start`

For Alchemy integration issues:
- Check Alchemy dashboard for errors
- Verify API key has required permissions
- Test endpoints directly in browser dev tools