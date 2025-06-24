import { NextRequest, NextResponse } from 'next/server'
import { chatRateLimiter } from '@/lib/rate-limiter'

// Admin endpoint for monitoring chat usage and abuse patterns
// Secure this endpoint in production!

export async function GET(request: NextRequest) {
  // Basic authentication check (replace with proper auth in production)
  const authHeader = request.headers.get('authorization')
  const adminPassword = process.env.CHAT_ADMIN_PASSWORD || 'dexter-admin-2024'
  
  if (!authHeader || authHeader !== `Bearer ${adminPassword}`) {
    return NextResponse.json(
      { error: 'Unauthorized. Admin access required.' },
      { status: 401 }
    )
  }

  try {
    const url = new URL(request.url)
    const action = url.searchParams.get('action')
    const address = url.searchParams.get('address')

    switch (action) {
      case 'block':
        if (!address) {
          return NextResponse.json(
            { error: 'Address parameter required for block action' },
            { status: 400 }
          )
        }
        chatRateLimiter.blockAddress(address)
        return NextResponse.json({
          success: true,
          message: `Address ${address} has been blocked`
        })

      case 'unblock':
        if (!address) {
          return NextResponse.json(
            { error: 'Address parameter required for unblock action' },
            { status: 400 }
          )
        }
        chatRateLimiter.unblockAddress(address)
        return NextResponse.json({
          success: true,
          message: `Address ${address} has been unblocked`
        })

      case 'stats':
        if (!address) {
          return NextResponse.json(
            { error: 'Address parameter required for stats action' },
            { status: 400 }
          )
        }
        const stats = chatRateLimiter.getUsageStats(address)
        return NextResponse.json({
          address,
          stats
        })

      case 'cleanup':
        chatRateLimiter.cleanup()
        return NextResponse.json({
          success: true,
          message: 'Cleanup completed'
        })

      default:
        return NextResponse.json({
          status: 'Dexter Chat Admin Panel',
          endpoints: [
            'GET ?action=stats&address=0x... - Get usage stats for address',
            'GET ?action=block&address=0x... - Block an address',
            'GET ?action=unblock&address=0x... - Unblock an address',
            'GET ?action=cleanup - Clean up expired entries'
          ],
          security: 'Requires Authorization: Bearer {CHAT_ADMIN_PASSWORD}',
          environment: process.env.NODE_ENV || 'development'
        })
    }
  } catch (error) {
    console.error('Chat admin error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// POST endpoint for batch operations
export async function POST(request: NextRequest) {
  const authHeader = request.headers.get('authorization')
  const adminPassword = process.env.CHAT_ADMIN_PASSWORD || 'dexter-admin-2024'
  
  if (!authHeader || authHeader !== `Bearer ${adminPassword}`) {
    return NextResponse.json(
      { error: 'Unauthorized. Admin access required.' },
      { status: 401 }
    )
  }

  try {
    const { action, addresses } = await request.json()

    if (!Array.isArray(addresses)) {
      return NextResponse.json(
        { error: 'addresses must be an array' },
        { status: 400 }
      )
    }

    switch (action) {
      case 'block_batch':
        addresses.forEach(address => chatRateLimiter.blockAddress(address))
        return NextResponse.json({
          success: true,
          message: `Blocked ${addresses.length} addresses`,
          addresses
        })

      case 'unblock_batch':
        addresses.forEach(address => chatRateLimiter.unblockAddress(address))
        return NextResponse.json({
          success: true,
          message: `Unblocked ${addresses.length} addresses`,
          addresses
        })

      default:
        return NextResponse.json(
          { error: 'Invalid action. Use: block_batch, unblock_batch' },
          { status: 400 }
        )
    }
  } catch (error) {
    console.error('Chat admin POST error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}