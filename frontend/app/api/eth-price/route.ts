import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Fetch ETH price from CoinGecko API
    const response = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd',
      {
        headers: {
          'Accept': 'application/json',
        },
        // Cache for 30 seconds
        next: { revalidate: 30 }
      }
    )

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    const ethPrice = data.ethereum?.usd

    if (!ethPrice) {
      throw new Error('ETH price not found in response')
    }

    return NextResponse.json({ 
      price: Math.round(ethPrice),
      timestamp: Date.now()
    })
  } catch (error) {
    console.error('Error fetching ETH price:', error)
    
    // Return fallback price
    return NextResponse.json({ 
      price: 3240,
      timestamp: Date.now(),
      error: 'Using fallback price'
    }, { status: 200 })
  }
}