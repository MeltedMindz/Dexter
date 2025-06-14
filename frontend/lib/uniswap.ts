// Uniswap V3 Integration Service
// Based on Revert Finance's approach but simplified for Dexter

export interface Token {
  address: string
  symbol: string
  decimals: number
  name: string
}

export interface Pool {
  id: string
  token0: Token
  token1: Token
  feeTier: number
  liquidity: string
  tick: number
  sqrtPrice: string
  volume24h: number
  tvl: number
  fees24h: number
}

export interface PoolsResponse {
  pools: Pool[]
}

// Subgraph endpoints for different networks
const SUBGRAPH_URLS = {
  ethereum: 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
  base: 'https://api.studio.thegraph.com/query/48427/uniswap-v3-base/version/latest',
  arbitrum: 'https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal',
  polygon: 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon'
}

// Popular tokens for each network
export const POPULAR_TOKENS = {
  ethereum: [
    { address: '0x0000000000000000000000000000000000000000', symbol: 'ETH', decimals: 18, name: 'Ethereum' },
    { address: '0xA0b86a33E6441986F82F5D8E58E3F7E8C2C2C6C7', symbol: 'USDC', decimals: 6, name: 'USD Coin' },
    { address: '0xdac17f958d2ee523a2206206994597c13d831ec7', symbol: 'USDT', decimals: 6, name: 'Tether USD' },
    { address: '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599', symbol: 'WBTC', decimals: 8, name: 'Wrapped Bitcoin' },
    { address: '0x6b175474e89094c44da98b954eedeac495271d0f', symbol: 'DAI', decimals: 18, name: 'Dai Stablecoin' }
  ],
  base: [
    { address: '0x4200000000000000000000000000000000000006', symbol: 'WETH', decimals: 18, name: 'Wrapped Ether' },
    { address: '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913', symbol: 'USDC', decimals: 6, name: 'USD Coin' },
    { address: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb', symbol: 'DAI', decimals: 18, name: 'Dai Stablecoin' },
    { address: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA', symbol: 'USDbC', decimals: 6, name: 'USD Base Coin' }
  ],
  arbitrum: [
    { address: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1', symbol: 'WETH', decimals: 18, name: 'Wrapped Ether' },
    { address: '0xaf88d065e77c8cC2239327C5EDb3A432268e5831', symbol: 'USDC', decimals: 6, name: 'USD Coin' },
    { address: '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9', symbol: 'USDT', decimals: 6, name: 'Tether USD' },
    { address: '0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f', symbol: 'WBTC', decimals: 8, name: 'Wrapped Bitcoin' }
  ],
  polygon: [
    { address: '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619', symbol: 'WETH', decimals: 18, name: 'Wrapped Ether' },
    { address: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174', symbol: 'USDC', decimals: 6, name: 'USD Coin' },
    { address: '0xc2132D05D31c914a87C6611C10748AEb04B58e8F', symbol: 'USDT', decimals: 6, name: 'Tether USD' },
    { address: '0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6', symbol: 'WBTC', decimals: 8, name: 'Wrapped Bitcoin' }
  ]
}

// Standard fee tiers for Uniswap V3
export const FEE_TIERS = [0.01, 0.05, 0.30, 1.00] // 0.01%, 0.05%, 0.30%, 1.00%

export class UniswapService {
  private network: keyof typeof SUBGRAPH_URLS
  
  constructor(network: keyof typeof SUBGRAPH_URLS = 'base') {
    this.network = network
  }

  setNetwork(network: keyof typeof SUBGRAPH_URLS) {
    this.network = network
  }

  async fetchPools(token0Address: string, token1Address: string): Promise<Pool[]> {
    const query = `
      query GetPools($token0: String!, $token1: String!) {
        pools(
          where: {
            or: [
              { token0: $token0, token1: $token1 },
              { token0: $token1, token1: $token0 }
            ]
          }
          orderBy: totalValueLockedUSD
          orderDirection: desc
          first: 10
        ) {
          id
          feeTier
          liquidity
          tick
          sqrtPrice
          totalValueLockedUSD
          volumeUSD
          feesUSD
          token0 {
            id
            symbol
            decimals
            name
          }
          token1 {
            id
            symbol
            decimals
            name
          }
          poolDayData(
            first: 1
            orderBy: date
            orderDirection: desc
          ) {
            volumeUSD
            feesUSD
            tvlUSD
          }
        }
      }
    `

    try {
      const response = await fetch(SUBGRAPH_URLS[this.network], {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          variables: {
            token0: token0Address.toLowerCase(),
            token1: token1Address.toLowerCase()
          }
        })
      })

      const data = await response.json()
      
      if (data.errors) {
        console.error('Subgraph errors:', data.errors)
        return this.getMockPools(token0Address, token1Address)
      }

      return data.data.pools.map((pool: any) => ({
        id: pool.id,
        token0: {
          address: pool.token0.id,
          symbol: pool.token0.symbol,
          decimals: pool.token0.decimals,
          name: pool.token0.name
        },
        token1: {
          address: pool.token1.id,
          symbol: pool.token1.symbol,
          decimals: pool.token1.decimals,
          name: pool.token1.name
        },
        feeTier: pool.feeTier / 10000, // Convert from basis points to percentage
        liquidity: pool.liquidity,
        tick: pool.tick,
        sqrtPrice: pool.sqrtPrice,
        volume24h: pool.poolDayData[0]?.volumeUSD || pool.volumeUSD || 0,
        tvl: pool.poolDayData[0]?.tvlUSD || pool.totalValueLockedUSD || 0,
        fees24h: pool.poolDayData[0]?.feesUSD || pool.feesUSD || 0
      }))

    } catch (error) {
      console.error('Error fetching pools:', error)
      return this.getMockPools(token0Address, token1Address)
    }
  }

  // Fallback mock data for development/testing
  private getMockPools(token0Address: string, token1Address: string): Pool[] {
    const tokens = POPULAR_TOKENS[this.network] || POPULAR_TOKENS.base
    const token0 = tokens.find(t => t.address.toLowerCase() === token0Address.toLowerCase()) || tokens[0]
    const token1 = tokens.find(t => t.address.toLowerCase() === token1Address.toLowerCase()) || tokens[1]

    return FEE_TIERS.map((feeTier, index) => ({
      id: `${token0Address}-${token1Address}-${feeTier}`,
      token0,
      token1,
      feeTier,
      liquidity: `${Math.random() * 1000000000}`,
      tick: Math.floor(Math.random() * 200000) - 100000,
      sqrtPrice: `${Math.random() * 1000000}`,
      volume24h: Math.random() * 100000000,
      tvl: Math.random() * 50000000,
      fees24h: Math.random() * 100000
    }))
  }

  async fetchPoolsByPair(token0Symbol: string, token1Symbol: string): Promise<Pool[]> {
    const tokens = POPULAR_TOKENS[this.network] || POPULAR_TOKENS.base
    const token0 = tokens.find(t => t.symbol === token0Symbol)
    const token1 = tokens.find(t => t.symbol === token1Symbol)

    if (!token0 || !token1) {
      throw new Error(`Token not found: ${token0Symbol} or ${token1Symbol}`)
    }

    return this.fetchPools(token0.address, token1.address)
  }

  getTokens(): Token[] {
    return POPULAR_TOKENS[this.network] || POPULAR_TOKENS.base
  }

  // Calculate APR from fees and TVL
  calculateAPR(fees24h: number, tvl: number): number {
    if (tvl === 0) return 0
    return (fees24h * 365) / tvl
  }

  // Format price for display
  formatPrice(sqrtPrice: string, token0Decimals: number, token1Decimals: number): number {
    const price = Math.pow(parseFloat(sqrtPrice) / Math.pow(2, 96), 2)
    return price * Math.pow(10, token0Decimals - token1Decimals)
  }
}

// Singleton instance
export const uniswapService = new UniswapService('base')