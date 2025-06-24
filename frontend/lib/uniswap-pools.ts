// Base Network Uniswap V3 Pool Data
// Real pool addresses from Base mainnet

export interface PoolInfo {
  address: string
  token0: string
  token1: string
  token0Symbol: string
  token1Symbol: string
  fee: number
  feeTier: string
  network: 'base' | 'mainnet'
}

// Popular Base Network Pools
export const BASE_POOLS: PoolInfo[] = [
  // ETH/USDC pools
  {
    address: '0xd0b53D9277642d899DF5C87A3966A349A798F224',
    token0: '0x4200000000000000000000000000000000000006', // WETH
    token1: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', // USDC
    token0Symbol: 'ETH',
    token1Symbol: 'USDC',
    fee: 500, // 0.05%
    feeTier: '0.05%',
    network: 'base'
  },
  {
    address: '0x1fA76Ae76a3b9976F4baf4e09817F683385e9E07',
    token0: '0x4200000000000000000000000000000000000006', // WETH
    token1: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', // USDC
    token0Symbol: 'ETH',
    token1Symbol: 'USDC',
    fee: 3000, // 0.3%
    feeTier: '0.3%',
    network: 'base'
  },
  // ETH/USDbC pool
  {
    address: '0x4C36388bE6F416A29C8d8Eee81C771cE6bE14B18',
    token0: '0x4200000000000000000000000000000000000006', // WETH
    token1: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA', // USDbC
    token0Symbol: 'ETH',
    token1Symbol: 'USDbC',
    fee: 500, // 0.05%
    feeTier: '0.05%',
    network: 'base'
  },
  // USDC/USDbC pool
  {
    address: '0x06959273E9A65433De71F5A452D529544E07dDD0',
    token0: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913', // USDC
    token1: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA', // USDbC
    token0Symbol: 'USDC',
    token1Symbol: 'USDbC',
    fee: 100, // 0.01%
    feeTier: '0.01%',
    network: 'base'
  },
  // cbETH/ETH pool
  {
    address: '0x7c9C6F5BEd9Cfe5B9070C7D3322CF39eAD3A4C9d',
    token0: '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22', // cbETH
    token1: '0x4200000000000000000000000000000000000006', // WETH
    token0Symbol: 'cbETH',
    token1Symbol: 'ETH',
    fee: 500, // 0.05%
    feeTier: '0.05%',
    network: 'base'
  }
]

// Token addresses on Base
export const BASE_TOKENS = {
  WETH: '0x4200000000000000000000000000000000000006',
  USDC: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
  USDbC: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA', // Bridged USDC
  DAI: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
  cbETH: '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
  USDT: '0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2'
}

// Get recommended pools based on tokens in wallet
export function getRecommendedPools(tokens: string[]): PoolInfo[] {
  const recommendations: PoolInfo[] = []
  
  // Normalize token symbols
  const hasETH = tokens.some(t => t === 'ETH' || t === 'WETH')
  const hasUSDC = tokens.some(t => t === 'USDC')
  const hasUSDbC = tokens.some(t => t === 'USDbC')
  const hasDAI = tokens.some(t => t === 'DAI')
  
  // Recommend ETH/Stable pools if user has both
  if (hasETH && (hasUSDC || hasUSDbC)) {
    recommendations.push(
      ...BASE_POOLS.filter(pool => 
        pool.token0Symbol === 'ETH' && 
        (pool.token1Symbol === 'USDC' || pool.token1Symbol === 'USDbC')
      )
    )
  }
  
  // Recommend stable/stable pools
  if ((hasUSDC && hasUSDbC) || (hasUSDC && hasDAI)) {
    recommendations.push(
      ...BASE_POOLS.filter(pool => 
        (pool.token0Symbol === 'USDC' && pool.token1Symbol === 'USDbC') ||
        (pool.token0Symbol === 'USDC' && pool.token1Symbol === 'DAI')
      )
    )
  }
  
  // Sort by fee tier (lower fees first for stable pairs)
  return recommendations.sort((a, b) => a.fee - b.fee)
}

// Calculate expected returns based on pool metrics
export function estimatePoolAPR(pool: PoolInfo): number {
  // This is a simplified estimation
  // In production, would fetch real volume/TVL data
  
  const baseAPRByFee = {
    100: 5,    // 0.01% - stable pairs
    500: 15,   // 0.05% - major pairs
    3000: 12,  // 0.3% - standard
    10000: 8   // 1% - exotic pairs
  }
  
  let baseAPR = baseAPRByFee[pool.fee] || 10
  
  // Adjust for token pair volatility
  if (pool.token0Symbol === 'ETH' || pool.token1Symbol === 'ETH') {
    baseAPR *= 1.2 // Higher volatility = more fees
  }
  
  // Stable pairs have lower but more consistent returns
  if (
    ['USDC', 'USDT', 'DAI', 'USDbC'].includes(pool.token0Symbol) &&
    ['USDC', 'USDT', 'DAI', 'USDbC'].includes(pool.token1Symbol)
  ) {
    baseAPR *= 0.8
  }
  
  return parseFloat(baseAPR.toFixed(1))
}

// Get impermanent loss risk assessment
export function assessImpermanentLossRisk(
  token0Symbol: string, 
  token1Symbol: string
): 'low' | 'medium' | 'high' {
  const stableTokens = ['USDC', 'USDT', 'DAI', 'USDbC']
  
  // Both tokens are stables = low risk
  if (stableTokens.includes(token0Symbol) && stableTokens.includes(token1Symbol)) {
    return 'low'
  }
  
  // One stable, one volatile = medium risk
  if (stableTokens.includes(token0Symbol) || stableTokens.includes(token1Symbol)) {
    return 'medium'
  }
  
  // Both volatile = high risk
  return 'high'
}

// Get strategy recommendation based on risk profile
export function getStrategyRecommendation(
  riskProfile: 'conservative' | 'moderate' | 'aggressive',
  ilRisk: 'low' | 'medium' | 'high'
): string {
  if (riskProfile === 'conservative') {
    if (ilRisk === 'low') return 'Wide Range (±20% from current price)'
    if (ilRisk === 'medium') return 'Wide Range (±15% from current price)'
    return 'Very Wide Range (±25% from current price)'
  }
  
  if (riskProfile === 'moderate') {
    if (ilRisk === 'low') return 'Moderate Range (±10% from current price)'
    if (ilRisk === 'medium') return 'Moderate Range (±8% from current price)'
    return 'Wide Range (±15% from current price)'
  }
  
  // Aggressive
  if (ilRisk === 'low') return 'Tight Range (±5% from current price)'
  if (ilRisk === 'medium') return 'Tight Range (±3% from current price)'
  return 'Moderate Range (±10% from current price)'
}