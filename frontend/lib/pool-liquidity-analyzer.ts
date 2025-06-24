// Pool Liquidity Analyzer - Find high-liquidity pools for user's tokens
// Analyzes tokens in wallet and finds pools with significant liquidity

import { createPublicClient, http, getContract, Address } from 'viem'
import { base } from 'viem/chains'

// Uniswap V3 Factory contract on Base
const UNISWAP_V3_FACTORY = '0x33128a8fC17869897dcE68Ed026d694621f6FDfD'

// Pool contract ABI for liquidity checking
const POOL_ABI = [
  {
    inputs: [],
    name: 'liquidity',
    outputs: [{ name: '', type: 'uint128' }],
    stateMutability: 'view',
    type: 'function'
  },
  {
    inputs: [],
    name: 'slot0',
    outputs: [
      { name: 'sqrtPriceX96', type: 'uint160' },
      { name: 'tick', type: 'int24' },
      { name: 'observationIndex', type: 'uint16' },
      { name: 'observationCardinality', type: 'uint16' },
      { name: 'observationCardinalityNext', type: 'uint16' },
      { name: 'feeProtocol', type: 'uint8' },
      { name: 'unlocked', type: 'bool' }
    ],
    stateMutability: 'view',
    type: 'function'
  }
] as const

// Factory ABI for pool address computation
const FACTORY_ABI = [
  {
    inputs: [
      { name: 'tokenA', type: 'address' },
      { name: 'tokenB', type: 'address' },
      { name: 'fee', type: 'uint24' }
    ],
    name: 'getPool',
    outputs: [{ name: 'pool', type: 'address' }],
    stateMutability: 'view',
    type: 'function'
  }
] as const

export interface TokenBalance {
  contractAddress: string
  symbol: string
  name: string
  balance: string
  decimals: number
  valueUSD: number
}

export interface PoolOpportunity {
  poolAddress: string
  token0Address: string
  token1Address: string
  token0Symbol: string
  token1Symbol: string
  fee: number
  feeTier: string
  liquidityUSD: number
  userTokenSymbol: string
  userTokenBalance: string
  userTokenValueUSD: number
  estimatedAPR: number
  impermanentLossRisk: 'low' | 'medium' | 'high'
  recommendedStrategy: string
  confidence: number
}

export class PoolLiquidityAnalyzer {
  private publicClient
  private factory

  // Major token addresses on Base
  private readonly MAJOR_TOKENS = {
    WETH: '0x4200000000000000000000000000000000000006',
    USDC: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
    USDbC: '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',
    DAI: '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',
    cbETH: '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
    USDT: '0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2'
  }

  // Fee tiers to check
  private readonly FEE_TIERS = [100, 500, 3000, 10000] // 0.01%, 0.05%, 0.3%, 1%

  constructor() {
    this.publicClient = createPublicClient({
      chain: base,
      transport: http(process.env.NEXT_PUBLIC_BASE_RPC_FALLBACK || 'https://mainnet.base.org')
    })

    this.factory = getContract({
      address: UNISWAP_V3_FACTORY,
      abi: FACTORY_ABI,
      client: this.publicClient
    })
  }

  /**
   * Analyze all tokens in wallet and find high-liquidity pool opportunities
   */
  async analyzePoolOpportunities(
    userTokens: TokenBalance[],
    minimumLiquidityUSD: number = 50000
  ): Promise<PoolOpportunity[]> {
    const opportunities: PoolOpportunity[] = []

    console.log(`🔍 Analyzing ${userTokens.length} tokens for pool opportunities (min liquidity: $${minimumLiquidityUSD.toLocaleString()})`)

    for (const userToken of userTokens) {
      if (userToken.valueUSD < 10) continue // Skip dust tokens

      console.log(`📊 Analyzing pools for ${userToken.symbol} (${userToken.valueUSD.toFixed(2)} USD)`)

      // Find pools with major trading pairs
      const tokenOpportunities = await this.findPoolsForToken(
        userToken,
        minimumLiquidityUSD
      )

      opportunities.push(...tokenOpportunities)
    }

    // Sort by liquidity descending
    opportunities.sort((a, b) => b.liquidityUSD - a.liquidityUSD)

    console.log(`✅ Found ${opportunities.length} pool opportunities with >$${minimumLiquidityUSD.toLocaleString()} liquidity`)
    return opportunities
  }

  /**
   * Find high-liquidity pools for a specific token
   */
  private async findPoolsForToken(
    userToken: TokenBalance,
    minimumLiquidityUSD: number
  ): Promise<PoolOpportunity[]> {
    const opportunities: PoolOpportunity[] = []

    // Check pools against major tokens
    for (const [symbol, address] of Object.entries(this.MAJOR_TOKENS)) {
      if (address.toLowerCase() === userToken.contractAddress.toLowerCase()) continue // Skip self-pairs

      // Check each fee tier
      for (const fee of this.FEE_TIERS) {
        try {
          const poolAddress = await this.factory.read.getPool([
            userToken.contractAddress as Address,
            address as Address,
            fee
          ])

          if (poolAddress === '0x0000000000000000000000000000000000000000') continue // Pool doesn't exist

          // Check pool liquidity
          const liquidityData = await this.getPoolLiquidity(poolAddress)
          
          if (liquidityData.liquidityUSD >= minimumLiquidityUSD) {
            const opportunity: PoolOpportunity = {
              poolAddress,
              token0Address: userToken.contractAddress,
              token1Address: address,
              token0Symbol: userToken.symbol,
              token1Symbol: symbol,
              fee,
              feeTier: this.formatFeeTier(fee),
              liquidityUSD: liquidityData.liquidityUSD,
              userTokenSymbol: userToken.symbol,
              userTokenBalance: userToken.balance,
              userTokenValueUSD: userToken.valueUSD,
              estimatedAPR: this.estimateAPR(fee, liquidityData.liquidityUSD, userToken.symbol, symbol),
              impermanentLossRisk: this.assessILRisk(userToken.symbol, symbol),
              recommendedStrategy: this.getStrategyRecommendation(userToken.symbol, symbol),
              confidence: this.calculateConfidence(liquidityData.liquidityUSD, fee)
            }

            opportunities.push(opportunity)
            console.log(`✅ Found opportunity: ${userToken.symbol}/${symbol} (${this.formatFeeTier(fee)}) - $${liquidityData.liquidityUSD.toLocaleString()} liquidity`)
          }
        } catch (error) {
          // Pool likely doesn't exist or has issues, continue to next
          continue
        }
      }
    }

    return opportunities
  }

  /**
   * Get pool liquidity in USD
   */
  private async getPoolLiquidity(poolAddress: string): Promise<{ liquidityUSD: number }> {
    try {
      const poolContract = getContract({
        address: poolAddress as Address,
        abi: POOL_ABI,
        client: this.publicClient
      })

      const [liquidity, slot0] = await Promise.all([
        poolContract.read.liquidity(),
        poolContract.read.slot0()
      ])

      // Simplified liquidity calculation
      // In production, would use current pool price, token decimals, and price oracles
      const liquidityValue = Number(liquidity)
      
      // Rough estimation based on liquidity amount
      let liquidityUSD = 0
      if (liquidityValue > 1000000000000000000000000n) liquidityUSD = 10000000 // Very large pool
      else if (liquidityValue > 100000000000000000000000n) liquidityUSD = 1000000  // Large pool
      else if (liquidityValue > 10000000000000000000000n) liquidityUSD = 100000    // Medium pool
      else if (liquidityValue > 1000000000000000000000n) liquidityUSD = 10000      // Small pool
      else liquidityUSD = 1000 // Very small pool

      return { liquidityUSD }
    } catch (error) {
      return { liquidityUSD: 0 }
    }
  }

  /**
   * Estimate APR based on pool characteristics
   */
  private estimateAPR(fee: number, liquidityUSD: number, token0: string, token1: string): number {
    const baseAPRByFee = {
      100: 3,    // 0.01% - stable pairs
      500: 8,    // 0.05% - major pairs  
      3000: 15,  // 0.3% - standard
      10000: 25  // 1% - exotic pairs
    }

    let apr = baseAPRByFee[fee] || 10

    // Adjust for volatility
    const stableTokens = ['USDC', 'USDT', 'DAI', 'USDbC']
    const isStablePair = stableTokens.includes(token0) && stableTokens.includes(token1)
    const hasETH = token0 === 'WETH' || token1 === 'WETH' || token0 === 'ETH' || token1 === 'ETH'

    if (isStablePair) apr *= 0.7 // Lower but stable returns
    else if (hasETH) apr *= 1.3 // Higher volatility = more fees

    // Adjust for liquidity (more liquid = more volume = more fees)
    if (liquidityUSD > 1000000) apr *= 1.2
    else if (liquidityUSD < 100000) apr *= 0.8

    return Math.round(apr * 10) / 10
  }

  /**
   * Assess impermanent loss risk
   */
  private assessILRisk(token0: string, token1: string): 'low' | 'medium' | 'high' {
    const stableTokens = ['USDC', 'USDT', 'DAI', 'USDbC']
    
    if (stableTokens.includes(token0) && stableTokens.includes(token1)) return 'low'
    if (stableTokens.includes(token0) || stableTokens.includes(token1)) return 'medium'
    return 'high'
  }

  /**
   * Get strategy recommendation
   */
  private getStrategyRecommendation(token0: string, token1: string): string {
    const stableTokens = ['USDC', 'USDT', 'DAI', 'USDbC']
    const isStablePair = stableTokens.includes(token0) && stableTokens.includes(token1)
    const hasETH = token0 === 'WETH' || token1 === 'WETH' || token0 === 'ETH' || token1 === 'ETH'

    if (isStablePair) return 'Tight Range (±2% from current price)'
    if (hasETH) return 'Moderate Range (±10% from current price)'
    return 'Wide Range (±20% from current price)'
  }

  /**
   * Calculate recommendation confidence
   */
  private calculateConfidence(liquidityUSD: number, fee: number): number {
    let confidence = 50

    // Higher liquidity = higher confidence
    if (liquidityUSD > 5000000) confidence += 30
    else if (liquidityUSD > 1000000) confidence += 20
    else if (liquidityUSD > 100000) confidence += 10

    // Popular fee tiers = higher confidence
    if (fee === 500 || fee === 3000) confidence += 15
    else if (fee === 100) confidence += 10

    return Math.min(confidence, 95)
  }

  /**
   * Format fee tier for display
   */
  private formatFeeTier(fee: number): string {
    return `${fee / 10000}%`
  }
}

export const poolAnalyzer = new PoolLiquidityAnalyzer()