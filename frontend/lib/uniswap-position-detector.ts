// Uniswap V3 Position Detection with Ownership Verification
// This module properly detects and validates LP positions owned by a wallet

import { createPublicClient, http, getContract, Address } from 'viem'
import { base } from 'viem/chains'

// Uniswap V3 NonfungiblePositionManager contract on Base
const NONFUNGIBLE_POSITION_MANAGER = '0x03a520b32c04bf3beef7beb72e919cf822ed34f1'

// Minimal ABI for position detection
const POSITION_MANAGER_ABI = [
  {
    inputs: [{ name: 'tokenId', type: 'uint256' }],
    name: 'positions',
    outputs: [
      { name: 'nonce', type: 'uint96' },
      { name: 'operator', type: 'address' },
      { name: 'token0', type: 'address' },
      { name: 'token1', type: 'address' },
      { name: 'fee', type: 'uint24' },
      { name: 'tickLower', type: 'int24' },
      { name: 'tickUpper', type: 'int24' },
      { name: 'liquidity', type: 'uint128' },
      { name: 'feeGrowthInside0LastX128', type: 'uint256' },
      { name: 'feeGrowthInside1LastX128', type: 'uint256' },
      { name: 'tokensOwed0', type: 'uint128' },
      { name: 'tokensOwed1', type: 'uint128' }
    ],
    stateMutability: 'view',
    type: 'function'
  },
  {
    inputs: [{ name: 'tokenId', type: 'uint256' }],
    name: 'ownerOf',
    outputs: [{ name: '', type: 'address' }],
    stateMutability: 'view',
    type: 'function'
  }
] as const

// ERC20 ABI for token metadata
const ERC20_ABI = [
  {
    inputs: [],
    name: 'symbol',
    outputs: [{ name: '', type: 'string' }],
    stateMutability: 'view',
    type: 'function'
  },
  {
    inputs: [],
    name: 'decimals',
    outputs: [{ name: '', type: 'uint8' }],
    stateMutability: 'view',
    type: 'function'
  }
] as const

export interface UniswapV3Position {
  tokenId: string
  pool: string
  token0: string
  token1: string
  token0Symbol: string
  token1Symbol: string
  fee: number
  feeTier: string
  tickLower: number
  tickUpper: number
  liquidity: string
  tokensOwed0: string
  tokensOwed1: string
  isActive: boolean
  estimatedValueUSD: number
}

export class UniswapV3PositionDetector {
  private publicClient
  private positionManager

  constructor() {
    // Initialize Base network client
    this.publicClient = createPublicClient({
      chain: base,
      transport: http(process.env.NEXT_PUBLIC_BASE_RPC_FALLBACK || 'https://mainnet.base.org')
    })

    this.positionManager = getContract({
      address: NONFUNGIBLE_POSITION_MANAGER,
      abi: POSITION_MANAGER_ABI,
      client: this.publicClient
    })
  }

  /**
   * Detect and validate Uniswap V3 positions owned by a wallet
   * @param walletAddress - The wallet address to check
   * @param nftTokenIds - Array of NFT token IDs from Alchemy NFT scan
   * @returns Array of validated positions owned by the wallet
   */
  async detectValidPositions(
    walletAddress: string, 
    nftTokenIds: string[]
  ): Promise<UniswapV3Position[]> {
    const validPositions: UniswapV3Position[] = []

    console.log(`🔍 Validating ${nftTokenIds.length} potential Uniswap V3 positions for ${walletAddress}`)

    for (const tokenId of nftTokenIds) {
      try {
        // Step 1: Verify ownership on-chain
        const actualOwner = await this.positionManager.read.ownerOf([BigInt(tokenId)])
        
        if (actualOwner.toLowerCase() !== walletAddress.toLowerCase()) {
          console.log(`❌ Token ID ${tokenId} not owned by wallet (owner: ${actualOwner})`)
          continue
        }

        console.log(`✅ Token ID ${tokenId} ownership verified`)

        // Step 2: Get position data
        const positionData = await this.positionManager.read.positions([BigInt(tokenId)])
        
        // Step 3: Get token symbols
        const [token0Symbol, token1Symbol] = await Promise.all([
          this.getTokenSymbol(positionData[2] as Address), // token0
          this.getTokenSymbol(positionData[3] as Address)  // token1
        ])

        // Step 4: Check if position has liquidity
        const liquidity = positionData[7].toString()
        const isActive = BigInt(liquidity) > 0n

        // Step 5: Format position data
        const position: UniswapV3Position = {
          tokenId,
          pool: this.getPoolAddress(positionData[2] as Address, positionData[3] as Address, positionData[4] as number),
          token0: positionData[2] as Address,
          token1: positionData[3] as Address,
          token0Symbol,
          token1Symbol,
          fee: positionData[4] as number,
          feeTier: this.formatFeeTier(positionData[4] as number),
          tickLower: positionData[5] as number,
          tickUpper: positionData[6] as number,
          liquidity,
          tokensOwed0: positionData[10].toString(),
          tokensOwed1: positionData[11].toString(),
          isActive,
          estimatedValueUSD: isActive ? await this.estimatePositionValue(positionData) : 0
        }

        validPositions.push(position)
        console.log(`✅ Valid position found: ${token0Symbol}/${token1Symbol} (${position.feeTier})`)

      } catch (error) {
        console.error(`❌ Error validating position ${tokenId}:`, error)
        // Continue to next position instead of failing completely
      }
    }

    console.log(`🎯 Found ${validPositions.length} valid positions out of ${nftTokenIds.length} NFTs`)
    return validPositions
  }

  /**
   * Get token symbol from contract
   */
  private async getTokenSymbol(tokenAddress: Address): Promise<string> {
    try {
      const tokenContract = getContract({
        address: tokenAddress,
        abi: ERC20_ABI,
        client: this.publicClient
      })
      
      return await tokenContract.read.symbol() as string
    } catch (error) {
      console.error(`Error getting symbol for token ${tokenAddress}:`, error)
      return 'UNKNOWN'
    }
  }

  /**
   * Format fee tier for display
   */
  private formatFeeTier(fee: number): string {
    const feePercent = fee / 10000
    return `${feePercent}%`
  }

  /**
   * Get pool address (simplified - would need pool factory contract)
   */
  private getPoolAddress(token0: Address, token1: Address, fee: number): string {
    // In production, would compute pool address using CREATE2
    // For now, return a placeholder
    return `pool_${token0.slice(0, 8)}_${token1.slice(0, 8)}_${fee}`
  }

  /**
   * Estimate position value in USD (simplified)
   */
  private async estimatePositionValue(positionData: any): Promise<number> {
    // This is a simplified estimation
    // In production, would need:
    // 1. Current pool price
    // 2. Position range vs current price
    // 3. Token prices from oracle
    // 4. Liquidity calculations
    
    const liquidity = BigInt(positionData[7])
    
    // Very rough estimation based on liquidity amount
    // This would be much more sophisticated in production
    if (liquidity > BigInt('1000000000000000000')) return 1000 // Large position
    if (liquidity > BigInt('100000000000000000')) return 100   // Medium position
    if (liquidity > BigInt('10000000000000000')) return 10     // Small position
    
    return 1 // Dust position
  }

  /**
   * Extract NFT token IDs from Alchemy NFT response
   */
  static extractUniswapNFTIds(nftData: any): string[] {
    if (!nftData?.ownedNfts) return []

    return nftData.ownedNfts
      .filter((nft: any) => 
        nft.contract.address?.toLowerCase() === NONFUNGIBLE_POSITION_MANAGER.toLowerCase()
      )
      .map((nft: any) => nft.tokenId)
      .filter((tokenId: string) => tokenId && tokenId !== '0')
  }
}

export const positionDetector = new UniswapV3PositionDetector()