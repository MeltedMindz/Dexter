import { Alchemy, Network, AlchemySettings, AssetTransfersCategory } from 'alchemy-sdk'

const alchemyApiKey = process.env.NEXT_PUBLIC_ALCHEMY_API_KEY

// Check if we have a valid Alchemy API key
const hasValidAlchemyKey = alchemyApiKey && 
  alchemyApiKey !== 'demo' && 
  alchemyApiKey !== 'demo-key-needs-replacement' && 
  alchemyApiKey.length > 10

// Base Network Alchemy instance
const baseSettings: AlchemySettings = {
  apiKey: alchemyApiKey || 'demo',
  network: Network.BASE_MAINNET,
  url: hasValidAlchemyKey ? undefined : 'https://mainnet.base.org',
}

// Mainnet Alchemy instance  
const mainnetSettings: AlchemySettings = {
  apiKey: alchemyApiKey || 'demo',
  network: Network.ETH_MAINNET,
  url: hasValidAlchemyKey ? undefined : 'https://eth.llamarpc.com',
}

export const alchemyBase = new Alchemy(baseSettings)
export const alchemyMainnet = new Alchemy(mainnetSettings)

// Enhanced token data fetching
export const getTokenMetadata = async (tokenAddress: string, network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getTokenMetadata(tokenAddress)
}

// Enhanced balance fetching
export const getTokenBalances = async (address: string, network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getTokenBalances(address)
}

// Get NFTs for address
export const getNftsForOwner = async (address: string, network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.nft.getNftsForOwner(address)
}

// Get transaction history
export const getAssetTransfers = async (
  address: string, 
  network: 'base' | 'mainnet' = 'base',
  options?: {
    fromBlock?: string
    toBlock?: string
    category?: AssetTransfersCategory[]
  }
) => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getAssetTransfers({
    fromAddress: address,
    category: options?.category || [AssetTransfersCategory.EXTERNAL, AssetTransfersCategory.ERC20],
    fromBlock: options?.fromBlock,
    toBlock: options?.toBlock,
    withMetadata: true
  })
}

// Enhanced gas estimation
export const getGasEstimate = async (transaction: any, network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.estimateGas(transaction)
}

// Get current gas prices
export const getGasPrice = async (network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getGasPrice()
}

// Enhanced block data
export const getBlockWithTransactions = async (blockNumber: number | string, network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getBlockWithTransactions(blockNumber)
}

// Get logs for contract events
export const getLogs = async (filter: any, network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getLogs(filter)
}

// WebSocket support for real-time updates
export const createWebSocketConnection = (network: 'base' | 'mainnet' = 'base') => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return alchemy.ws
}

const alchemyExports = { alchemyBase, alchemyMainnet }
export default alchemyExports