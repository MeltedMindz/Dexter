import { Alchemy, Network, AlchemySettings } from 'alchemy-sdk'

const alchemyApiKey = process.env.NEXT_PUBLIC_ALCHEMY_API_KEY || 'ory0F2cLFNIXsovAmrtJj'

// Base Network Alchemy instance
const baseSettings: AlchemySettings = {
  apiKey: alchemyApiKey,
  network: Network.BASE_MAINNET,
}

// Mainnet Alchemy instance
const mainnetSettings: AlchemySettings = {
  apiKey: alchemyApiKey,
  network: Network.ETH_MAINNET,
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
    category?: string[]
  }
) => {
  const alchemy = network === 'base' ? alchemyBase : alchemyMainnet
  return await alchemy.core.getAssetTransfers({
    fromAddress: address,
    ...options
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

export default { alchemyBase, alchemyMainnet }