import { useState, useEffect } from 'react'
import { useAccount } from 'wagmi'
import { AssetTransfersCategory, TokenBalanceSuccess, AssetTransfersWithMetadataResult } from 'alchemy-sdk'
import { getTokenBalances, getTokenMetadata, getAssetTransfers, getNftsForOwner } from '../alchemy'

export interface EnhancedTokenBalance {
  contractAddress: string
  tokenBalance: string
  metadata?: {
    name?: string | null
    symbol?: string | null
    decimals?: number | null
    logo?: string | null
  }
}

export const useAlchemyTokenBalances = (network: 'base' | 'mainnet' = 'base') => {
  const { address } = useAccount()
  const [balances, setBalances] = useState<EnhancedTokenBalance[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!address) return

    const fetchBalances = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const result = await getTokenBalances(address, network)
        
        // Enhance with metadata
        const enhancedBalances = await Promise.all(
          result.tokenBalances
            .filter(token => token.tokenBalance && token.tokenBalance !== '0x0')
            .map(async (token) => {
              try {
                const metadata = await getTokenMetadata(token.contractAddress, network)
                return {
                  contractAddress: token.contractAddress,
                  tokenBalance: token.tokenBalance || '0',
                  metadata: {
                    name: metadata.name,
                    symbol: metadata.symbol,
                    decimals: metadata.decimals,
                    logo: metadata.logo
                  }
                }
              } catch {
                return {
                  contractAddress: token.contractAddress,
                  tokenBalance: token.tokenBalance || '0',
                  metadata: undefined
                }
              }
            })
        )
        
        setBalances(enhancedBalances)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch balances')
      } finally {
        setLoading(false)
      }
    }

    fetchBalances()
  }, [address, network])

  return { balances, loading, error, refetch: () => {
    if (address) {
      // Re-trigger the effect
      setBalances([])
    }
  }}
}

export const useAlchemyTransactionHistory = (network: 'base' | 'mainnet' = 'base') => {
  const { address } = useAccount()
  const [transactions, setTransactions] = useState<AssetTransfersWithMetadataResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!address) return

    const fetchTransactions = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const result = await getAssetTransfers(address, network, {
          category: [
            AssetTransfersCategory.EXTERNAL, 
            AssetTransfersCategory.ERC20, 
            AssetTransfersCategory.ERC721, 
            AssetTransfersCategory.ERC1155
          ]
        })
        
        setTransactions(result.transfers)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch transactions')
      } finally {
        setLoading(false)
      }
    }

    fetchTransactions()
  }, [address, network])

  return { transactions, loading, error }
}

export const useAlchemyNFTs = (network: 'base' | 'mainnet' = 'base') => {
  const { address } = useAccount()
  const [nfts, setNfts] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!address) return

    const fetchNFTs = async () => {
      setLoading(true)
      setError(null)
      
      try {
        const result = await getNftsForOwner(address, network)
        setNfts(result.ownedNfts)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch NFTs')
      } finally {
        setLoading(false)
      }
    }

    fetchNFTs()
  }, [address, network])

  return { nfts, loading, error }
}