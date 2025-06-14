'use client'

import { useState } from 'react'
import { useAccount, useWriteContract, useWaitForTransactionReceipt } from 'wagmi'
import { parseEther, parseUnits, encodeFunctionData } from 'viem'
import { base } from 'wagmi/chains'

// Uniswap V4 contract addresses on Base
const CONTRACT_ADDRESSES = {
  POOL_MANAGER: '0x498581ff718922c3f8e6a244956af099b2652b2b',
  POSITION_MANAGER: '0x7c5f5a4bbd8fd63184577525326123b519429bdc',
  UNIVERSAL_ROUTER: '0x6ff5693b99212da76ad316178a184ab56d299b43',
  QUOTER: '0x0d5e0f971ed27fbff6c2837bf31316121532048d'
}

// Simplified ABI for PositionManager core functions
const POSITION_MANAGER_ABI = [
  {
    inputs: [
      { name: 'unlockData', type: 'bytes' },
      { name: 'deadline', type: 'uint256' }
    ],
    name: 'modifyLiquidities',
    outputs: [],
    stateMutability: 'payable',
    type: 'function'
  }
] as const

// Action types for encoding
const ACTIONS = {
  MINT_POSITION: 0,
  INCREASE_LIQUIDITY: 1,
  DECREASE_LIQUIDITY: 2,
  BURN_POSITION: 3,
  SETTLE_PAIR: 4,
  TAKE_PAIR: 5,
  SWEEP: 6,
  CLEAR_OR_TAKE: 7,
  CLOSE_CURRENCY: 8
} as const

export interface PoolKey {
  currency0: `0x${string}`
  currency1: `0x${string}`
  fee: number
  tickSpacing: number
  hooks: `0x${string}`
}

export interface TransactionState {
  isLoading: boolean
  isSuccess: boolean
  isError: boolean
  hash?: string
  error?: string
}

export function useV4Transactions() {
  const { address } = useAccount()
  const [transactionState, setTransactionState] = useState<TransactionState>({
    isLoading: false,
    isSuccess: false,
    isError: false
  })

  const { writeContract } = useWriteContract()

  // Helper function to encode actions
  const encodeActions = (actionNames: string[]): `0x${string}` => {
    const actionBytes = actionNames.map(name => {
      const actionValue = ACTIONS[name as keyof typeof ACTIONS]
      return actionValue.toString(16).padStart(2, '0')
    }).join('')
    return `0x${actionBytes}`
  }

  // Helper function to create deadline (30 minutes from now)
  const createDeadline = (): bigint => {
    return BigInt(Math.floor(Date.now() / 1000) + 1800)
  }

  // Mint new position
  const mintPosition = async (params: {
    poolKey: PoolKey
    tickLower: number
    tickUpper: number
    liquidity: string
    amount0Max: string
    amount1Max: string
    recipient: string
  }) => {
    if (!address) throw new Error('Wallet not connected')

    setTransactionState({ isLoading: true, isSuccess: false, isError: false })

    try {
      const actions = encodeActions(['MINT_POSITION', 'SETTLE_PAIR'])
      
      // Encode mint parameters
      const mintParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'poolKey', type: 'tuple', components: [
              { name: 'currency0', type: 'address' },
              { name: 'currency1', type: 'address' },
              { name: 'fee', type: 'uint24' },
              { name: 'tickSpacing', type: 'int24' },
              { name: 'hooks', type: 'address' }
            ]},
            { name: 'tickLower', type: 'int24' },
            { name: 'tickUpper', type: 'int24' },
            { name: 'liquidity', type: 'uint256' },
            { name: 'amount0Max', type: 'uint256' },
            { name: 'amount1Max', type: 'uint256' },
            { name: 'recipient', type: 'address' },
            { name: 'hookData', type: 'bytes' }
          ],
          name: 'mint',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'mint',
        args: [
          params.poolKey,
          params.tickLower,
          params.tickUpper,
          BigInt(params.liquidity),
          parseEther(params.amount0Max),
          parseUnits(params.amount1Max, 6), // Assuming USDC (6 decimals)
          params.recipient as `0x${string}`,
          '0x' as `0x${string}`
        ]
      })

      const settleParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'currency0', type: 'address' },
            { name: 'currency1', type: 'address' }
          ],
          name: 'settle',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'settle',
        args: [
          params.poolKey.currency0 as `0x${string}`,
          params.poolKey.currency1 as `0x${string}`
        ]
      })

      const unlockData = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'actions', type: 'bytes' },
            { name: 'params', type: 'bytes[]' }
          ],
          name: 'unlock',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'unlock',
        args: [
          actions,
          [mintParams, settleParams]
        ]
      })

      const hash = await writeContract({
        address: CONTRACT_ADDRESSES.POSITION_MANAGER as `0x${string}`,
        abi: POSITION_MANAGER_ABI,
        functionName: 'modifyLiquidities',
        args: [unlockData, createDeadline()],
        value: params.poolKey.currency0 === '0x0000000000000000000000000000000000000000' 
          ? parseEther(params.amount0Max) 
          : BigInt(0)
      })

      setTransactionState({
        isLoading: false,
        isSuccess: true,
        isError: false,
        hash: 'pending'
      })

      return 'pending'
    } catch (error) {
      setTransactionState({
        isLoading: false,
        isSuccess: false,
        isError: true,
        error: error instanceof Error ? error.message : 'Unknown error'
      })
      throw error
    }
  }

  // Increase liquidity
  const increaseLiquidity = async (params: {
    tokenId: number
    liquidity: string
    amount0Max: string
    amount1Max: string
  }) => {
    if (!address) throw new Error('Wallet not connected')

    setTransactionState({ isLoading: true, isSuccess: false, isError: false })

    try {
      const actions = encodeActions(['INCREASE_LIQUIDITY', 'SETTLE_PAIR'])
      
      const increaseParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'liquidity', type: 'uint256' },
            { name: 'amount0Max', type: 'uint256' },
            { name: 'amount1Max', type: 'uint256' },
            { name: 'hookData', type: 'bytes' }
          ],
          name: 'increase',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'increase',
        args: [
          BigInt(params.tokenId),
          BigInt(params.liquidity),
          parseEther(params.amount0Max),
          parseUnits(params.amount1Max, 6),
          '0x' as `0x${string}`
        ]
      })

      const settleParams = '0x' // Simplified for demo

      const unlockData = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'actions', type: 'bytes' },
            { name: 'params', type: 'bytes[]' }
          ],
          name: 'unlock',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'unlock',
        args: [
          actions,
          [increaseParams, settleParams]
        ]
      })

      const hash = await writeContract({
        address: CONTRACT_ADDRESSES.POSITION_MANAGER as `0x${string}`,
        abi: POSITION_MANAGER_ABI,
        functionName: 'modifyLiquidities',
        args: [unlockData, createDeadline()]
      })

      setTransactionState({
        isLoading: false,
        isSuccess: true,
        isError: false,
        hash: 'pending'
      })

      return 'pending'
    } catch (error) {
      setTransactionState({
        isLoading: false,
        isSuccess: false,
        isError: true,
        error: error instanceof Error ? error.message : 'Unknown error'
      })
      throw error
    }
  }

  // Decrease liquidity
  const decreaseLiquidity = async (params: {
    tokenId: number
    liquidity: string
    amount0Min: string
    amount1Min: string
    recipient: string
  }) => {
    if (!address) throw new Error('Wallet not connected')

    setTransactionState({ isLoading: true, isSuccess: false, isError: false })

    try {
      const actions = encodeActions(['DECREASE_LIQUIDITY', 'TAKE_PAIR'])
      
      const decreaseParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'liquidity', type: 'uint256' },
            { name: 'amount0Min', type: 'uint256' },
            { name: 'amount1Min', type: 'uint256' },
            { name: 'hookData', type: 'bytes' }
          ],
          name: 'decrease',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'decrease',
        args: [
          BigInt(params.tokenId),
          BigInt(params.liquidity),
          parseEther(params.amount0Min),
          parseUnits(params.amount1Min, 6),
          '0x' as `0x${string}`
        ]
      })

      const takeParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'currency0', type: 'address' },
            { name: 'currency1', type: 'address' },
            { name: 'recipient', type: 'address' }
          ],
          name: 'take',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'take',
        args: [
          '0x0000000000000000000000000000000000000000' as `0x${string}`, // ETH
          '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`, // USDC
          params.recipient as `0x${string}`
        ]
      })

      const unlockData = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'actions', type: 'bytes' },
            { name: 'params', type: 'bytes[]' }
          ],
          name: 'unlock',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'unlock',
        args: [
          actions,
          [decreaseParams, takeParams]
        ]
      })

      const hash = await writeContract({
        address: CONTRACT_ADDRESSES.POSITION_MANAGER as `0x${string}`,
        abi: POSITION_MANAGER_ABI,
        functionName: 'modifyLiquidities',
        args: [unlockData, createDeadline()]
      })

      setTransactionState({
        isLoading: false,
        isSuccess: true,
        isError: false,
        hash: 'pending'
      })

      return 'pending'
    } catch (error) {
      setTransactionState({
        isLoading: false,
        isSuccess: false,
        isError: true,
        error: error instanceof Error ? error.message : 'Unknown error'
      })
      throw error
    }
  }

  // Collect fees
  const collectFees = async (params: {
    tokenId: number
    recipient: string
  }) => {
    if (!address) throw new Error('Wallet not connected')

    setTransactionState({ isLoading: true, isSuccess: false, isError: false })

    try {
      const actions = encodeActions(['DECREASE_LIQUIDITY', 'TAKE_PAIR'])
      
      // Decrease with 0 liquidity to collect fees only
      const decreaseParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'liquidity', type: 'uint256' },
            { name: 'amount0Min', type: 'uint256' },
            { name: 'amount1Min', type: 'uint256' },
            { name: 'hookData', type: 'bytes' }
          ],
          name: 'decrease',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'decrease',
        args: [
          BigInt(params.tokenId),
          BigInt(0), // 0 liquidity = fees only
          BigInt(0),
          BigInt(0),
          '0x' as `0x${string}`
        ]
      })

      const takeParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'currency0', type: 'address' },
            { name: 'currency1', type: 'address' },
            { name: 'recipient', type: 'address' }
          ],
          name: 'take',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'take',
        args: [
          '0x0000000000000000000000000000000000000000' as `0x${string}`,
          '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`,
          params.recipient as `0x${string}`
        ]
      })

      const unlockData = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'actions', type: 'bytes' },
            { name: 'params', type: 'bytes[]' }
          ],
          name: 'unlock',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'unlock',
        args: [
          actions,
          [decreaseParams, takeParams]
        ]
      })

      const hash = await writeContract({
        address: CONTRACT_ADDRESSES.POSITION_MANAGER as `0x${string}`,
        abi: POSITION_MANAGER_ABI,
        functionName: 'modifyLiquidities',
        args: [unlockData, createDeadline()]
      })

      setTransactionState({
        isLoading: false,
        isSuccess: true,
        isError: false,
        hash: 'pending'
      })

      return 'pending'
    } catch (error) {
      setTransactionState({
        isLoading: false,
        isSuccess: false,
        isError: true,
        error: error instanceof Error ? error.message : 'Unknown error'
      })
      throw error
    }
  }

  // Burn position (close completely)
  const burnPosition = async (params: {
    tokenId: number
    amount0Min: string
    amount1Min: string
    recipient: string
  }) => {
    if (!address) throw new Error('Wallet not connected')

    setTransactionState({ isLoading: true, isSuccess: false, isError: false })

    try {
      const actions = encodeActions(['BURN_POSITION', 'TAKE_PAIR'])
      
      const burnParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'tokenId', type: 'uint256' },
            { name: 'amount0Min', type: 'uint256' },
            { name: 'amount1Min', type: 'uint256' },
            { name: 'hookData', type: 'bytes' }
          ],
          name: 'burn',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'burn',
        args: [
          BigInt(params.tokenId),
          parseEther(params.amount0Min),
          parseUnits(params.amount1Min, 6),
          '0x' as `0x${string}`
        ]
      })

      const takeParams = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'currency0', type: 'address' },
            { name: 'currency1', type: 'address' },
            { name: 'recipient', type: 'address' }
          ],
          name: 'take',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'take',
        args: [
          '0x0000000000000000000000000000000000000000' as `0x${string}`,
          '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913' as `0x${string}`,
          params.recipient as `0x${string}`
        ]
      })

      const unlockData = encodeFunctionData({
        abi: [{
          inputs: [
            { name: 'actions', type: 'bytes' },
            { name: 'params', type: 'bytes[]' }
          ],
          name: 'unlock',
          outputs: [],
          stateMutability: 'nonpayable',
          type: 'function'
        }],
        functionName: 'unlock',
        args: [
          actions,
          [burnParams, takeParams]
        ]
      })

      const hash = await writeContract({
        address: CONTRACT_ADDRESSES.POSITION_MANAGER as `0x${string}`,
        abi: POSITION_MANAGER_ABI,
        functionName: 'modifyLiquidities',
        args: [unlockData, createDeadline()]
      })

      setTransactionState({
        isLoading: false,
        isSuccess: true,
        isError: false,
        hash: 'pending'
      })

      return 'pending'
    } catch (error) {
      setTransactionState({
        isLoading: false,
        isSuccess: false,
        isError: true,
        error: error instanceof Error ? error.message : 'Unknown error'
      })
      throw error
    }
  }

  return {
    transactionState,
    mintPosition,
    increaseLiquidity,
    decreaseLiquidity,
    collectFees,
    burnPosition
  }
}