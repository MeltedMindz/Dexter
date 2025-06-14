'use client'

import React, { useState } from 'react'
import { useAccount } from 'wagmi'

// Simplified mock data instead of useWalletHoldings
const mockHoldings = {
  tokens: [
    { symbol: 'ETH', balance: '0.025' },
    { symbol: 'USDC', balance: '42.50' },
    { symbol: 'DAI', balance: '27.34' }
  ],
  totalValue: 69.84,
  isLoading: false
}

export function V4PositionCreator() {
  const { isConnected } = useAccount()
  const [testState, setTestState] = useState('initial')
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center py-12">
        <h1 className="text-4xl font-bold text-white mb-4">
          ✅ CULPRIT FOUND: useWalletHoldings Hook
        </h1>
        <p className="text-gray-300 mb-8">
          Navigation stopped working when useWalletHoldings was added. Using mock data now.
        </p>
        <div className="bg-gray-800 rounded-lg p-6 border-2 border-white space-y-4">
          <p className="text-white">
            Wallet Connected: {isConnected ? 'Yes' : 'No'}
          </p>
          <p className="text-white">
            Holdings Loading: {mockHoldings.isLoading ? 'Yes' : 'No'} (Mock Data)
          </p>
          <p className="text-white">
            Total Value: ${mockHoldings.totalValue.toFixed(2)} (Mock Data)
          </p>
          <p className="text-white">
            Token Count: {mockHoldings.tokens.length} (Mock Data)
          </p>
          <p className="text-white">
            Test State: {testState}
          </p>
          <button 
            onClick={() => setTestState('clicked')}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Test Button - Should Work Now!
          </button>
        </div>
        <div className="mt-6 bg-red-800 border-2 border-red-400 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">🐛 Problem Identified:</h3>
          <p className="text-red-200 text-sm">
            useWalletHoldings hook causes navigation blocking due to:
            <br />• Multiple useBalance hooks (6 total)
            <br />• Complex useEffect with many dependencies  
            <br />• Re-render loops from price fetching
            <br />• Hooks called in map function (React rules violation)
          </p>
        </div>
      </div>
    </div>
  )
}