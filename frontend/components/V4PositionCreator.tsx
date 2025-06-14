'use client'

import React, { useState } from 'react'
import { useAccount } from 'wagmi'
import { useWalletHoldingsFixed } from '@/lib/hooks/useWalletHoldings-fixed'

export function V4PositionCreator() {
  const { isConnected } = useAccount()
  const { tokens: holdings, totalValue, isLoading } = useWalletHoldingsFixed()
  const [testState, setTestState] = useState('initial')
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center py-12">
        <h1 className="text-4xl font-bold text-white mb-4">
          🔧 Testing FIXED useWalletHoldings Hook
        </h1>
        <p className="text-gray-300 mb-8">
          Testing the optimized version that should NOT block navigation
        </p>
        <div className="bg-gray-800 rounded-lg p-6 border-2 border-white space-y-4">
          <p className="text-white">
            Wallet Connected: {isConnected ? 'Yes' : 'No'}
          </p>
          <p className="text-white">
            Holdings Loading: {isLoading ? 'Yes' : 'No'} (REAL DATA)
          </p>
          <p className="text-white">
            Total Value: ${totalValue.toFixed(2)} (REAL DATA)
          </p>
          <p className="text-white">
            Token Count: {holdings.length} (REAL DATA)
          </p>
          <p className="text-white">
            Test State: {testState}
          </p>
          <button 
            onClick={() => setTestState('clicked')}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Test Button
          </button>
        </div>
        <div className="mt-6 bg-green-800 border-2 border-green-400 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">🔧 Fixed Hook Features:</h3>
          <p className="text-green-200 text-sm">
            ✅ Only 3 useBalance hooks (ETH, WETH, USDC)
            <br />✅ Simplified useEffect with useMemo optimization
            <br />✅ Fixed prices (no API calls)
            <br />✅ No hooks in map function
            <br />✅ Proper enabled conditions to prevent unnecessary calls
          </p>
        </div>
      </div>
    </div>
  )
}