'use client'

import React, { useState } from 'react'
import { useAccount } from 'wagmi'
import { useWalletHoldings } from '@/lib/hooks/useWalletHoldings'

export function V4PositionCreator() {
  const { isConnected } = useAccount()
  const { tokens: holdings, totalValue, isLoading } = useWalletHoldings()
  const [testState, setTestState] = useState('initial')
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center py-12">
        <h1 className="text-4xl font-bold text-white mb-4">
          V4 Position Creator - Testing useWalletHoldings
        </h1>
        <p className="text-gray-300 mb-8">
          Step 2: Testing if useWalletHoldings hook blocks navigation
        </p>
        <div className="bg-gray-800 rounded-lg p-6 border-2 border-white space-y-4">
          <p className="text-white">
            Wallet Connected: {isConnected ? 'Yes' : 'No'}
          </p>
          <p className="text-white">
            Holdings Loading: {isLoading ? 'Yes' : 'No'}
          </p>
          <p className="text-white">
            Total Value: ${totalValue.toFixed(2)}
          </p>
          <p className="text-white">
            Token Count: {holdings.length}
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
        <div className="mt-6">
          <p className="text-gray-400 text-sm">
            If navbar navigation stops working, useWalletHoldings hook is the culprit
          </p>
        </div>
      </div>
    </div>
  )
}