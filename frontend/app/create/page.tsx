'use client'

import { V4PositionCreator } from '@/components/V4PositionCreator'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useState } from 'react'

export default function CreatePositionPage() {
  const router = useRouter()
  const [testMode, setTestMode] = useState(false)

  if (testMode) {
    return (
      <div className="max-w-2xl mx-auto p-8 space-y-6">
        <h1 className="text-3xl font-bold text-white">Create Page - Test Mode</h1>
        
        <div className="bg-gray-800 p-6 rounded-lg space-y-4">
          <h2 className="text-xl text-white mb-4">Navigation Test from Create Page:</h2>
          
          <div className="space-y-4">
            <div>
              <h3 className="text-lg text-white mb-2">1. Next.js Link:</h3>
              <Link 
                href="/" 
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
              >
                Go to Home (Link)
              </Link>
            </div>
            
            <div>
              <h3 className="text-lg text-white mb-2">2. Router Push:</h3>
              <button 
                onClick={() => {
                  console.log('🚀 CREATE PAGE: router.push to /')
                  router.push('/')
                }}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
              >
                Go to Home (router.push)
              </button>
            </div>
            
            <div>
              <h3 className="text-lg text-white mb-2">3. Window Location:</h3>
              <button 
                onClick={() => {
                  console.log('🚀 CREATE PAGE: window.location to /')
                  window.location.href = '/'
                }}
                className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
              >
                Go to Home (window.location)
              </button>
            </div>
          </div>
          
          <div className="mt-6 pt-4 border-t border-gray-600">
            <button 
              onClick={() => setTestMode(false)}
              className="bg-yellow-500 text-black px-4 py-2 rounded hover:bg-yellow-600"
            >
              Exit Test Mode (Show V4PositionCreator)
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Test Mode Toggle */}
      <div className="fixed top-24 right-4 z-50">
        <button 
          onClick={() => setTestMode(true)}
          className="bg-yellow-500 text-black px-3 py-1 rounded text-sm hover:bg-yellow-600"
        >
          Test Nav
        </button>
      </div>
      
      <V4PositionCreator />
    </div>
  )
}