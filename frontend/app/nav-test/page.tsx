'use client'

import { useRouter } from 'next/navigation'
import Link from 'next/link'

export default function NavigationTestPage() {
  const router = useRouter()

  return (
    <div className="max-w-2xl mx-auto p-8 space-y-6">
      <h1 className="text-3xl font-bold text-white">Navigation Test Page</h1>
      
      <div className="bg-gray-800 p-6 rounded-lg space-y-4">
        <h2 className="text-xl text-white mb-4">Test Different Navigation Methods:</h2>
        
        <div className="space-y-4">
          <div>
            <h3 className="text-lg text-white mb-2">1. Next.js Link Component:</h3>
            <Link 
              href="/create" 
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 inline-block"
            >
              Go to Create Page (Link)
            </Link>
          </div>
          
          <div>
            <h3 className="text-lg text-white mb-2">2. Router Push:</h3>
            <button 
              onClick={() => {
                console.log('🚀 Test: router.push to /create')
                router.push('/create')
              }}
              className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
            >
              Go to Create Page (router.push)
            </button>
          </div>
          
          <div>
            <h3 className="text-lg text-white mb-2">3. Window Location:</h3>
            <button 
              onClick={() => {
                console.log('🚀 Test: window.location to /create')
                window.location.href = '/create'
              }}
              className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
            >
              Go to Create Page (window.location)
            </button>
          </div>
        </div>
        
        <div className="mt-6 pt-4 border-t border-gray-600">
          <h3 className="text-lg text-white mb-2">Other Pages:</h3>
          <div className="flex space-x-2">
            <Link href="/" className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">
              Home
            </Link>
            <Link href="/dashboard" className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">
              Dashboard
            </Link>
            <Link href="/positions" className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">
              Positions
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}