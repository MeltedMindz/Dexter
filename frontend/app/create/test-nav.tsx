'use client'

import { useRouter } from 'next/navigation'
import Link from 'next/link'

export default function TestNavigation() {
  const router = useRouter()

  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-bold text-white">Navigation Test</h1>
      
      <div className="space-y-4">
        <div>
          <h3 className="text-lg text-white mb-2">Using Next.js Link:</h3>
          <Link 
            href="/" 
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Go to Home (Link)
          </Link>
        </div>
        
        <div>
          <h3 className="text-lg text-white mb-2">Using router.push():</h3>
          <button 
            onClick={() => {
              console.log('🚀 Test: Attempting router.push to /')
              router.push('/')
            }}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Go to Home (router.push)
          </button>
        </div>
        
        <div>
          <h3 className="text-lg text-white mb-2">Using window.location:</h3>
          <button 
            onClick={() => {
              console.log('🚀 Test: Attempting window.location.href = /')
              window.location.href = '/'
            }}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Go to Home (window.location)
          </button>
        </div>
      </div>
      
      <div className="mt-8">
        <h3 className="text-lg text-white mb-2">Back to Create Page:</h3>
        <Link 
          href="/create" 
          className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
        >
          Back to Create
        </Link>
      </div>
    </div>
  )
}