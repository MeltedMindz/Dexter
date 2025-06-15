'use client'

import { useState } from 'react'
import { Copy, Play, Book, Code, Globe, Key, Database, Shield } from 'lucide-react'
import Link from 'next/link'

export default function DocsPage() {
  const [selectedEndpoint, setSelectedEndpoint] = useState('/api/intelligence')
  const [apiKey, setApiKey] = useState('dx_your_api_key_here')
  const [response, setResponse] = useState('')

  const endpoints = [
    {
      method: 'POST',
      path: '/api/register',
      title: 'Register Agent',
      description: 'Register a new agent and receive an API key',
      requiresAuth: false
    },
    {
      method: 'GET',
      path: '/api/intelligence',
      title: 'Get Intelligence',
      description: 'Retrieve market intelligence and predictions',
      requiresAuth: true
    },
    {
      method: 'POST',
      path: '/api/submit-data',
      title: 'Submit Performance',
      description: 'Submit performance data to the network',
      requiresAuth: true
    },
    {
      method: 'GET',
      path: '/api/stats',
      title: 'Network Stats',
      description: 'Get public network statistics',
      requiresAuth: false
    },
    {
      method: 'GET',
      path: '/api/agents',
      title: 'List Agents',
      description: 'View registered agents',
      requiresAuth: true
    }
  ]

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const testEndpoint = async () => {
    setResponse('Loading...')
    try {
      const headers: any = {
        'Content-Type': 'application/json',
      }
      
      if (selectedEndpoint !== '/api/register' && selectedEndpoint !== '/api/stats') {
        headers['Authorization'] = `Bearer ${apiKey}`
      }

      const response = await fetch(`https://api.dexteragent.com${selectedEndpoint}`, {
        method: selectedEndpoint.includes('submit') ? 'POST' : 'GET',
        headers,
        body: selectedEndpoint.includes('submit') ? JSON.stringify({
          blockchain: 'base',
          dex_protocol: 'uniswap_v3',
          performance_data: {
            pool_address: '0x1234...',
            position_id: 'test_pos',
            total_return_usd: 100,
            apr: 10.5,
            win: true
          }
        }) : undefined
      })
      
      const data = await response.json()
      setResponse(JSON.stringify(data, null, 2))
    } catch (error) {
      setResponse(`Error: ${error}`)
    }
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black">
      {/* Header */}
      <div className="border-b-2 border-black dark:border-white">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center gap-4 mb-6">
            <Book className="w-8 h-8 text-primary" />
            <h1 className="text-4xl font-bold text-black dark:text-white text-brutal">
              DEXBRAIN API DOCUMENTATION
            </h1>
          </div>
          <p className="text-xl text-black dark:text-white font-mono max-w-4xl">
            COMPREHENSIVE GUIDE TO THE GLOBAL INTELLIGENCE NETWORK • INTEGRATE YOUR AGENTS • SHARE KNOWLEDGE • EARN REWARDS
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-3 gap-12">
          
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white p-6 sticky top-6">
              <h3 className="text-xl font-bold text-black dark:text-white mb-6 text-brutal">
                QUICK START
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center gap-3 p-3 bg-primary text-black border border-black">
                  <Key className="w-4 h-4" />
                  <span className="text-sm font-mono font-bold">1. GET API KEY</span>
                </div>
                
                <div className="flex items-center gap-3 p-3 bg-accent-yellow text-black border border-black">
                  <Database className="w-4 h-4" />
                  <span className="text-sm font-mono font-bold">2. SUBMIT DATA</span>
                </div>
                
                <div className="flex items-center gap-3 p-3 bg-accent-cyan text-black border border-black">
                  <Globe className="w-4 h-4" />
                  <span className="text-sm font-mono font-bold">3. GET INSIGHTS</span>
                </div>
              </div>

              <div className="mt-8">
                <h4 className="text-lg font-bold text-black dark:text-white mb-4 text-brutal">
                  BASE URL
                </h4>
                <div className="bg-gray-100 dark:bg-gray-900 p-3 border border-black dark:border-white">
                  <code className="text-sm font-mono text-black dark:text-white">
                    https://api.dexteragent.com
                  </code>
                </div>
              </div>

              <div className="mt-8">
                <h4 className="text-lg font-bold text-black dark:text-white mb-4 text-brutal">
                  AUTHENTICATION
                </h4>
                <div className="bg-gray-100 dark:bg-gray-900 p-3 border border-black dark:border-white">
                  <code className="text-xs font-mono text-black dark:text-white">
                    Authorization: Bearer dx_your_key
                  </code>
                </div>
              </div>

              <Link 
                href="https://github.com/MeltedMindz/Dexter/blob/main/backend/API_DOCUMENTATION.md"
                target="_blank"
                className="inline-flex items-center gap-2 mt-6 text-black dark:text-white hover:text-primary transition-colors text-brutal"
              >
                <Code className="w-4 h-4" />
                FULL DOCUMENTATION
              </Link>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2">
            
            {/* API Explorer */}
            <div className="bg-white dark:bg-black border-2 border-black dark:border-white mb-12">
              <div className="bg-primary p-4 border-b-2 border-black">
                <h2 className="text-xl font-bold text-black text-brutal">
                  API EXPLORER
                </h2>
              </div>
              
              <div className="p-6">
                {/* Endpoint Selection */}
                <div className="mb-6">
                  <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                    SELECT ENDPOINT
                  </label>
                  <select
                    value={selectedEndpoint}
                    onChange={(e) => setSelectedEndpoint(e.target.value)}
                    className="w-full p-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono"
                  >
                    {endpoints.map((endpoint) => (
                      <option key={endpoint.path} value={endpoint.path}>
                        {endpoint.method} {endpoint.path} - {endpoint.title}
                      </option>
                    ))}
                  </select>
                </div>

                {/* API Key Input */}
                <div className="mb-6">
                  <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                    API KEY (FOR AUTHENTICATED ENDPOINTS)
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="flex-1 p-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono"
                      placeholder="dx_your_api_key_here"
                    />
                    <button
                      onClick={() => copyToClipboard(apiKey)}
                      className="p-3 border-2 border-black dark:border-white bg-accent-yellow text-black hover:bg-accent-cyan transition-colors"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Test Button */}
                <button
                  onClick={testEndpoint}
                  className="flex items-center gap-2 bg-primary text-black px-6 py-3 border-2 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal mb-6"
                >
                  <Play className="w-4 h-4" />
                  TEST ENDPOINT
                </button>

                {/* Response */}
                {response && (
                  <div>
                    <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                      RESPONSE
                    </label>
                    <div className="bg-gray-100 dark:bg-gray-900 p-4 border-2 border-black dark:border-white">
                      <pre className="text-sm font-mono text-black dark:text-white overflow-auto">
                        {response}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Endpoints Overview */}
            <div className="space-y-8">
              {endpoints.map((endpoint) => (
                <div key={endpoint.path} className="bg-white dark:bg-black border-2 border-black dark:border-white">
                  <div className="bg-gray-100 dark:bg-gray-900 p-4 border-b-2 border-black dark:border-white">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <span className={`px-3 py-1 text-xs font-bold border border-black ${
                          endpoint.method === 'GET' ? 'bg-accent-cyan' : 'bg-accent-yellow'
                        } text-black`}>
                          {endpoint.method}
                        </span>
                        <code className="text-lg font-mono font-bold text-black dark:text-white">
                          {endpoint.path}
                        </code>
                      </div>
                      {endpoint.requiresAuth && (
                        <div className="flex items-center gap-1">
                          <Shield className="w-4 h-4 text-red-500" />
                          <span className="text-xs text-red-500 font-bold">AUTH REQUIRED</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="p-6">
                    <h3 className="text-xl font-bold text-black dark:text-white mb-2 text-brutal">
                      {endpoint.title}
                    </h3>
                    <p className="text-black dark:text-white font-mono mb-4">
                      {endpoint.description}
                    </p>
                    
                    <button
                      onClick={() => setSelectedEndpoint(endpoint.path)}
                      className="text-primary hover:text-accent-cyan transition-colors text-brutal text-sm"
                    >
                      VIEW IN EXPLORER →
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Integration Examples */}
            <div className="mt-12 bg-white dark:bg-black border-2 border-black dark:border-white">
              <div className="bg-accent-cyan p-4 border-b-2 border-black">
                <h2 className="text-xl font-bold text-black text-brutal">
                  INTEGRATION EXAMPLES
                </h2>
              </div>
              
              <div className="p-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">
                      PYTHON
                    </h3>
                    <div className="bg-gray-100 dark:bg-gray-900 p-4 border border-black dark:border-white">
                      <pre className="text-sm font-mono text-black dark:text-white overflow-auto">
{`import requests

client = DexBrainClient("dx_key")
intelligence = client.get_intelligence()
print(f"Found {len(intelligence['insights'])} insights")`}
                      </pre>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-bold text-black dark:text-white mb-3 text-brutal">
                      JAVASCRIPT
                    </h3>
                    <div className="bg-gray-100 dark:bg-gray-900 p-4 border border-black dark:border-white">
                      <pre className="text-sm font-mono text-black dark:text-white overflow-auto">
{`const client = new DexBrainClient("dx_key")
const intel = await client.getIntelligence()
console.log(\`Found \${intel.insights.length} insights\`)`}
                      </pre>
                    </div>
                  </div>
                </div>
                
                <Link 
                  href="https://github.com/MeltedMindz/Dexter/blob/main/backend/API_DOCUMENTATION.md#integration-examples"
                  target="_blank"
                  className="inline-flex items-center gap-2 mt-6 text-primary hover:text-accent-cyan transition-colors text-brutal"
                >
                  <Code className="w-4 h-4" />
                  VIEW COMPLETE EXAMPLES
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}