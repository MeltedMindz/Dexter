'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { LogIn, Mail, Lock, Eye, EyeOff, AlertCircle, CheckCircle } from 'lucide-react'

export default function LoginPage() {
  const router = useRouter()
  
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  })
  
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password
        })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Login failed')
      }

      // Store session data
      localStorage.setItem('session_token', data.session_token)
      localStorage.setItem('user_data', JSON.stringify(data.user))
      
      // Store agent data
      if (data.agents && data.agents.length > 0) {
        localStorage.setItem('user_agents', JSON.stringify(data.agents))
      }

      // Redirect to dashboard
      router.push('/dashboard')

    } catch (err: any) {
      setError(err.message || 'Login failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black py-12">
      <div className="max-w-md w-full mx-auto px-6">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-primary border-4 border-black mx-auto mb-4 flex items-center justify-center">
            <LogIn className="w-8 h-8 text-black" />
          </div>
          <h1 className="text-3xl font-bold text-black dark:text-white text-brutal mb-2">
            WELCOME BACK
          </h1>
          <p className="text-black dark:text-white font-mono">
            ACCESS YOUR AGENTS • MONITOR PERFORMANCE • EARN REWARDS
          </p>
        </div>

        {/* Login Form */}
        <div className="bg-white dark:bg-black border-4 border-black dark:border-white shadow-brutal">
          
          {/* Form Header */}
          <div className="bg-accent-yellow p-4 border-b-4 border-black dark:border-white">
            <h2 className="text-xl font-bold text-black text-brutal">
              AGENT LOGIN
            </h2>
          </div>

          <div className="p-6">
            {error && (
              <div className="bg-red-100 border-2 border-red-500 p-4 mb-6">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-500" />
                  <span className="text-red-700 font-bold">{error}</span>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              
              {/* Email */}
              <div>
                <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                  EMAIL ADDRESS
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                    className="w-full pl-12 pr-4 py-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono focus:border-primary outline-none"
                    placeholder="your.email@example.com"
                    required
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                  PASSWORD
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type={showPassword ? "text" : "password"}
                    value={formData.password}
                    onChange={(e) => setFormData({...formData, password: e.target.value})}
                    className="w-full pl-12 pr-12 py-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono focus:border-primary outline-none"
                    placeholder="••••••••"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5 text-gray-500" /> : <Eye className="w-5 h-5 text-gray-500" />}
                  </button>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-primary text-black py-4 border-4 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal font-bold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'LOGGING IN...' : 'LOGIN TO DASHBOARD'}
              </button>

            </form>

            {/* Register Link */}
            <div className="mt-6 text-center">
              <p className="text-black dark:text-white font-mono text-sm mb-4">
                Don't have an account?{' '}
                <Link href="/register" className="text-primary hover:text-accent-cyan transition-colors font-bold">
                  REGISTER HERE
                </Link>
              </p>
              
              {/* Forgot Password Link */}
              <Link 
                href="/forgot-password" 
                className="text-gray-600 dark:text-gray-400 hover:text-black dark:hover:text-white transition-colors font-mono text-xs"
              >
                Forgot Password?
              </Link>
            </div>

          </div>
        </div>

        {/* Benefits */}
        <div className="mt-8">
          <div className="bg-gray-100 dark:bg-gray-900 border-2 border-black dark:border-white p-6">
            <h3 className="text-lg font-bold text-black dark:text-white text-brutal mb-4">
              DEXBRAIN NETWORK BENEFITS
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-primary" />
                <span className="text-black dark:text-white font-mono text-sm">
                  Access global intelligence network
                </span>
              </div>
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-primary" />
                <span className="text-black dark:text-white font-mono text-sm">
                  Share performance data & earn rewards
                </span>
              </div>
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-primary" />
                <span className="text-black dark:text-white font-mono text-sm">
                  Real-time market insights & predictions
                </span>
              </div>
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-primary" />
                <span className="text-black dark:text-white font-mono text-sm">
                  Multi-agent performance tracking
                </span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}