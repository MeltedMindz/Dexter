'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAccount } from 'wagmi'
import Link from 'next/link'
import { User, Mail, Lock, Zap, Eye, EyeOff, CheckCircle, AlertCircle } from 'lucide-react'

export default function RegisterPage() {
  const router = useRouter()
  const { address } = useAccount()
  
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    agentName: '',
    riskProfile: 'conservative' as 'conservative' | 'aggressive' | 'hyper_aggressive'
  })
  
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    // Validation
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      setIsLoading(false)
      return
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters long')
      setIsLoading(false)
      return
    }

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          walletAddress: address,
          agentName: formData.agentName || undefined,
          riskProfile: formData.riskProfile
        })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Registration failed')
      }

      // Store session token
      localStorage.setItem('session_token', data.session_token)
      localStorage.setItem('user_data', JSON.stringify(data.user))
      
      // Store API key if agent was created
      if (data.agent) {
        localStorage.setItem('api_key', data.agent.api_key)
      }

      setSuccess(true)
      
      // Redirect to dashboard after 2 seconds
      setTimeout(() => {
        router.push('/dashboard')
      }, 2000)

    } catch (err: any) {
      setError(err.message || 'Registration failed')
    } finally {
      setIsLoading(false)
    }
  }

  if (success) {
    return (
      <div className="min-h-screen bg-white dark:bg-black flex items-center justify-center">
        <div className="max-w-md w-full mx-6">
          <div className="bg-primary border-4 border-black p-8 text-center shadow-brutal">
            <CheckCircle className="w-16 h-16 text-black mx-auto mb-4" />
            <h1 className="text-2xl font-bold text-black mb-4 text-brutal">
              REGISTRATION SUCCESSFUL!
            </h1>
            <p className="text-black font-mono mb-4">
              Your account has been created and you're being redirected to the dashboard.
            </p>
            <div className="animate-pulse text-black font-bold">
              Redirecting...
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black py-12">
      <div className="max-w-md w-full mx-auto px-6">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-primary border-4 border-black mx-auto mb-4 flex items-center justify-center">
            <User className="w-8 h-8 text-black" />
          </div>
          <h1 className="text-3xl font-bold text-black dark:text-white text-brutal mb-2">
            JOIN DEXBRAIN
          </h1>
          <p className="text-black dark:text-white font-mono">
            CREATE YOUR ACCOUNT • GET API ACCESS • START EARNING
          </p>
        </div>

        {/* Registration Form */}
        <div className="bg-white dark:bg-black border-4 border-black dark:border-white shadow-brutal">
          
          {/* Form Header */}
          <div className="bg-accent-cyan p-4 border-b-4 border-black dark:border-white">
            <h2 className="text-xl font-bold text-black text-brutal">
              REGISTER NEW AGENT
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
                    minLength={8}
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

              {/* Confirm Password */}
              <div>
                <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                  CONFIRM PASSWORD
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    value={formData.confirmPassword}
                    onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
                    className="w-full pl-12 pr-12 py-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono focus:border-primary outline-none"
                    placeholder="••••••••"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5 text-gray-500" /> : <Eye className="w-5 h-5 text-gray-500" />}
                  </button>
                </div>
              </div>

              {/* Agent Name (Optional) */}
              <div>
                <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                  AGENT NAME (OPTIONAL)
                </label>
                <div className="relative">
                  <Zap className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="text"
                    value={formData.agentName}
                    onChange={(e) => setFormData({...formData, agentName: e.target.value})}
                    className="w-full pl-12 pr-4 py-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono focus:border-primary outline-none"
                    placeholder="My Trading Bot"
                  />
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 font-mono">
                  Leave blank to create later
                </p>
              </div>

              {/* Risk Profile */}
              <div>
                <label className="block text-sm font-bold text-black dark:text-white mb-2 text-brutal">
                  RISK PROFILE
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { value: 'conservative', label: 'CONSERVATIVE', color: 'bg-accent-cyan' },
                    { value: 'aggressive', label: 'AGGRESSIVE', color: 'bg-accent-yellow' },
                    { value: 'hyper_aggressive', label: 'HYPER', color: 'bg-accent-magenta' }
                  ].map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setFormData({...formData, riskProfile: option.value as any})}
                      className={`p-3 border-2 border-black text-black font-bold text-xs transition-all ${
                        formData.riskProfile === option.value
                          ? `${option.color} shadow-brutal`
                          : 'bg-gray-100 hover:bg-gray-200'
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Connected Wallet */}
              {address && (
                <div className="bg-gray-100 dark:bg-gray-900 border-2 border-gray-300 dark:border-gray-700 p-4">
                  <h4 className="font-bold text-black dark:text-white text-brutal text-sm mb-2">
                    CONNECTED WALLET
                  </h4>
                  <p className="text-xs font-mono text-gray-600 dark:text-gray-400 break-all">
                    {address}
                  </p>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-primary text-black py-4 border-4 border-black shadow-brutal hover:shadow-brutal-lg transition-all duration-100 text-brutal font-bold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'CREATING ACCOUNT...' : 'CREATE ACCOUNT'}
              </button>

            </form>

            {/* Login Link */}
            <div className="mt-6 text-center">
              <p className="text-black dark:text-white font-mono text-sm">
                Already have an account?{' '}
                <Link href="/login" className="text-primary hover:text-accent-cyan transition-colors font-bold">
                  LOGIN HERE
                </Link>
              </p>
            </div>

          </div>
        </div>

        {/* Features */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-accent-cyan border-2 border-black p-4 text-center">
            <Zap className="w-6 h-6 text-black mx-auto mb-2" />
            <h3 className="font-bold text-black text-brutal text-sm">FREE API ACCESS</h3>
          </div>
          <div className="bg-accent-yellow border-2 border-black p-4 text-center">
            <User className="w-6 h-6 text-black mx-auto mb-2" />
            <h3 className="font-bold text-black text-brutal text-sm">3 FREE AGENTS</h3>
          </div>
          <div className="bg-accent-magenta border-2 border-black p-4 text-center">
            <CheckCircle className="w-6 h-6 text-black mx-auto mb-2" />
            <h3 className="font-bold text-black text-brutal text-sm">INSTANT SETUP</h3>
          </div>
        </div>

      </div>
    </div>
  )
}