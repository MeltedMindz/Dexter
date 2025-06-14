import { Wallet, Zap, Shield, TrendingUp } from 'lucide-react'

export function ConnectPrompt() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="text-center space-y-8">
        {/* Hero Section */}
        <div className="space-y-4">
          <h1 className="text-4xl font-bold text-slate-900">
            AI-Powered Liquidity Management
          </h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Auto-compound your Uniswap V3 positions with performance-based fees. 
            Only pay when you profit.
          </p>
        </div>

        {/* Connect Prompt */}
        <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <Wallet className="w-8 h-8 text-primary" />
            <span className="text-2xl font-semibold text-slate-900">
              Connect Your Wallet
            </span>
          </div>
          <p className="text-slate-600 mb-8">
            Connect your wallet to start managing your liquidity positions with Dexter Protocol
          </p>
          <div className="mt-6">
            <a 
              href="/about"
              className="inline-flex items-center text-primary hover:text-primary/80 font-medium transition-colors"
            >
              Learn how the Dexter flywheel works →
            </a>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <div className="w-12 h-12 bg-success/10 rounded-lg flex items-center justify-center mb-4">
              <TrendingUp className="w-6 h-6 text-success" />
            </div>
            <h3 className="text-lg font-semibold text-slate-900 mb-2">
              Auto-Compounding
            </h3>
            <p className="text-slate-600 text-sm">
              Automatically reinvest your fees to maximize returns with optimal timing
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-lg font-semibold text-slate-900 mb-2">
              Performance Fees
            </h3>
            <p className="text-slate-600 text-sm">
              Only 8% fee on profits. No fees if your position loses money
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <div className="w-12 h-12 bg-warning/10 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-warning" />
            </div>
            <h3 className="text-lg font-semibold text-slate-900 mb-2">
              Any Token Pair
            </h3>
            <p className="text-slate-600 text-sm">
              Manage positions for ETH/USDC, WBTC/ETH, or any Uniswap V3 pair
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}