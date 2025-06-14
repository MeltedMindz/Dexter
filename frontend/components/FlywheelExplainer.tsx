'use client'

import { ArrowRight, Users, Zap, TrendingUp, Coins, Shield, Target, BarChart3 } from 'lucide-react'

export function FlywheelExplainer() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center space-y-6 mb-16">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900">
          The Dexter Protocol Flywheel
        </h1>
        <p className="text-xl text-slate-600 max-w-3xl mx-auto">
          A self-reinforcing growth engine where liquidity providers earn more, 
          $DEX stakers collect 100% of protocol fees, and the ecosystem grows stronger together.
        </p>
      </div>

      {/* Visual Flywheel */}
      <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-8 mb-16">
        <h2 className="text-2xl font-bold text-slate-900 text-center mb-12">
          🔄 How the Flywheel Works
        </h2>
        
        <div className="grid md:grid-cols-4 gap-8 relative">
          {/* Step 1: Liquidity Providers */}
          <div className="text-center space-y-4 relative">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
              <Users className="w-8 h-8 text-primary" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900">Liquidity Providers</h3>
              <p className="text-sm text-slate-600">
                Deposit ANY Uniswap V3 positions for auto-compounding
              </p>
            </div>
            {/* Arrow */}
            <div className="hidden md:block absolute -right-4 top-8">
              <ArrowRight className="w-6 h-6 text-slate-400" />
            </div>
          </div>

          {/* Step 2: Auto-Compound */}
          <div className="text-center space-y-4 relative">
            <div className="w-16 h-16 bg-success/10 rounded-full flex items-center justify-center mx-auto">
              <Zap className="w-8 h-8 text-success" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900">Auto-Compound</h3>
              <p className="text-sm text-slate-600">
                AI optimizes timing, 92% reinvested, 8% performance fee
              </p>
            </div>
            {/* Arrow */}
            <div className="hidden md:block absolute -right-4 top-8">
              <ArrowRight className="w-6 h-6 text-slate-400" />
            </div>
          </div>

          {/* Step 3: Fees to Stakers */}
          <div className="text-center space-y-4 relative">
            <div className="w-16 h-16 bg-warning/10 rounded-full flex items-center justify-center mx-auto">
              <Coins className="w-8 h-8 text-warning" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900">100% to Stakers</h3>
              <p className="text-sm text-slate-600">
                All fees converted to WETH, distributed to $DEX stakers
              </p>
            </div>
            {/* Arrow */}
            <div className="hidden md:block absolute -right-4 top-8">
              <ArrowRight className="w-6 h-6 text-slate-400" />
            </div>
          </div>

          {/* Step 4: Growth */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
              <TrendingUp className="w-8 h-8 text-primary" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-slate-900">Ecosystem Growth</h3>
              <p className="text-sm text-slate-600">
                Higher $DEX value attracts more LPs, creating more fees
              </p>
            </div>
          </div>
        </div>

        {/* Return arrow */}
        <div className="flex justify-center mt-12">
          <div className="flex items-center space-x-2 px-6 py-3 bg-slate-100 rounded-full">
            <TrendingUp className="w-5 h-5 text-slate-600" />
            <span className="text-sm font-medium text-slate-700">Growth Loop Continues</span>
            <TrendingUp className="w-5 h-5 text-slate-600" />
          </div>
        </div>
      </div>

      {/* Tokenomics Breakdown */}
      <div className="grid md:grid-cols-2 gap-8 mb-16">
        {/* $DEX Token Utility */}
        <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm">
          <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
            <span>💎</span>
            <span>$DEX Token Utility</span>
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-success/10 rounded-full flex items-center justify-center mt-1">
                <Coins className="w-3 h-3 text-success" />
              </div>
              <div>
                <h4 className="font-semibold text-slate-900">Revenue Sharing</h4>
                <p className="text-sm text-slate-600">
                  Stake $DEX to earn 100% of protocol fees in WETH
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center mt-1">
                <Shield className="w-3 h-3 text-primary" />
              </div>
              <div>
                <h4 className="font-semibold text-slate-900">Governance Rights</h4>
                <p className="text-sm text-slate-600">
                  Vote on protocol parameters and upgrades
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-warning/10 rounded-full flex items-center justify-center mt-1">
                <Target className="w-3 h-3 text-warning" />
              </div>
              <div>
                <h4 className="font-semibold text-slate-900">Aligned Incentives</h4>
                <p className="text-sm text-slate-600">
                  Token value grows with protocol success
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Revenue Distribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm">
          <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
            <span>💰</span>
            <span>Revenue Distribution</span>
          </h3>
          
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-slate-700">Performance Fee</span>
                <span className="text-sm font-bold text-slate-900">8%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div className="bg-primary h-2 rounded-full" style={{ width: '8%' }}></div>
              </div>
              <p className="text-xs text-slate-600 mt-1">Only charged on profits</p>
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-slate-700">LP Reinvestment</span>
                <span className="text-sm font-bold text-slate-900">92%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div className="bg-success h-2 rounded-full" style={{ width: '92%' }}></div>
              </div>
              <p className="text-xs text-slate-600 mt-1">Auto-compounded back to position</p>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <h4 className="font-semibold text-primary mb-2">Key Insight</h4>
              <p className="text-sm text-slate-700">
                <strong>100% of fees</strong> go to $DEX stakers as WETH rewards. 
                No team allocation, no treasury cuts.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Staking Mechanism */}
      <div className="bg-gradient-to-r from-primary/5 to-success/5 rounded-2xl border border-primary/20 p-8 mb-16">
        <h3 className="text-2xl font-bold text-slate-900 text-center mb-8">
          📊 Staking Mechanism
        </h3>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto shadow-sm">
              <Coins className="w-8 h-8 text-primary" />
            </div>
            <h4 className="text-lg font-semibold text-slate-900">Stake $DEX</h4>
            <p className="text-sm text-slate-600">
              Lock your $DEX tokens to start earning. No minimum amount, no lock-up period.
            </p>
          </div>
          
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto shadow-sm">
              <BarChart3 className="w-8 h-8 text-success" />
            </div>
            <h4 className="text-lg font-semibold text-slate-900">Earn Pro-Rata</h4>
            <p className="text-sm text-slate-600">
              Your share of WETH rewards equals your % of total staked $DEX.
            </p>
          </div>
          
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto shadow-sm">
              <TrendingUp className="w-8 h-8 text-warning" />
            </div>
            <h4 className="text-lg font-semibold text-slate-900">Compound Returns</h4>
            <p className="text-sm text-slate-600">
              More protocol usage = more fees = higher staking yields = higher token value.
            </p>
          </div>
        </div>
      </div>

      {/* Growth Metrics */}
      <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm">
        <h3 className="text-xl font-bold text-slate-900 text-center mb-8">
          🎯 Why This Model Works
        </h3>
        
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-6">
            <h4 className="text-lg font-semibold text-slate-900">For Liquidity Providers:</h4>
            <ul className="space-y-3">
              <li className="flex items-start space-x-3">
                <div className="w-5 h-5 bg-success/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs text-success">✓</span>
                </div>
                <span className="text-sm text-slate-700">Higher returns through AI-optimized compounding</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-5 h-5 bg-success/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs text-success">✓</span>
                </div>
                <span className="text-sm text-slate-700">No management hassle - fully automated</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-5 h-5 bg-success/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs text-success">✓</span>
                </div>
                <span className="text-sm text-slate-700">Performance-based fees (only pay when profitable)</span>
              </li>
            </ul>
          </div>
          
          <div className="space-y-6">
            <h4 className="text-lg font-semibold text-slate-900">For $DEX Stakers:</h4>
            <ul className="space-y-3">
              <li className="flex items-start space-x-3">
                <div className="w-5 h-5 bg-primary/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs text-primary">✓</span>
                </div>
                <span className="text-sm text-slate-700">Earn WETH (blue-chip asset) not random tokens</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-5 h-5 bg-primary/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs text-primary">✓</span>
                </div>
                <span className="text-sm text-slate-700">Revenue scales with protocol growth</span>
              </li>
              <li className="flex items-start space-x-3">
                <div className="w-5 h-5 bg-primary/20 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs text-primary">✓</span>
                </div>
                <span className="text-sm text-slate-700">Sustainable yield from real usage</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="text-center mt-16">
        <div className="bg-gradient-to-r from-primary to-success text-white rounded-2xl p-8">
          <h3 className="text-2xl font-bold mb-4">Ready to Join the Flywheel?</h3>
          <p className="text-lg opacity-90 mb-6">
            Connect your wallet to start earning with Dexter Protocol
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-primary px-8 py-3 rounded-lg font-semibold hover:bg-slate-100 transition-colors">
              Add Liquidity Position
            </button>
            <button className="bg-primary-600 bg-opacity-20 border border-white border-opacity-30 text-white px-8 py-3 rounded-lg font-semibold hover:bg-opacity-30 transition-colors">
              Stake $DEX Tokens
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}