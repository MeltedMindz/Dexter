'use client'

import { Target, BarChart3, Users } from 'lucide-react'

export function RevenuePool() {
  // Mock data - in real app this would come from API/contracts
  const poolData = {
    pendingWETH: 12.34,
    threshold: 0.1,
    totalStakers: 1247,
    isReadyToDistribute: true
  }

  const formatNumber = (value: number, decimals = 2) => 
    new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(value)

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center space-x-2">
        <span>💧 Protocol Revenue Pool</span>
      </h2>
      
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        <div className="grid md:grid-cols-3 gap-6">
          {/* Pending WETH */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                <Target className="w-4 h-4 text-primary" />
              </div>
              <span className="font-medium text-slate-900">Pending WETH</span>
            </div>
            <div className="text-2xl font-bold text-slate-900 mono-numbers">
              {formatNumber(poolData.pendingWETH, 2)} WETH
            </div>
            <div className="text-sm text-slate-600">
              Threshold: {formatNumber(poolData.threshold, 1)} WETH
              {poolData.isReadyToDistribute && (
                <span className="ml-2 text-success font-medium">
                  ✅ Ready to distribute
                </span>
              )}
            </div>
          </div>

          {/* Total Stakers */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-success/10 rounded-lg flex items-center justify-center">
                <Users className="w-4 h-4 text-success" />
              </div>
              <span className="font-medium text-slate-900">Total Stakers</span>
            </div>
            <div className="text-2xl font-bold text-slate-900 mono-numbers">
              {formatNumber(poolData.totalStakers, 0)} users
            </div>
          </div>

          {/* Actions */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-warning/10 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-4 h-4 text-warning" />
              </div>
              <span className="font-medium text-slate-900">Actions</span>
            </div>
            <div className="space-y-2">
              <button 
                className={`w-full px-4 py-2 rounded-lg font-medium transition-colors ${
                  poolData.isReadyToDistribute
                    ? 'bg-primary text-white hover:bg-primary/90'
                    : 'bg-slate-100 text-slate-400 cursor-not-allowed'
                }`}
                disabled={!poolData.isReadyToDistribute}
              >
                🎯 Trigger Distribution
              </button>
              <button className="w-full px-4 py-2 bg-slate-100 text-slate-700 rounded-lg font-medium hover:bg-slate-200 transition-colors">
                📊 View History
              </button>
            </div>
          </div>
        </div>

        {/* Distribution Info */}
        {poolData.isReadyToDistribute && (
          <div className="mt-6 p-4 bg-success/5 border border-success/20 rounded-lg">
            <p className="text-sm text-success">
              💡 <strong>Ready for distribution!</strong> 
              {formatNumber(poolData.pendingWETH, 2)} WETH will be distributed pro-rata to all {formatNumber(poolData.totalStakers, 0)} stakers.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}