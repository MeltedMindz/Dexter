'use client'

import { Plus, RotateCcw, Coins } from 'lucide-react'

export function QuickActions() {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-900 flex items-center space-x-2">
        <span>🎯 Quick Actions</span>
      </h2>
      
      <div className="flex flex-wrap gap-4">
        <button className="bg-primary text-white px-6 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors flex items-center space-x-2 shadow-sm">
          <Plus className="w-5 h-5" />
          <span>Add Position</span>
        </button>
        
        <button className="bg-white text-slate-700 px-6 py-3 rounded-lg font-medium hover:bg-slate-50 transition-colors flex items-center space-x-2 border border-slate-200 shadow-sm">
          <RotateCcw className="w-5 h-5" />
          <span>Compound All</span>
        </button>
        
        <button className="bg-success text-white px-6 py-3 rounded-lg font-medium hover:bg-success/90 transition-colors flex items-center space-x-2 shadow-sm">
          <Coins className="w-5 h-5" />
          <span>Claim Rewards</span>
        </button>
      </div>
    </div>
  )
}