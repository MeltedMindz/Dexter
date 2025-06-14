'use client'

import { Plus, RotateCcw, Coins } from 'lucide-react'

export function QuickActions() {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
        <span className="text-2xl">🎯</span>
        <span>Quick Actions</span>
      </h2>
      
      <div className="flex flex-wrap gap-4">
        <button className="bg-green-400 text-black px-6 py-3 rounded-lg font-medium hover:bg-green-300 transition-colors flex items-center space-x-2 border-brutal shadow-brutal hover:shadow-brutal-lg">
          <Plus className="w-5 h-5" />
          <span>Add Position</span>
        </button>
        
        <button className="bg-white dark:bg-black text-slate-900 dark:text-white px-6 py-3 rounded-lg font-medium hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors flex items-center space-x-2 border-brutal shadow-brutal hover:shadow-brutal-lg">
          <RotateCcw className="w-5 h-5" />
          <span>Compound All</span>
        </button>
        
        <button className="bg-yellow-400 text-black px-6 py-3 rounded-lg font-medium hover:bg-yellow-300 transition-colors flex items-center space-x-2 border-brutal shadow-brutal hover:shadow-brutal-lg">
          <Coins className="w-5 h-5" />
          <span>Claim Rewards</span>
        </button>
      </div>
    </div>
  )
}