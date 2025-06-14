'use client'

import { PositionCard } from './PositionCard'
import { useData } from '@/lib/data-context'
import { RefreshCw, Clock } from 'lucide-react'

export function PositionsList() {
  const { positions, isLoading, lastUpdated, refreshData } = useData()

  const formatLastUpdated = (date: Date) => {
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
          <span>📈 Your Positions</span>
          {positions.length > 0 && (
            <span className="text-sm font-normal text-slate-500 dark:text-slate-400">
              ({positions.length})
            </span>
          )}
        </h2>
        
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1 text-xs text-slate-500 dark:text-slate-400">
            <Clock className="w-3 h-3" />
            <span>Updated {formatLastUpdated(lastUpdated)}</span>
          </div>
          <button
            onClick={refreshData}
            disabled={isLoading}
            className="flex items-center space-x-1 px-3 py-1.5 text-xs font-medium text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-dark-600 hover:bg-slate-200 dark:hover:bg-dark-500 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            <span>{isLoading ? 'Updating...' : 'Refresh'}</span>
          </button>
        </div>
      </div>
      
      {isLoading && positions.length === 0 ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white dark:bg-dark-700 rounded-xl border border-slate-200 dark:border-white/10 p-6 animate-pulse">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-slate-300 dark:bg-dark-600 rounded-lg"></div>
                <div className="flex-1">
                  <div className="h-4 bg-slate-300 dark:bg-dark-600 rounded mb-2"></div>
                  <div className="h-3 bg-slate-200 dark:bg-dark-500 rounded w-1/2"></div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="h-3 bg-slate-200 dark:bg-dark-500 rounded"></div>
                <div className="h-3 bg-slate-200 dark:bg-dark-500 rounded w-3/4"></div>
              </div>
            </div>
          ))}
        </div>
      ) : positions.length > 0 ? (
        <div className="space-y-4">
          {positions.map((position) => (
            <PositionCard key={position.id} position={position} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto mb-4 bg-slate-100 dark:bg-dark-600 rounded-full flex items-center justify-center">
            <span className="text-2xl">📊</span>
          </div>
          <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
            No positions found
          </h3>
          <p className="text-slate-600 dark:text-slate-400 mb-4">
            Connect your wallet and add liquidity to get started
          </p>
        </div>
      )}
    </div>
  )
}