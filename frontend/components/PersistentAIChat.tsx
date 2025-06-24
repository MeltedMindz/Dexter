'use client'

import { useState } from 'react'
import { MessageCircle, Brain } from 'lucide-react'
import { AIChat } from './AIChat'
import { useAccount } from 'wagmi'

export function PersistentAIChat() {
  const [isOpen, setIsOpen] = useState(false)
  const { address } = useAccount()

  return (
    <>
      {/* Floating Chat Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 bg-primary text-black p-4 border-2 border-black dark:border-white shadow-brutal hover:shadow-brutal-lg transition-all duration-100 z-40 group"
        aria-label="Open AI Chat"
      >
        <div className="flex items-center gap-2">
          <Brain className="w-6 h-6" />
          <MessageCircle className="w-6 h-6" />
        </div>
        
        {/* Tooltip */}
        <div className="absolute bottom-full right-0 mb-2 px-3 py-2 bg-black text-white text-sm font-mono opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
          Chat with Dexter AI
        </div>
      </button>

      {/* Chat Component */}
      <AIChat 
        isOpen={isOpen} 
        onClose={() => setIsOpen(false)} 
        walletAddress={address}
      />
    </>
  )
}