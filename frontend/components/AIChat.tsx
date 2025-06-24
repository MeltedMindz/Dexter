'use client'

import { useState, useRef, useEffect } from 'react'
import { X, Send, Brain, User, Loader2 } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'ai'
  content: string
  timestamp: Date
}

interface AIChatProps {
  isOpen: boolean
  onClose: () => void
  walletAddress?: string
  portfolioContext?: any
}

export function AIChat({ isOpen, onClose, walletAddress, portfolioContext }: AIChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'ai',
      content: walletAddress 
        ? `Hello! I'm Dexter, your professional liquidity management AI agent. I can see you have wallet ${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)} connected. I specialize in DeFi strategy analysis, risk assessment, and Base Network optimization. How can I assist with your liquidity management needs today?`
        : "Hello! I'm Dexter, your professional liquidity management AI agent specializing in DeFi trading strategies. I can help you understand protocols, assess market risks, optimize yields, and navigate decentralized finance with precision. What aspect of liquidity management interests you?",
      timestamp: new Date()
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = inputValue
    setInputValue('')
    setIsTyping(true)

    try {
      // Send to Dexter AI via API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          chatHistory: messages.map(msg => ({
            role: msg.type === 'user' ? 'user' : 'assistant',
            content: msg.content
          })),
          walletAddress,
          portfolioContext
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response from Dexter')
      }

      const data = await response.json()
      
      // Handle rate limiting and authentication responses
      if (data.rateLimited || data.requiresAuth) {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'ai',
          content: data.response,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, errorMessage])
        return
      }
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: data.response,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, aiMessage])
      
      // Show usage info if available
      if (data.usage) {
        console.log(`💬 Dexter Chat Usage: ${data.usage.requestsToday}/${data.usage.dailyLimit} daily, ${data.usage.requestsThisMinute}/${data.usage.minuteLimit} per minute`)
      }
    } catch (error) {
      console.error('Error getting Dexter response:', error)
      
      // Check if it's a rate limit error from the response
      let errorContent = "I'm experiencing technical difficulties with my processing systems. As a liquidity management specialist, I recommend checking back shortly while I resolve this issue. In the meantime, remember that proper risk management is crucial in any DeFi strategy."
      
      if (error instanceof Error && error.message.includes('rate limit')) {
        errorContent = "I'm temporarily unavailable due to high demand. This helps me maintain optimal service quality for all users while managing operational costs effectively. Please try again in a moment."
      }
      
      // Fallback response in Dexter's style
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: errorContent,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, fallbackMessage])
    } finally {
      setIsTyping(false)
    }
  }


  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-black border-4 border-black dark:border-white w-full max-w-4xl h-[80vh] flex flex-col shadow-brutal">
        {/* Header */}
        <div className="bg-primary border-b-2 border-black dark:border-white p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-black" />
            <div>
              <h2 className="text-xl font-bold text-black text-brutal">
                DEXTER AI - LIQUIDITY SPECIALIST
              </h2>
              <p className="text-sm text-black font-mono">
                {walletAddress ? `Connected: ${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}` : 'Professional DeFi Analysis Ready'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="bg-black text-white p-2 hover:opacity-80"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-900">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.type === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.type === 'ai' && (
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                    <Brain className="w-5 h-5 text-black" />
                  </div>
                )}
                
                <div
                  className={`max-w-[70%] p-4 border-2 ${
                    message.type === 'user'
                      ? 'bg-accent-cyan text-black border-black dark:border-white'
                      : 'bg-white dark:bg-black text-black dark:text-white border-black dark:border-white'
                  }`}
                >
                  <div className="whitespace-pre-wrap font-mono text-sm">
                    {message.content}
                  </div>
                  <div className="text-xs text-gray-500 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>

                {message.type === 'user' && (
                  <div className="w-8 h-8 bg-accent-cyan rounded-full flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-black" />
                  </div>
                )}
              </div>
            ))}
            
            {isTyping && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                  <Brain className="w-5 h-5 text-black" />
                </div>
                <div className="bg-white dark:bg-black text-black dark:text-white border-2 border-black dark:border-white p-4">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="font-mono text-sm">Dexter is analyzing...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t-2 border-black dark:border-white p-4 bg-white dark:bg-black">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask Dexter about liquidity strategies, risk assessment, or DeFi protocols..."
              className="flex-1 p-3 border-2 border-black dark:border-white bg-white dark:bg-black text-black dark:text-white font-mono text-sm focus:outline-none"
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || isTyping}
              className="bg-primary text-black px-6 py-3 border-2 border-black dark:border-white font-bold hover:opacity-80 disabled:opacity-50 flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              SEND
            </button>
          </div>
          
          {/* Quick Actions */}
          <div className="flex flex-wrap gap-2 mt-3">
            {[
              "Analyze my portfolio risk",
              "Best liquidity strategies for Base Network",
              "How do you assess market volatility?",
              "Compare manual vs AI-managed vaults"
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setInputValue(suggestion)}
                className="bg-gray-200 dark:bg-gray-700 text-black dark:text-white px-3 py-1 text-xs font-mono border border-gray-300 dark:border-gray-600 hover:opacity-80"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}