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
}

export function AIChat({ isOpen, onClose, walletAddress }: AIChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'ai',
      content: walletAddress 
        ? `Hello! I've analyzed your wallet ${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}. I can help you understand liquidity pools, assess risks, explain strategies, or answer any DeFi questions. What would you like to know?`
        : "Hello! I'm your AI liquidity assistant. I can help you understand pools, assess risks, explain strategies, and answer DeFi questions. What would you like to know?",
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
    setInputValue('')
    setIsTyping(true)

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = generateAIResponse(inputValue, walletAddress)
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: aiResponse,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, aiMessage])
      setIsTyping(false)
    }, 1500)
  }

  const generateAIResponse = (query: string, wallet?: string): string => {
    const lowerQuery = query.toLowerCase()
    
    if (lowerQuery.includes('pool') || lowerQuery.includes('liquidity')) {
      return `Based on current market conditions, here are some insights about liquidity pools:

🏊 **ETH/USDC 0.3%**: Currently showing 18.5% APR with moderate IL risk. Good for balanced exposure.

🏊 **ETH/USDT 0.3%**: Similar to ETH/USDC but with slightly lower fees at 16.2% APR.

📊 **Risk Assessment**: With current ETH volatility at ~45%, tight-range strategies could capture more fees but require more active management.

${wallet ? `For your specific wallet (${wallet.slice(0, 6)}...${wallet.slice(-4)}), I'd recommend starting with ETH/USDC given your current holdings.` : ''}

Would you like me to explain impermanent loss or help you understand different range strategies?`
    }
    
    if (lowerQuery.includes('risk') || lowerQuery.includes('impermanent') || lowerQuery.includes('il')) {
      return `Let me explain impermanent loss (IL) and risk management:

⚠️ **Impermanent Loss**: Occurs when token prices diverge from your entry ratio. The more volatile the pair, the higher the potential IL.

📈 **Risk Levels**:
• **Low Risk**: Stable pairs (USDC/USDT) - minimal IL
• **Medium Risk**: ETH/Stablecoin - moderate IL, good fee generation  
• **High Risk**: Volatile pairs (ETH/BTC) - high IL potential

🛡️ **Mitigation Strategies**:
1. **Wide ranges** reduce IL but capture fewer fees
2. **Active rebalancing** can minimize IL impact
3. **Fee generation** often compensates for IL in popular pools

The key is ensuring your fee earnings exceed any impermanent loss. Would you like me to calculate IL scenarios for specific pairs?`
    }
    
    if (lowerQuery.includes('strategy') || lowerQuery.includes('range')) {
      return `Here are the main liquidity provision strategies:

🎯 **Tight Range (Concentrated)**:
• Higher fee capture (80-95% of trades)
• Requires active management
• Best for stable pairs or trending markets

📏 **Wide Range (Conservative)**:
• Lower maintenance, more "set and forget"
• Captures 30-60% of trades
• Better for volatile or uncertain markets

🔄 **Dual Position (Gamma Style)**:
• Base position (wide range) + limit position (tight range)
• Balances fee capture with risk management
• 75% base, 25% limit is a common split

🤖 **AI-Managed**:
• Dynamic range adjustments based on volatility
• Automated rebalancing and fee compounding
• Uses ML models to predict optimal timing

Which strategy interests you most? I can provide specific parameters for your situation.`
    }
    
    if (lowerQuery.includes('gas') || lowerQuery.includes('fee') || lowerQuery.includes('cost')) {
      return `Gas optimization is crucial for profitable liquidity provision:

⛽ **Current Base Network Costs**:
• Mint Position: ~$2-5
• Add Liquidity: ~$1-3
• Remove Liquidity: ~$1-3
• Compound Fees: ~$0.50-2

💡 **Optimization Tips**:
1. **Batch operations** when possible
2. **Time transactions** during low usage (3-6 AM UTC)
3. **Use Base Network** instead of Mainnet (90% cheaper)
4. **Auto-compound** only when fees > gas costs

📊 **Profitability Threshold**:
• Manual management: $1000+ positions
• Auto-compounding: $500+ positions
• AI optimization: $200+ positions (due to efficiency)

Would you like me to estimate gas costs for a specific strategy or position size?`
    }

    if (lowerQuery.includes('vault') || lowerQuery.includes('dexter')) {
      return `Let me explain Dexter's vault system:

🏦 **Vault Types**:
• **Manual Vaults**: You control all parameters, AI provides insights
• **AI-Managed Vaults**: AI handles rebalancing, you maintain oversight
• **Hybrid Mode**: AI recommendations + manual approval

🎚️ **Fee Structure** (performance-based):
• Retail: 1% management, 10% performance
• Premium: 0.75% management, 7.5% performance  
• Institutional: 0.5% management, 5% performance

🧠 **AI Features**:
• Real-time market analysis
• Optimal range recommendations
• Gas-efficient rebalancing
• Risk assessment and alerts

✨ **Key Benefits**:
• ERC4626 standard compliance
• Battle-tested security patterns
• Optional AI control (not forced)
• Professional analytics dashboard

Would you like me to help you choose between manual and AI-managed vaults?`
    }
    
    // Default response
    return `I understand you're asking about "${query}". 

I can help you with:
• 🏊 **Pool Analysis**: Best pools for your tokens, APR calculations
• ⚠️ **Risk Assessment**: Impermanent loss, volatility analysis  
• 📊 **Strategy Planning**: Range selection, position sizing
• ⛽ **Gas Optimization**: Cost-efficient transaction timing
• 🏦 **Vault Selection**: Manual vs AI-managed options

${wallet ? `I have your wallet context (${wallet.slice(0, 6)}...${wallet.slice(-4)}) and can provide personalized recommendations.` : 'Connect your wallet for personalized analysis!'}

Could you be more specific about what you'd like to know? For example:
• "What's the best pool for ETH and USDC?"
• "How much impermanent loss should I expect?"
• "Should I use a tight or wide range strategy?"`
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
                AI LIQUIDITY ASSISTANT
              </h2>
              <p className="text-sm text-black font-mono">
                {walletAddress ? `Connected: ${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}` : 'No wallet connected'}
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
                    <span className="font-mono text-sm">AI is thinking...</span>
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
              placeholder="Ask about pools, risks, strategies, or anything DeFi..."
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
              "What's the best pool for ETH?",
              "Explain impermanent loss",
              "Gas optimization tips",
              "Manual vs AI vaults"
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