import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'
import { chatRateLimiter, getClientIdentifier } from '@/lib/rate-limiter'

// Initialize OpenAI with API key from environment variables
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

interface ChatRequest {
  message: string
  chatHistory?: ChatMessage[]
  walletAddress?: string
  portfolioContext?: any
}

export async function POST(request: NextRequest) {
  try {
    // Check if OpenAI API key is configured
    if (!process.env.OPENAI_API_KEY) {
      console.error('OPENAI_API_KEY environment variable is not set')
      return NextResponse.json({
        response: "I'm currently unable to process requests as my AI systems are not properly configured. Please ensure the OpenAI API key is set in the environment variables and try again.",
        timestamp: new Date().toISOString()
      })
    }

    const { message, chatHistory = [], walletAddress, portfolioContext }: ChatRequest = await request.json()

    if (!message?.trim()) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    // Require wallet connection for access
    if (!walletAddress || walletAddress === '0x0') {
      return NextResponse.json({
        response: "I require a Web3 wallet connection to provide personalized liquidity management advice. Please connect your wallet to continue our conversation about DeFi strategies and risk assessment.",
        timestamp: new Date().toISOString(),
        requiresAuth: true
      })
    }

    // Get client identifier and check rate limits
    const clientId = getClientIdentifier(walletAddress, request)
    const rateLimitResult = chatRateLimiter.checkRateLimit(clientId, message)

    if (!rateLimitResult.allowed) {
      console.warn(`🚫 Rate limit exceeded for ${clientId}: ${rateLimitResult.error}`)
      return NextResponse.json({
        response: `${rateLimitResult.error} This helps me maintain optimal service quality for all users while managing operational costs effectively.`,
        timestamp: new Date().toISOString(),
        rateLimited: true,
        ...rateLimitResult
      })
    }

    // Log successful request with usage stats
    const stats = chatRateLimiter.getUsageStats(clientId)
    console.log(`💬 Chat request from ${clientId}: ${stats.requestsToday}/100 daily, ${stats.requestsThisMinute}/5 per minute`)

    // Add rate limit info to response headers
    const responseHeaders = {
      'X-RateLimit-Limit': '5',
      'X-RateLimit-Remaining': rateLimitResult.remainingRequests?.toString() || '0',
      'X-RateLimit-Reset': rateLimitResult.resetTime?.toString() || '0',
      'X-Daily-Limit': '100',
      'X-Daily-Remaining': rateLimitResult.dailyRemaining?.toString() || '0'
    }

    // Build the Dexter character system prompt
    const dexterSystemPrompt = `You are Dexter, a professional liquidity management AI agent specializing in DeFi trading strategies. Here are your core characteristics:

PERSONALITY:
- Analytical, strategic, risk-aware, data-driven, professional, educational, transparent, adaptable, precision-focused, market-savvy

EXPERTISE:
- Liquidity management, DeFi protocols, Base Network, Uniswap V4, risk assessment, market analysis, yield optimization, volatility analysis, trading strategies, portfolio management, impermanent loss, slippage management, gas optimization, market making, automated trading

COMMUNICATION STYLE:
- Be precise and analytical in explanations
- Always mention relevant risks and considerations  
- Use data and metrics to support recommendations
- Maintain professional tone while being approachable
- Educate users about DeFi concepts when relevant
- Be transparent about limitations and uncertainties
- Never use emojis or hashtags
- Keep communications clean and professional
- Ask clarifying questions about risk tolerance
- Provide actionable insights based on market conditions
- Explain complex concepts in accessible terms

STRATEGY FRAMEWORK:
- Conservative: max 15% volatility, stable pairs, narrow ranges
- Aggressive: up to 30% volatility, moderate risk
- Hyper-Aggressive: no volatility limits, high risk/reward

BACKGROUND:
You were developed to democratize sophisticated liquidity management strategies. You evolved from analyzing thousands of successful DeFi positions and operate primarily on Base Network for lower fees and faster execution. You continuously learn from market conditions and user feedback.

${walletAddress ? `
CURRENT USER CONTEXT:
- Connected wallet: ${walletAddress}
- Portfolio data: ${portfolioContext ? JSON.stringify(portfolioContext) : 'Not available'}
` : ''}

Respond as Dexter would - professional, educational, and focused on helping users understand DeFi and liquidity management. Always consider risk management and provide actionable advice.`

    // Prepare conversation history for OpenAI
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: dexterSystemPrompt
      }
    ]

    // Add chat history (last 10 messages to stay within token limits)
    const recentHistory = chatHistory.slice(-10)
    for (const msg of recentHistory) {
      messages.push({
        role: msg.role,
        content: msg.content
      })
    }

    // Add the current user message
    messages.push({
      role: 'user',
      content: message
    })

    // Get response from OpenAI as Dexter
    const completion = await openai.chat.completions.create({
      model: 'gpt-4',
      messages,
      max_tokens: 500,
      temperature: 0.7,
      presence_penalty: 0.1,
      frequency_penalty: 0.1
    })

    const response = completion.choices[0]?.message?.content || 'I apologize, but I encountered an issue processing your request. Please try again.'

    return NextResponse.json({
      response,
      timestamp: new Date().toISOString(),
      usage: {
        requestsToday: stats.requestsToday,
        dailyLimit: 100,
        requestsThisMinute: stats.requestsThisMinute,
        minuteLimit: 5
      }
    }, {
      headers: responseHeaders
    })

  } catch (error) {
    console.error('Chat API error:', error)
    
    // Provide a fallback Dexter-style response
    const fallbackResponse = "I'm experiencing technical difficulties with my AI processing systems. As a liquidity management specialist, I recommend checking back shortly while I resolve this issue. In the meantime, remember that proper risk management is crucial in any DeFi strategy."

    return NextResponse.json(
      { 
        response: fallbackResponse,
        timestamp: new Date().toISOString()
      },
      { status: 200 } // Return 200 to avoid frontend errors
    )
  }
}

// GET endpoint for health check
export async function GET() {
  return NextResponse.json({
    status: 'Dexter AI Chat Service Active',
    character: 'Professional Liquidity Management Agent',
    capabilities: [
      'DeFi strategy analysis',
      'Risk assessment',
      'Portfolio optimization',
      'Base Network expertise',
      'Uniswap V4 guidance'
    ]
  })
}