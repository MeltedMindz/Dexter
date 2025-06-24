// Rate Limiting and Abuse Protection for Dexter AI Chat
// Multi-layered protection: Rate limiting + Usage caps + Abuse detection

interface RateLimitEntry {
  count: number
  resetTime: number
  dailyCount: number
  dailyResetTime: number
  blocked: boolean
  blockUntil?: number
}

interface AbusePattern {
  spamKeywords: string[]
  maxSimilarQueries: number
  maxRequestsPerSecond: number
}

class ChatRateLimiter {
  private store = new Map<string, RateLimitEntry>()
  private queryHistory = new Map<string, string[]>()
  
  // Configuration
  private readonly REQUESTS_PER_MINUTE = 5
  private readonly DAILY_REQUEST_LIMIT = 100
  private readonly MINUTE_IN_MS = 60 * 1000
  private readonly DAY_IN_MS = 24 * 60 * 60 * 1000
  private readonly BLOCK_DURATION_MS = 15 * 60 * 1000 // 15 minutes
  
  // Abuse detection patterns
  private readonly abusePatterns: AbusePattern = {
    spamKeywords: ['test', 'spam', 'hello', 'hi', 'hey', '123', 'aaa', 'lol'],
    maxSimilarQueries: 3, // Max identical queries in 10 minutes
    maxRequestsPerSecond: 3 // More than 3 requests per second = suspicious
  }

  private readonly blacklistedAddresses = new Set<string>([
    // Add known problematic addresses here
  ])

  /**
   * Check if a request should be allowed
   */
  public checkRateLimit(identifier: string, message?: string): {
    allowed: boolean
    error?: string
    remainingRequests?: number
    resetTime?: number
    dailyRemaining?: number
  } {
    // Check if identifier is blacklisted
    if (this.blacklistedAddresses.has(identifier)) {
      return {
        allowed: false,
        error: 'Access temporarily restricted. Please contact support if you believe this is an error.'
      }
    }

    const now = Date.now()
    let entry = this.store.get(identifier)

    // Initialize entry if doesn't exist
    if (!entry) {
      entry = {
        count: 0,
        resetTime: now + this.MINUTE_IN_MS,
        dailyCount: 0,
        dailyResetTime: now + this.DAY_IN_MS,
        blocked: false
      }
    }

    // Check if user is temporarily blocked
    if (entry.blocked && entry.blockUntil && now < entry.blockUntil) {
      const remainingBlockTime = Math.ceil((entry.blockUntil - now) / 60000)
      return {
        allowed: false,
        error: `Temporarily blocked due to suspicious activity. Please try again in ${remainingBlockTime} minutes.`
      }
    }

    // Reset minute counter if time window expired
    if (now >= entry.resetTime) {
      entry.count = 0
      entry.resetTime = now + this.MINUTE_IN_MS
    }

    // Reset daily counter if day expired
    if (now >= entry.dailyResetTime) {
      entry.dailyCount = 0
      entry.dailyResetTime = now + this.DAY_IN_MS
    }

    // Check daily limit
    if (entry.dailyCount >= this.DAILY_REQUEST_LIMIT) {
      return {
        allowed: false,
        error: `Daily limit reached (${this.DAILY_REQUEST_LIMIT} messages). Please try again tomorrow.`,
        dailyRemaining: 0
      }
    }

    // Check per-minute rate limit
    if (entry.count >= this.REQUESTS_PER_MINUTE) {
      const resetInSeconds = Math.ceil((entry.resetTime - now) / 1000)
      return {
        allowed: false,
        error: `Rate limit exceeded. Please wait ${resetInSeconds} seconds before sending another message.`,
        remainingRequests: 0,
        resetTime: entry.resetTime
      }
    }

    // Check for abuse patterns if message provided
    if (message && this.detectAbuse(identifier, message)) {
      entry.blocked = true
      entry.blockUntil = now + this.BLOCK_DURATION_MS
      this.store.set(identifier, entry)
      
      console.warn(`🚨 ABUSE DETECTED: ${identifier} - Message: "${message.substring(0, 50)}..."`)
      
      return {
        allowed: false,
        error: 'Suspicious activity detected. Access temporarily restricted.'
      }
    }

    // Increment counters
    entry.count += 1
    entry.dailyCount += 1
    entry.blocked = false
    delete entry.blockUntil
    
    // Store updated entry
    this.store.set(identifier, entry)

    // Track query for abuse detection
    if (message) {
      this.trackQuery(identifier, message)
    }

    return {
      allowed: true,
      remainingRequests: this.REQUESTS_PER_MINUTE - entry.count,
      resetTime: entry.resetTime,
      dailyRemaining: this.DAILY_REQUEST_LIMIT - entry.dailyCount
    }
  }

  /**
   * Detect abuse patterns in messages
   */
  private detectAbuse(identifier: string, message: string): boolean {
    const lowerMessage = message.toLowerCase().trim()
    
    // Check for spam keywords (short messages with spam keywords)
    if (lowerMessage.length < 10) {
      const hasSpamKeyword = this.abusePatterns.spamKeywords.some(keyword => 
        lowerMessage.includes(keyword)
      )
      if (hasSpamKeyword) return true
    }

    // Check for identical repeated queries
    const recentQueries = this.queryHistory.get(identifier) || []
    const identicalCount = recentQueries.filter(q => q === lowerMessage).length
    if (identicalCount >= this.abusePatterns.maxSimilarQueries) {
      return true
    }

    // Check for very rapid requests (basic timing check)
    const entry = this.store.get(identifier)
    if (entry && entry.count >= this.abusePatterns.maxRequestsPerSecond) {
      const timeWindow = Date.now() - (entry.resetTime - this.MINUTE_IN_MS)
      if (timeWindow < 1000) { // Less than 1 second for multiple requests
        return true
      }
    }

    return false
  }

  /**
   * Track recent queries for abuse detection
   */
  private trackQuery(identifier: string, message: string): void {
    const queries = this.queryHistory.get(identifier) || []
    const lowerMessage = message.toLowerCase().trim()
    
    // Keep last 10 queries
    queries.push(lowerMessage)
    if (queries.length > 10) {
      queries.shift()
    }
    
    this.queryHistory.set(identifier, queries)
  }

  /**
   * Manually block a wallet address
   */
  public blockAddress(address: string): void {
    this.blacklistedAddresses.add(address)
    console.warn(`🚫 MANUALLY BLOCKED: ${address}`)
  }

  /**
   * Unblock a wallet address
   */
  public unblockAddress(address: string): void {
    this.blacklistedAddresses.delete(address)
    const entry = this.store.get(address)
    if (entry) {
      entry.blocked = false
      delete entry.blockUntil
      this.store.set(address, entry)
    }
    console.log(`✅ UNBLOCKED: ${address}`)
  }

  /**
   * Get usage stats for an address
   */
  public getUsageStats(identifier: string): {
    requestsThisMinute: number
    requestsToday: number
    dailyLimitRemaining: number
    isBlocked: boolean
  } {
    const entry = this.store.get(identifier)
    const now = Date.now()
    
    if (!entry) {
      return {
        requestsThisMinute: 0,
        requestsToday: 0,
        dailyLimitRemaining: this.DAILY_REQUEST_LIMIT,
        isBlocked: false
      }
    }

    // Reset counters if expired
    const requestsThisMinute = now >= entry.resetTime ? 0 : entry.count
    const requestsToday = now >= entry.dailyResetTime ? 0 : entry.dailyCount
    
    return {
      requestsThisMinute,
      requestsToday,
      dailyLimitRemaining: this.DAILY_REQUEST_LIMIT - requestsToday,
      isBlocked: entry.blocked && entry.blockUntil ? now < entry.blockUntil : false
    }
  }

  /**
   * Clean up expired entries (run periodically)
   */
  public cleanup(): void {
    const now = Date.now()
    
    for (const [identifier, entry] of this.store.entries()) {
      // Remove entries that are fully expired
      if (now >= entry.dailyResetTime && !entry.blocked) {
        this.store.delete(identifier)
        this.queryHistory.delete(identifier)
      }
    }
    
    console.log(`🧹 Cleaned up expired rate limit entries. Active entries: ${this.store.size}`)
  }
}

// Export singleton instance
export const chatRateLimiter = new ChatRateLimiter()

// Cleanup every hour
if (typeof window === 'undefined') { // Server-side only
  setInterval(() => {
    chatRateLimiter.cleanup()
  }, 60 * 60 * 1000) // 1 hour
}

// Helper function to get client identifier
export function getClientIdentifier(walletAddress?: string, request?: Request): string {
  // Prefer wallet address for authenticated users
  if (walletAddress && walletAddress !== '0x0') {
    return `wallet:${walletAddress.toLowerCase()}`
  }
  
  // Fallback to IP address for unauthenticated users
  const forwarded = request?.headers.get?.('x-forwarded-for')
  const ip = forwarded ? forwarded.split(',')[0] : 'unknown'
  return `ip:${ip}`
}