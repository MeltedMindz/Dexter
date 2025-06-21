import { NextRequest, NextResponse } from 'next/server'
import { v4 as uuidv4 } from 'uuid'
import pg from 'pg'
import crypto from 'crypto'

const { Pool } = pg

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/dexter_db'
})

interface CreateAgentRequest {
  name: string
  description?: string
  riskProfile: 'conservative' | 'aggressive' | 'hyper_aggressive'
  supportedBlockchains?: string[]
  supportedDexs?: string[]
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json(
        { error: 'Missing or invalid authorization header' },
        { status: 401 }
      )
    }

    const sessionToken = authHeader.substring(7)
    const body: CreateAgentRequest = await request.json()
    
    // Validate required fields
    if (!body.name || !body.riskProfile) {
      return NextResponse.json(
        { error: 'Name and risk profile are required' },
        { status: 400 }
      )
    }

    // Validate risk profile
    const validRiskProfiles = ['conservative', 'aggressive', 'hyper_aggressive']
    if (!validRiskProfiles.includes(body.riskProfile)) {
      return NextResponse.json(
        { error: 'Invalid risk profile' },
        { status: 400 }
      )
    }

    const client = await pool.connect()
    
    try {
      // Verify session and get user
      const sessionResult = await client.query(
        `SELECT us.user_id, u.email, u.subscription_tier
         FROM user_sessions us
         JOIN users u ON us.user_id = u.id
         WHERE us.session_token = $1 AND us.expires_at > NOW()`,
        [sessionToken]
      )

      if (sessionResult.rows.length === 0) {
        return NextResponse.json(
          { error: 'Invalid or expired session' },
          { status: 401 }
        )
      }

      const { user_id, email, subscription_tier } = sessionResult.rows[0]

      // Check agent limit based on subscription tier
      const agentCountResult = await client.query(
        'SELECT COUNT(*) as count FROM user_agents WHERE user_id = $1 AND is_active = true',
        [user_id]
      )
      
      const currentAgentCount = parseInt(agentCountResult.rows[0].count)
      const maxAgents = subscription_tier === 'premium' ? 10 : 3 // Free: 3, Premium: 10
      
      if (currentAgentCount >= maxAgents) {
        return NextResponse.json(
          { error: `Maximum ${maxAgents} agents allowed for ${subscription_tier} tier` },
          { status: 400 }
        )
      }

      // Generate unique agent ID
      const agentIdBase = body.name.toLowerCase().replace(/[^a-z0-9]/g, '_')
      const agentId = `user_${user_id}_${agentIdBase}_${Date.now()}`

      // Check if agent name already exists for this user
      const existingAgentResult = await client.query(
        'SELECT id FROM user_agents WHERE user_id = $1 AND name = $2',
        [user_id, body.name]
      )

      if (existingAgentResult.rows.length > 0) {
        return NextResponse.json(
          { error: 'Agent with this name already exists' },
          { status: 400 }
        )
      }

      // Register agent with DexBrain API
      const dexbrainUrl = process.env.DEXBRAIN_API_URL || 'http://localhost:8080'
      const dexbrainResponse = await fetch(`${dexbrainUrl}/api/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_id: agentId,
          metadata: {
            user_id: user_id,
            user_email: email,
            name: body.name,
            description: body.description || `${body.riskProfile} trading agent`,
            risk_profile: body.riskProfile,
            supported_blockchains: body.supportedBlockchains || ['base'],
            supported_dexs: body.supportedDexs || ['uniswap_v3', 'uniswap_v4'],
            registration_type: 'frontend_user',
            subscription_tier: subscription_tier,
            created_via: 'web_dashboard'
          }
        })
      })

      if (!dexbrainResponse.ok) {
        const errorData = await dexbrainResponse.text()
        console.error('DexBrain registration failed:', errorData)
        return NextResponse.json(
          { error: 'Failed to register agent with DexBrain network' },
          { status: 500 }
        )
      }

      const dexbrainData = await dexbrainResponse.json()

      // Store user-agent relationship in database
      const insertResult = await client.query(
        `INSERT INTO user_agents (
          user_id, agent_id, api_key_hash, name, description, risk_profile,
          supported_blockchains, supported_dexs, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        RETURNING id, created_at`,
        [
          user_id,
          agentId,
          hashApiKey(dexbrainData.api_key),
          body.name,
          body.description || `${body.riskProfile} trading agent`,
          body.riskProfile,
          body.supportedBlockchains || ['base'],
          body.supportedDexs || ['uniswap_v3', 'uniswap_v4']
        ]
      )

      const userAgent = insertResult.rows[0]

      return NextResponse.json({
        agent: {
          id: userAgent.id,
          agent_id: agentId,
          name: body.name,
          description: body.description || `${body.riskProfile} trading agent`,
          risk_profile: body.riskProfile,
          supported_blockchains: body.supportedBlockchains || ['base'],
          supported_dexs: body.supportedDexs || ['uniswap_v3', 'uniswap_v4'],
          created_at: userAgent.created_at,
          api_key: dexbrainData.api_key, // Only returned once during creation
          api_key_preview: hashApiKey(dexbrainData.api_key).substring(0, 8) + '...'
        },
        message: 'Agent created successfully'
      }, { status: 201 })

    } finally {
      client.release()
    }

  } catch (error) {
    console.error('Create agent error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// Helper function to hash API key
function hashApiKey(apiKey: string): string {
  return crypto.createHash('sha256').update(apiKey).digest('hex')
}