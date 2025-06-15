import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcrypt'
import { v4 as uuidv4 } from 'uuid'
import pg from 'pg'

const { Pool } = pg

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/dexter_db'
})

interface RegisterRequest {
  email: string
  password: string
  walletAddress?: string
  agentName?: string
  riskProfile?: string
}

export async function POST(request: NextRequest) {
  try {
    const body: RegisterRequest = await request.json()
    
    // Validate required fields
    if (!body.email || !body.password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      )
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(body.email)) {
      return NextResponse.json(
        { error: 'Invalid email format' },
        { status: 400 }
      )
    }

    // Validate password strength
    if (body.password.length < 8) {
      return NextResponse.json(
        { error: 'Password must be at least 8 characters long' },
        { status: 400 }
      )
    }

    const client = await pool.connect()
    
    try {
      // Check if user already exists
      const existingUser = await client.query(
        'SELECT id FROM users WHERE email = $1',
        [body.email.toLowerCase()]
      )

      if (existingUser.rows.length > 0) {
        return NextResponse.json(
          { error: 'Email already registered' },
          { status: 409 }
        )
      }

      // Check if wallet address is already registered (if provided)
      if (body.walletAddress) {
        const existingWallet = await client.query(
          'SELECT id FROM users WHERE wallet_address = $1',
          [body.walletAddress.toLowerCase()]
        )

        if (existingWallet.rows.length > 0) {
          return NextResponse.json(
            { error: 'Wallet address already registered' },
            { status: 409 }
          )
        }
      }

      // Hash password
      const saltRounds = 12
      const passwordHash = await bcrypt.hash(body.password, saltRounds)

      // Generate verification token
      const verificationToken = uuidv4()

      // Create user
      const userResult = await client.query(
        `INSERT INTO users (email, password_hash, wallet_address, verification_token, created_at, updated_at)
         VALUES ($1, $2, $3, $4, NOW(), NOW())
         RETURNING id, email, wallet_address, created_at`,
        [
          body.email.toLowerCase(),
          passwordHash,
          body.walletAddress?.toLowerCase() || null,
          verificationToken
        ]
      )

      const user = userResult.rows[0]

      // Create session token
      const sessionToken = uuidv4()
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours

      await client.query(
        `INSERT INTO user_sessions (user_id, session_token, expires_at, ip_address, user_agent)
         VALUES ($1, $2, $3, $4, $5)`,
        [
          user.id,
          sessionToken,
          expiresAt,
          request.ip || request.headers.get('x-forwarded-for'),
          request.headers.get('user-agent')
        ]
      )

      // If agent name provided, create default agent
      let agentData = null
      if (body.agentName) {
        const agentId = `user_${user.id}_${body.agentName.toLowerCase().replace(/[^a-z0-9]/g, '_')}`
        
        // Generate API key by calling DexBrain registration
        const dexbrainResponse = await fetch(`${process.env.DEXBRAIN_API_URL || 'http://localhost:8080'}/api/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            agent_id: agentId,
            metadata: {
              user_id: user.id,
              name: body.agentName,
              risk_profile: body.riskProfile || 'conservative',
              supported_blockchains: ['base'],
              supported_dexs: ['uniswap_v3', 'uniswap_v4'],
              registration_type: 'frontend_user'
            }
          })
        })

        if (dexbrainResponse.ok) {
          const dexbrainData = await dexbrainResponse.json()
          
          // Store user-agent relationship
          await client.query(
            `INSERT INTO user_agents (user_id, agent_id, api_key_hash, name, description, risk_profile, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
            [
              user.id,
              agentId,
              hashApiKey(dexbrainData.api_key),
              body.agentName,
              `${body.riskProfile || 'Conservative'} trading agent`,
              body.riskProfile || 'conservative'
            ]
          )

          agentData = {
            agent_id: agentId,
            api_key: dexbrainData.api_key,
            name: body.agentName,
            risk_profile: body.riskProfile || 'conservative'
          }
        }
      }

      // TODO: Send verification email in production
      // await sendVerificationEmail(user.email, verificationToken)

      return NextResponse.json({
        user: {
          id: user.id,
          email: user.email,
          wallet_address: user.wallet_address,
          created_at: user.created_at
        },
        session_token: sessionToken,
        agent: agentData,
        message: 'User registered successfully'
      }, { status: 201 })

    } finally {
      client.release()
    }

  } catch (error) {
    console.error('Registration error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// Helper function to hash API key for storage
function hashApiKey(apiKey: string): string {
  const crypto = require('crypto')
  return crypto.createHash('sha256').update(apiKey).digest('hex')
}