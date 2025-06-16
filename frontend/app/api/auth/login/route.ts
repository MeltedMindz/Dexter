import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcrypt'
import { v4 as uuidv4 } from 'uuid'
import pg from 'pg'

const { Pool } = pg

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/dexter_db'
})

interface LoginRequest {
  email: string
  password: string
}

export async function POST(request: NextRequest) {
  try {
    const body: LoginRequest = await request.json()
    
    if (!body.email || !body.password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      )
    }

    const client = await pool.connect()
    
    try {
      // Get user by email
      const userResult = await client.query(
        `SELECT id, email, password_hash, wallet_address, is_verified, is_active, subscription_tier
         FROM users WHERE email = $1`,
        [body.email.toLowerCase()]
      )

      if (userResult.rows.length === 0) {
        return NextResponse.json(
          { error: 'Invalid credentials' },
          { status: 401 }
        )
      }

      const user = userResult.rows[0]

      // Check if user is active
      if (!user.is_active) {
        return NextResponse.json(
          { error: 'Account is deactivated' },
          { status: 401 }
        )
      }

      // Verify password
      const isValidPassword = await bcrypt.compare(body.password, user.password_hash)
      if (!isValidPassword) {
        return NextResponse.json(
          { error: 'Invalid credentials' },
          { status: 401 }
        )
      }

      // Create new session
      const sessionToken = uuidv4()
      const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours

      await client.query(
        `INSERT INTO user_sessions (user_id, session_token, expires_at, ip_address, user_agent)
         VALUES ($1, $2, $3, $4, $5)`,
        [
          user.id,
          sessionToken,
          expiresAt,
          request.headers.get('x-forwarded-for') || request.headers.get('x-real-ip') || 'unknown',
          request.headers.get('user-agent')
        ]
      )

      // Get user's agents
      const agentsResult = await client.query(
        `SELECT agent_id, name, description, risk_profile, supported_blockchains, 
                supported_dexs, is_active, created_at, last_used, request_count, data_submissions
         FROM user_agents 
         WHERE user_id = $1 AND is_active = true
         ORDER BY created_at DESC`,
        [user.id]
      )

      const agents = agentsResult.rows.map(agent => ({
        agent_id: agent.agent_id,
        name: agent.name,
        description: agent.description,
        risk_profile: agent.risk_profile,
        supported_blockchains: agent.supported_blockchains,
        supported_dexs: agent.supported_dexs,
        created_at: agent.created_at,
        last_used: agent.last_used,
        stats: {
          request_count: agent.request_count,
          data_submissions: agent.data_submissions
        }
      }))

      // Update user's last login
      await client.query(
        'UPDATE users SET updated_at = NOW() WHERE id = $1',
        [user.id]
      )

      return NextResponse.json({
        user: {
          id: user.id,
          email: user.email,
          wallet_address: user.wallet_address,
          is_verified: user.is_verified,
          subscription_tier: user.subscription_tier
        },
        session_token: sessionToken,
        agents: agents,
        message: 'Login successful'
      })

    } finally {
      client.release()
    }

  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}