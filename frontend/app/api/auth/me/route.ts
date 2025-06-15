import { NextRequest, NextResponse } from 'next/server'
import pg from 'pg'

const { Pool } = pg

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/dexter_db'
})

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json(
        { error: 'Missing or invalid authorization header' },
        { status: 401 }
      )
    }

    const sessionToken = authHeader.substring(7)
    const client = await pool.connect()
    
    try {
      // Verify session and get user
      const sessionResult = await client.query(
        `SELECT us.user_id, us.expires_at, u.id, u.email, u.wallet_address, 
                u.is_verified, u.subscription_tier, u.is_active
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

      const user = sessionResult.rows[0]

      if (!user.is_active) {
        return NextResponse.json(
          { error: 'Account is deactivated' },
          { status: 401 }
        )
      }

      // Get user's agents with recent activity
      const agentsResult = await client.query(
        `SELECT ua.agent_id, ua.name, ua.description, ua.risk_profile, 
                ua.supported_blockchains, ua.supported_dexs, ua.is_active, 
                ua.created_at, ua.last_used, ua.request_count, ua.data_submissions,
                ua.api_key_hash
         FROM user_agents ua
         WHERE ua.user_id = $1 AND ua.is_active = true
         ORDER BY ua.last_used DESC NULLS LAST, ua.created_at DESC`,
        [user.id]
      )

      // Get recent API usage statistics
      const usageResult = await client.query(
        `SELECT agent_id, endpoint, SUM(request_count) as total_requests
         FROM user_api_usage 
         WHERE user_id = $1 AND date_bucket >= CURRENT_DATE - INTERVAL '30 days'
         GROUP BY agent_id, endpoint
         ORDER BY total_requests DESC`,
        [user.id]
      )

      const agents = agentsResult.rows.map(agent => {
        const agentUsage = usageResult.rows
          .filter(usage => usage.agent_id === agent.agent_id)
          .reduce((acc, usage) => {
            acc[usage.endpoint] = parseInt(usage.total_requests)
            return acc
          }, {} as Record<string, number>)

        return {
          agent_id: agent.agent_id,
          name: agent.name,
          description: agent.description,
          risk_profile: agent.risk_profile,
          supported_blockchains: agent.supported_blockchains,
          supported_dexs: agent.supported_dexs,
          created_at: agent.created_at,
          last_used: agent.last_used,
          api_key_preview: agent.api_key_hash.substring(0, 8) + '...',
          stats: {
            request_count: agent.request_count,
            data_submissions: agent.data_submissions,
            usage_30d: agentUsage
          }
        }
      })

      return NextResponse.json({
        user: {
          id: user.id,
          email: user.email,
          wallet_address: user.wallet_address,
          is_verified: user.is_verified,
          subscription_tier: user.subscription_tier
        },
        agents: agents,
        session_info: {
          expires_at: user.expires_at
        }
      })

    } finally {
      client.release()
    }

  } catch (error) {
    console.error('Session verification error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}