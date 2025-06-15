-- Migration: Add user management tables for API key system
-- This extends the existing schema to support user registration and multi-agent management

-- Users table for account management
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    wallet_address VARCHAR(42) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_verified BOOLEAN DEFAULT FALSE,
    verification_token VARCHAR(255),
    reset_token VARCHAR(255),
    subscription_tier VARCHAR(20) DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE
);

-- User agents table - links users to their DexBrain agents
CREATE TABLE IF NOT EXISTS user_agents (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(64) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    risk_profile VARCHAR(20) DEFAULT 'conservative',
    supported_blockchains TEXT[] DEFAULT ARRAY['base'],
    supported_dexs TEXT[] DEFAULT ARRAY['uniswap_v3'],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP,
    request_count INTEGER DEFAULT 0,
    data_submissions INTEGER DEFAULT 0,
    UNIQUE(user_id, agent_id)
);

-- User sessions for authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- API usage tracking per user
CREATE TABLE IF NOT EXISTS user_api_usage (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 1,
    last_request TIMESTAMP DEFAULT NOW(),
    date_bucket DATE DEFAULT CURRENT_DATE,
    UNIQUE(user_id, agent_id, endpoint, date_bucket)
);

-- Email verification tokens
CREATE TABLE IF NOT EXISTS email_verifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    verified_at TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_wallet ON users(wallet_address);
CREATE INDEX IF NOT EXISTS idx_user_agents_user_id ON user_agents(user_id);
CREATE INDEX IF NOT EXISTS idx_user_agents_agent_id ON user_agents(agent_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_api_usage_user_id ON user_api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_email_verifications_token ON email_verifications(token);

-- Insert default admin user (password: 'admin123' - CHANGE IN PRODUCTION!)
INSERT INTO users (email, password_hash, is_verified, subscription_tier) 
VALUES (
    'admin@dexteragent.com', 
    '$2b$12$LQv3c1yqBw2hnHnEjTJ0Ouqn.nYrH0K7XA5o8Q9vP5sKe4Z9xF2nC', -- bcrypt hash of 'admin123'
    TRUE, 
    'admin'
) ON CONFLICT (email) DO NOTHING;