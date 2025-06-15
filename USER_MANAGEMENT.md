# DexBrain User Management System

## Overview

Complete user registration and API key management system that integrates with the existing DexBrain API infrastructure.

## Architecture

### Database Schema

New tables added to extend existing PostgreSQL database:

- **`users`** - User accounts with email/password authentication
- **`user_agents`** - Links users to their DexBrain agents 
- **`user_sessions`** - Session management for web authentication
- **`user_api_usage`** - API usage tracking per user/agent
- **`email_verifications`** - Email verification tokens

### API Endpoints

#### Authentication
- `POST /api/auth/register` - Create new user account
- `POST /api/auth/login` - User login with session creation
- `GET /api/auth/me` - Get current user profile and agents
- `POST /api/auth/create-agent` - Create new agent for authenticated user

#### Features
- **Password hashing** with bcrypt (12 rounds)
- **Session management** with UUID tokens (24-hour expiry)
- **Multi-agent support** per user (3 free, 10 premium)
- **Automatic DexBrain integration** - agents registered with existing API
- **Wallet address linking** for Web3 users

### Frontend Pages

#### Registration (`/register`)
- Email/password account creation
- Optional agent creation during signup
- Risk profile selection (Conservative/Aggressive/Hyper)
- Wallet address integration
- Neo-brutalism design matching site aesthetic

#### Login (`/login`)
- Email/password authentication
- Session token storage
- Automatic dashboard redirect
- User agent loading

#### Dashboard Enhancement
- Multi-agent management interface
- Performance tracking per agent
- API key management
- Usage statistics

## Integration Flow

### 1. User Registration
```
Frontend (/register) → API (/api/auth/register) → Database (users table) 
                                               → DexBrain API (/api/register)
                                               → Database (user_agents table)
```

### 2. Agent Creation
```
Dashboard → API (/api/auth/create-agent) → DexBrain API (/api/register)
                                        → Database (user_agents table)
                                        → Return API key (one-time)
```

### 3. API Usage
```
External Agent → DexBrain API (with API key) → Usage tracking
                                             → Database (user_api_usage table)
```

## Environment Variables

Add to `.env` files:

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/dexter_db

# DexBrain API
DEXBRAIN_API_URL=http://localhost:8080

# Authentication
JWT_SECRET=your-super-secret-jwt-key
SESSION_TIMEOUT=86400

# Features
MAX_AGENTS_FREE=3
MAX_AGENTS_PREMIUM=10
```

## Migration Steps

### 1. Database Setup
```bash
# Run migration
psql $DATABASE_URL -f backend/db/migrations/001_create_user_management.sql
```

### 2. Install Dependencies
```bash
cd frontend
npm install bcrypt uuid pg @types/bcrypt @types/uuid @types/pg
```

### 3. Environment Configuration
- Add database connection string
- Set DexBrain API URL
- Configure JWT secret

### 4. Deploy
- Update docker-compose.yml with new environment variables
- Restart services

## User Journey

### New User
1. **Visit `/register`** - Create account with email/password
2. **Optional agent creation** - Create first agent during signup
3. **Receive API key** - Get DexBrain API key for immediate use
4. **Dashboard access** - View agent performance and create more agents

### Existing User
1. **Visit `/login`** - Login with credentials
2. **Dashboard** - View all agents and performance metrics
3. **Create agents** - Add up to subscription limit
4. **Monitor usage** - Track API usage and performance

### Developer Integration
1. **Register account** via web interface
2. **Create agents** for different strategies/risk profiles
3. **Get API keys** for each agent
4. **Integrate** with existing DexBrain API endpoints
5. **Monitor** performance through dashboard

## Security Features

- **Password hashing** with bcrypt (salt rounds: 12)
- **Session tokens** with expiration (24 hours)
- **API key hashing** for secure storage
- **Rate limiting** inheritance from DexBrain system
- **Input validation** on all endpoints
- **SQL injection protection** with parameterized queries

## Subscription Tiers

### Free Tier
- 3 agents maximum
- Standard rate limits
- Basic dashboard access

### Premium Tier (Future)
- 10 agents maximum  
- Higher rate limits
- Advanced analytics
- Priority support

## Backward Compatibility

- **Existing agents** continue to work unchanged
- **Legacy API registrations** still supported
- **File-based storage** maintained alongside database
- **API endpoints** unchanged for existing users

## Monitoring & Analytics

### User Metrics
- Total registered users
- Active users (last 30 days)
- Agent creation rate
- API usage per user

### Agent Metrics  
- Agents per user distribution
- Risk profile distribution
- Performance by subscription tier
- Usage patterns

## Support Features

### Dashboard
- Agent performance visualization
- API usage statistics
- Key management (view/regenerate)
- Account settings

### Admin Features (Future)
- User management interface
- Usage analytics
- Subscription management
- Support ticket system

---

**Result**: Complete user management system that seamlessly integrates with existing DexBrain infrastructure while providing modern web-based registration and multi-agent management capabilities.