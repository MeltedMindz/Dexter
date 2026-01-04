# Database

## Purpose
The db directory contains database schema definitions, migration scripts, and database initialization utilities for the Dexter Protocol backend. This manages the PostgreSQL database structure for storing strategies, performance metrics, datasets, and user management data.

## What Lives Here
- **schema.sql** - Core database schema (strategies, performance_metrics, datasets tables)
- **migrations/** - Database migration scripts (user management, schema updates)
- **initialize.py** - Database initialization script

## How It Fits Into the System
- **Interacts with**: All backend services that need database access (DexBrain, compound service, API server)
- **Depends on**: PostgreSQL database server
- **Provides**: Persistent storage for strategies, metrics, datasets, user accounts, and agent data
- **Part of**: The backend infrastructure, enabling data persistence across the Dexter Protocol

## Current Status
✅ **Active / In use** - Database schema defined and migrations available

## What This Is NOT
- This is not the database connection management (that's in `backend/dexbrain/db_manager.py`)
- This is not the ORM models (those may be in other backend modules)
- This is not the database server itself (external PostgreSQL instance)

## Relevant Docs / Entry Points
- **Schema**: `schema.sql` - Core table definitions
- **Migrations**: `migrations/` - Versioned schema updates
- **Initialization**: `initialize.py` - Database setup script
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Database Schema
- **strategies** - Strategy definitions and details
- **performance_metrics** - Agent and strategy performance data
- **datasets** - ML training dataset metadata
- **users** - User account management
- **user_agents** - Links users to their DexBrain agents
- **user_sessions** - Authentication session management

## Usage
```bash
# Initialize database
python db/initialize.py

# Run migrations (if migration system is set up)
# See migrations/ directory for SQL migration files
```

