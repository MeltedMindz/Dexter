# Automation

## Purpose
The automation directory contains the ultra-frequent keeper service that executes automated operations for Dexter MVP contracts on Base chain. This service runs compound operations every 5 minutes and rebalancing operations every 30 minutes, optimized for Base chain's low gas costs.

## What Lives Here
- **ultra-frequent-keeper.js** - Main keeper service that monitors and executes automated operations
- **config.js** - Configuration management for network settings, contract addresses, and automation parameters
- **package.json** - Node.js dependencies and scripts for the keeper service

## How It Fits Into the System
- **Interacts with**: `contracts/mvp/` smart contracts (DexterMVP, UltraFrequentCompounder, BinRebalancer)
- **Depends on**: Base chain RPC endpoint, keeper wallet with ETH for gas
- **Provides**: Automated compound and rebalance execution for Uniswap V3 positions
- **Part of**: The overall Dexter Protocol automation infrastructure, working alongside the AI backend for position management

## Current Status
✅ **Active / In use** - Production-ready keeper service optimized for Base chain's low gas environment

## What This Is NOT
- This is not the AI decision-making system (that lives in `backend/`)
- This is not the smart contracts themselves (those are in `contracts/mvp/`)
- This is not a user-facing service (it runs as a background process)

## Relevant Docs / Entry Points
- **Main entry point**: `ultra-frequent-keeper.js`
- **Configuration**: `config.js` (uses environment variables)
- **Related contracts**: See `contracts/mvp/README.md`
- **Root documentation**: See `../README.md`

## Usage
```bash
# Install dependencies
npm install

# Start the keeper service
npm start

# Run in development mode (with auto-restart)
npm run dev

# Check service health
npm run health
```

## Environment Variables
Required environment variables (set in `.env` or environment):
- `BASE_RPC_URL` - Base chain RPC endpoint
- `KEEPER_PRIVATE_KEY` - Private key for keeper wallet
- `DEXTER_MVP_ADDRESS` - Deployed DexterMVP contract address
- `COMPOUNDER_ADDRESS` - Deployed UltraFrequentCompounder contract address
- `REBALANCER_ADDRESS` - Deployed BinRebalancer contract address

