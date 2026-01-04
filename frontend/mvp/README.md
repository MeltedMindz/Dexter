# MVP

## Purpose
The mvp directory contains MVP-specific dashboard and interface components for the Dexter Protocol MVP implementation.

## What Lives Here
- **MVPDashboard.tsx** - Main dashboard component for viewing and managing MVP positions

## How It Fits Into the System
- **Interacts with**: Smart contracts in `contracts/mvp/`, backend services
- **Depends on**: React, Web3 (Wagmi/Viem), UI components
- **Provides**: MVP-specific user interface for position management
- **Part of**: The frontend application, specifically for the MVP implementation

## Current Status
🚧 **In development** - MVP dashboard component exists, full integration in progress

## What This Is NOT
- This is not the general frontend components (those are in `../components/`)
- This is not the production website (see separate dexter-website repository)

## Relevant Docs / Entry Points
- **MVP Dashboard**: `MVPDashboard.tsx`
- **Contract documentation**: See `../../contracts/mvp/README.md`
- **Frontend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

