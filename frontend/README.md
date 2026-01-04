# Frontend

## Purpose
The frontend directory contains React/Next.js components and pages for the Dexter MVP user interface. This provides the web interface for users to create and manage automated liquidity positions on Uniswap V3.

## What Lives Here
- **components/** - Reusable React components (PoolSelector, PositionCreator, TokenSelector, UI components)
- **pages/** - Next.js page components (create-position page)
- **lib/** - Utility libraries (cache management)
- **mvp/** - MVP-specific dashboard components

## How It Fits Into the System
- **Interacts with**: Smart contracts in `contracts/mvp/` via Web3 (Wagmi/Viem)
- **Depends on**: Backend API services in `backend/` for data and analytics
- **Provides**: User interface for position creation, monitoring, and management
- **Part of**: The complete Dexter Protocol stack, connecting users to the automated liquidity management system

## Current Status
🚧 **In development** - Core components exist but full integration with contracts and backend is in progress

## What This Is NOT
- This is not the main website repository (see [dexter-website](https://github.com/MeltedMindz/dexter-website) for the production site)
- This is not the backend API (that's in `backend/`)
- This is not the smart contracts (those are in `contracts/`)

## Relevant Docs / Entry Points
- **Main components**: `components/PositionCreator.tsx`, `components/PoolSelector.tsx`
- **Pages**: `pages/create-position.tsx`
- **MVP Dashboard**: `mvp/MVPDashboard.tsx`
- **Root documentation**: See `../README.md`
- **Production website**: See separate [dexter-website](https://github.com/MeltedMindz/dexter-website) repository

## Technology Stack
- **Framework**: Next.js with TypeScript
- **Web3**: Wagmi + Viem for blockchain interactions
- **UI**: Custom components with Tailwind CSS
- **State Management**: React hooks and context

