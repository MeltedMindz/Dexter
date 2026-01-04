# Components

## Purpose
The components directory contains reusable React components for the Dexter MVP frontend interface, including UI components, form elements, and DeFi-specific components for position management.

## What Lives Here
- **PositionCreator.tsx** - Main component for creating and configuring liquidity positions
- **PoolSelector.tsx** - Component for selecting Uniswap V3 pools
- **TokenSelector.tsx** - Component for selecting tokens
- **ui/** - Reusable UI components (scroll-area, buttons, cards, etc.)

## How It Fits Into the System
- **Interacts with**: Pages (`../pages/`), smart contracts via Web3 (Wagmi/Viem)
- **Depends on**: React, Next.js, Tailwind CSS, Wagmi, Viem
- **Provides**: Reusable UI components for the Dexter MVP interface
- **Part of**: The frontend application, enabling user interaction with the protocol

## Current Status
🚧 **In development** - Core components exist, full integration in progress

## What This Is NOT
- This is not the page components (those are in `../pages/`)
- This is not the MVP dashboard (that's in `../mvp/`)
- This is not the utility libraries (those are in `../lib/`)

## Relevant Docs / Entry Points
- **Frontend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

