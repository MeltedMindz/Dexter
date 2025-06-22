# Dexter Protocol - Fundraising Materials

This folder contains materials for Dexter Protocol's fundraising efforts.

## Contents

- `INVESTOR_PITCH_DECK.md` - Complete 12-slide investor pitch deck with market analysis
- `.gitignore` - Protects sensitive fundraising materials from being committed

## Key Updates

### Accurate Fee Structure
The pitch deck now reflects the actual tiered fee structure implemented in `contracts/fees/FeeManager.sol`:

| Tier | Min Balance | Performance Fee | Management Fee | AI Fee |
|------|-------------|-----------------|----------------|---------|
| **Retail** | $0 | 15% | 1% | 0.5% |
| **Premium** | $100k | 12.5% | 0.75% | 0.4% |
| **Institutional** | $1M | 10% | 0.5% | 0.25% |
| **VIP** | $10M | 7.5% | 0.25% | 0.15% |

### Positioning Update
Changed from "AI-Powered Vault Infrastructure for Institutional DeFi" to **"Retail and Institutional DeFi"** to reflect broader market appeal.

## Usage

1. **Copy sections** from `INVESTOR_PITCH_DECK.md` into presentation software (Canva, PowerPoint, etc.)
2. **Customize visuals** - the content is structured for easy transfer to slide format
3. **Adapt for audience** - emphasize different aspects for different investor types

## Security Note

The `.gitignore` file protects sensitive materials like:
- Financial models with proprietary data
- Due diligence materials
- Investor-specific documents
- Legal agreements

Only the template pitch deck is committed to the repository.