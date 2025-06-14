# Dexter Protocol Token Flow

## 🔄 Complete Token Flow Diagram

```mermaid
graph TB
    subgraph "1. Users Provide Liquidity"
        A[User] -->|Deposits NFT| B[Liquidity Manager]
        B -->|Manages| C[ETH/USDC Position]
        B -->|Manages| D[WBTC/ETH Position]
        B -->|Manages| E[Any Uni V3 Position]
    end
    
    subgraph "2. Auto-Compound & Fee Collection"
        C -->|Generates Fees| F[Collect Fees]
        D -->|Generates Fees| F
        E -->|Generates Fees| F
        F -->|Auto-compound 92%| C
        F -->|Auto-compound 92%| D
        F -->|Auto-compound 92%| E
        F -->|8% Performance Fee| G[Fee Collector]
    end
    
    subgraph "3. Fee Conversion"
        G -->|Various Tokens| H[Fee Dispenser]
        H -->|Swap All| I[Convert to WETH]
        I -->|Accumulate| J[WETH Pool]
    end
    
    subgraph "4. Distribution to Stakers"
        J -->|Threshold Met| K[Distribute WETH]
        K -->|Pro-rata| L[$DEX Stakers]
        M[DEX Token] -->|Stake| L
        L -->|Claim| N[WETH Rewards]
    end
    
    style A fill:#e1f5e1
    style N fill:#ffe1e1
    style M fill:#fff4e1
```

## 📊 Key Components

### 1. **Liquidity Positions** (Any Uniswap V3 Pair)
- Users can deposit ANY Uni V3 position NFT
- Examples: ETH/USDC, WBTC/ETH, PEPE/ETH, etc.
- Dexter manages these positions automatically

### 2. **Fee Collection & Auto-Compound**
- Collects trading fees from positions
- Auto-compounds 92% back into positions
- Takes 8% performance fee on profits only

### 3. **Fee Conversion to WETH**
- All collected fees (various tokens) sent to Fee Dispenser
- Automatically swapped to WETH via Uniswap
- WETH accumulates until distribution threshold

### 4. **$DEX Token Staking**
- Users stake $DEX tokens to earn protocol revenue
- WETH distributed pro-rata based on stake %
- Example: If you stake 10% of all $DEX, you get 10% of WETH rewards

## 💰 Example Scenario

1. **Alice deposits** ETH/USDC position worth $100,000
2. **Position earns** $1,000 in trading fees (mix of ETH and USDC)
3. **Auto-compound**: $920 reinvested into position
4. **Protocol fee**: $80 (8% of profit) sent to Fee Dispenser
5. **Conversion**: $80 worth of ETH/USDC converted to WETH
6. **Distribution**: When threshold hit, WETH sent to $DEX stakers

## 🎯 Benefits

### For Liquidity Providers:
- Manage ANY token pair, not just $DEX pairs
- Auto-compounding maximizes returns
- Only pay fees on actual profits

### For $DEX Stakers:
- Earn WETH (blue-chip asset) not random tokens
- Revenue from ALL positions in protocol
- Sustainable yield from real protocol usage

## 📈 Growth Flywheel

```
More LPs → More Positions → More Fees → More WETH to Stakers → 
Higher $DEX Value → More Attention → More LPs
```