# DexterMVP Keeper Specification

**Version:** 1.0.0
**Date:** 2026-01-19
**Status:** Production

---

## 1. Overview

Keepers are authorized off-chain services that execute time-sensitive operations on DexterMVP contracts. They handle:

- **Auto-compounding**: Collecting accumulated fees and reinvesting into positions
- **Rebalancing**: Adjusting position ranges when price moves outside optimal bounds
- **Batch operations**: Processing multiple positions efficiently

### Contract Addresses (Base Network)

| Contract | Function |
|----------|----------|
| DexterMVP | Primary position management, compound + rebalance |
| BinRebalancer | Advanced rebalancing with discrete tick bins |
| UltraFrequentCompounder | High-frequency compounding with profitability checks |

---

## 2. Authorization

### Setting Up Keepers

Only the contract owner can authorize keepers:

```solidity
// Authorize a keeper
await dexterMVP.setKeeperAuthorization(keeperAddress, true);

// Revoke authorization
await dexterMVP.setKeeperAuthorization(keeperAddress, false);
```

### Authorization Model

- `authorizedKeepers` mapping: `mapping(address => bool)`
- Owner address always has keeper privileges
- Each contract maintains its own keeper whitelist
- No time-based or rate-limited authorization

### Security Considerations

1. **Private key management**: Use hardware wallets or HSMs for keeper keys
2. **Separate authorization**: Keep keeper keys separate from owner keys
3. **Multiple keepers**: Consider authorizing backup keepers for redundancy
4. **Monitor events**: Track `KeeperAuthorized` events for unauthorized changes

---

## 3. Keeper Functions

### 3.1 DexterMVP

#### `executeCompound(uint256 tokenId)`

Compounds accumulated fees back into the position.

```solidity
function executeCompound(uint256 tokenId)
    external
    nonReentrant
    onlyAuthorizedKeeper
    validPosition(tokenId)
    whenNotPaused
```

**Pre-conditions:**
- Position exists and is registered
- Contract is not paused
- Caller is authorized keeper

**Logic:**
1. Collects fees from Uniswap position
2. Validates against TWAP (if enabled)
3. Calculates optimal liquidity addition
4. Adds liquidity back to position
5. Emits `FeesCompounded` event

**Gas estimate:** ~200,000 - 350,000 gas

---

#### `batchCompound(uint256[] calldata tokenIds)`

Compounds multiple positions in a single transaction.

```solidity
function batchCompound(uint256[] calldata tokenIds)
    external
    onlyAuthorizedKeeper
    whenNotPaused
```

**Pre-conditions:**
- All positions exist
- Maximum 50 positions per batch

**Logic:**
1. Iterates through tokenIds
2. Calls `executeCompound` internally (try/catch)
3. Continues on individual failures
4. Returns successful count

**Gas estimate:** ~(200,000 + 150,000 * n) gas where n = number of positions

---

#### `executeRebalance(uint256 tokenId)`

Rebalances a position to a new price range.

```solidity
function executeRebalance(uint256 tokenId)
    external
    nonReentrant
    onlyAuthorizedKeeper
    validPosition(tokenId)
    whenNotPaused
```

**Pre-conditions:**
- Position registered and enabled for rebalancing
- Current price outside position range OR meets rebalance criteria
- Contract is not paused

**Logic:**
1. Validates TWAP protection (if enabled)
2. Calls `_closePosition` to withdraw liquidity
3. Calculates new tick range
4. Calls `_openNewPosition` with new range
5. Emits `PositionRebalanced` event

**Gas estimate:** ~400,000 - 600,000 gas

---

### 3.2 BinRebalancer

#### `executeRebalance(uint256 tokenId)`

Advanced rebalancing with discrete tick bins.

```solidity
function executeRebalance(uint256 tokenId)
    external
    onlyKeeper
    nonReentrant
```

**Features:**
- Uses tick bins for discrete range boundaries
- Tracks total rebalance gas costs per position
- Supports both upward and downward rebalancing

---

### 3.3 UltraFrequentCompounder

#### `executeCompound(uint256 tokenId)`

Single position compound with profitability check.

```solidity
function executeCompound(uint256 tokenId)
    external
    onlyKeeper
    nonReentrant
```

**Features:**
- Minimum fee threshold check (`minFeeThreshold`)
- Gas cost vs. yield comparison
- High-frequency operation support

---

#### `batchCompound(uint256[] calldata tokenIds)`

Standard batch compound operation.

---

#### `smartBatchCompound(uint256[] calldata tokenIds)`

Intelligent batch compound that skips unprofitable positions.

```solidity
function smartBatchCompound(uint256[] calldata tokenIds)
    external
    onlyKeeper
```

**Features:**
- Checks each position for profitability before compounding
- Skips positions with fees below threshold
- Returns count of successfully compounded positions

---

## 4. Trigger Conditions

### When to Compound

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| Fee accumulation | > $5-10 USD equivalent | Cover gas costs |
| Time elapsed | > 24 hours since last compound | Maximize yield |
| Price volatility | Low volatility period | Minimize slippage |
| Gas price | < 30 gwei (Base) | Cost optimization |

### When to Rebalance

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| Price out of range | Current tick outside position bounds | Position earning 0 fees |
| Range efficiency | < 50% of liquidity in active range | Suboptimal capital efficiency |
| Volatility spike | > 2 standard deviations | Protect against IL |
| Manual trigger | User requests rebalance | User preference |

### Recommended Monitoring Queries

```javascript
// Check if position needs compound
const position = await nfpm.positions(tokenId);
const fees = await nfpm.collect.staticCall({
    tokenId,
    recipient: address(0),
    amount0Max: MaxUint128,
    amount1Max: MaxUint128
});

const needsCompound = fees.amount0 > minThreshold0 ||
                      fees.amount1 > minThreshold1;

// Check if position needs rebalance
const pool = await IPool.attach(poolAddress);
const slot0 = await pool.slot0();
const currentTick = slot0.tick;
const needsRebalance = currentTick < position.tickLower ||
                       currentTick >= position.tickUpper;
```

---

## 5. Gas Optimization

### Base Network Gas Costs (Estimates)

| Operation | Gas Units | Cost at 0.001 gwei | Cost at 0.01 gwei |
|-----------|-----------|---------------------|-------------------|
| executeCompound | 250,000 | ~$0.00025 | ~$0.0025 |
| executeRebalance | 500,000 | ~$0.0005 | ~$0.005 |
| batchCompound (10) | 1,500,000 | ~$0.0015 | ~$0.015 |

### Optimization Strategies

1. **Batch operations**: Group multiple positions in single transaction
2. **Off-peak execution**: Execute during low gas periods
3. **Profitability checks**: Skip positions with fees < gas cost
4. **Priority ordering**: Process highest-value positions first

---

## 6. Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `Not authorized keeper` | Caller not whitelisted | Get owner to authorize |
| `Position not found` | Invalid tokenId | Check position exists |
| `Contract paused` | Emergency pause active | Wait for unpause |
| `TWAP validation failed` | Price manipulation detected | Wait for price stabilization |
| `Slippage too high` | Market conditions unfavorable | Retry later or adjust slippage |

### Retry Logic

```javascript
async function executeWithRetry(fn, maxRetries = 3, delay = 5000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (error.message.includes("TWAP validation")) {
                await sleep(delay * 2); // Longer wait for TWAP
            } else if (error.message.includes("paused")) {
                throw error; // Don't retry paused contracts
            } else {
                await sleep(delay);
            }
        }
    }
    throw new Error("Max retries exceeded");
}
```

---

## 7. Monitoring & Alerts

### Events to Monitor

```solidity
// DexterMVP
event FeesCompounded(uint256 indexed tokenId, uint256 amount0, uint256 amount1);
event PositionRebalanced(uint256 indexed oldTokenId, uint256 indexed newTokenId, int24 tickLower, int24 tickUpper);
event KeeperAuthorized(address indexed keeper, bool authorized);
event EmergencyPause(address indexed by);

// UltraFrequentCompounder
event CompoundExecuted(uint256 indexed tokenId, uint256 amount0Added, uint256 amount1Added);
event BatchCompoundCompleted(uint256 totalCompounded, uint256 totalSkipped);
```

### Recommended Alerts

| Condition | Severity | Action |
|-----------|----------|--------|
| Contract paused | CRITICAL | Notify immediately |
| Keeper authorization changed | HIGH | Verify legitimate |
| Multiple compound failures | MEDIUM | Investigate cause |
| Gas price spike | LOW | Pause operations |
| Position out of range > 24h | MEDIUM | Prioritize rebalance |

---

## 8. Example Keeper Implementation

### Minimal Keeper Service

```javascript
const { ethers } = require("ethers");

class DexterKeeper {
    constructor(provider, signer, contractAddress) {
        this.contract = new ethers.Contract(
            contractAddress,
            DexterMVPABI,
            signer
        );
        this.provider = provider;
    }

    async checkAndCompound(tokenId) {
        // 1. Check fees
        const fees = await this.getAccumulatedFees(tokenId);
        if (fees.total < this.minThreshold) return;

        // 2. Check gas
        const gasPrice = await this.provider.getGasPrice();
        if (gasPrice > this.maxGasPrice) return;

        // 3. Execute
        const tx = await this.contract.executeCompound(tokenId);
        await tx.wait();

        console.log(`Compounded position ${tokenId}: ${tx.hash}`);
    }

    async checkAndRebalance(tokenId) {
        // 1. Check if out of range
        const needsRebalance = await this.isOutOfRange(tokenId);
        if (!needsRebalance) return;

        // 2. Execute
        const tx = await this.contract.executeRebalance(tokenId);
        await tx.wait();

        console.log(`Rebalanced position ${tokenId}: ${tx.hash}`);
    }

    async runLoop(positions, interval = 60000) {
        while (true) {
            for (const tokenId of positions) {
                try {
                    await this.checkAndCompound(tokenId);
                    await this.checkAndRebalance(tokenId);
                } catch (error) {
                    console.error(`Error on ${tokenId}:`, error.message);
                }
            }
            await sleep(interval);
        }
    }
}
```

---

## 9. Deployment Checklist

Before activating keepers:

- [ ] Deploy contracts to network
- [ ] Verify contracts on block explorer
- [ ] Authorize keeper addresses
- [ ] Configure TWAP settings (if using)
- [ ] Test with small position first
- [ ] Set up monitoring and alerts
- [ ] Document keeper wallet addresses
- [ ] Establish key rotation schedule

---

## 10. References

- [DexterMVP Contract](../contracts/DexterMVP.sol)
- [BinRebalancer Contract](../contracts/BinRebalancer.sol)
- [UltraFrequentCompounder Contract](../contracts/UltraFrequentCompounder.sol)
- [Deployment Script](../scripts/deploy.js)
- [Keeper Setup Script](../scripts/setup-keeper.js)
