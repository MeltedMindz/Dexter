# RISK_REGISTER.md - The Ralph Collective Phase 1 Audit

**Date:** 2026-01-19
**Auditor:** The Ralph Collective
**Scope:** Operational, security, and design risk analysis

---

## Risk Severity Definitions

| Severity | Impact | Likelihood | Action Required |
|----------|--------|------------|-----------------|
| CRITICAL | System failure, fund loss, legal liability | High | Must fix before any deployment |
| HIGH | Major functionality broken, security exposure | Medium-High | Must fix before production |
| MEDIUM | Degraded experience, maintainability issues | Medium | Should fix soon |
| LOW | Minor issues, cosmetic problems | Low | Fix when convenient |

---

## 1. CRITICAL RISKS

### RISK-001: Smart Contract Functions Return Placeholder Values

**Category:** Security / Operational
**Severity:** CRITICAL
**Status:** ACTIVE

**Description:**
Core smart contract functions return hardcoded placeholder values instead of actual calculations.

**Evidence:**
```solidity
// DexterMVP.sol:400-404
function _getUnclaimedFeesUSD(uint256 tokenId) internal view returns (uint256) {
    // Simplified fee calculation - would integrate with price oracle in production
    // For MVP, using basic estimation
    return 1e18; // Placeholder $1 USD
}

// DexterMVP.sol:411-414
function _closePosition(uint256 tokenId) internal returns (uint256 amount0, uint256 amount1) {
    // Simplified position closure - full implementation would handle liquidity removal
    return (0, 0);  // RETURNS ZERO
}
```

**Impact:**
- `shouldCompound()` always returns same result regardless of actual fees
- `executeRebalance()` loses all liquidity (returns 0, 0)
- Users could lose funds if deployed as-is

**Likelihood:** HIGH if deployed
**Fixable:** YES

**Mitigation:**
1. DO NOT deploy contracts until functions implemented
2. Add oracle integration for fee calculation
3. Implement proper liquidity withdrawal
4. Add comprehensive tests for edge cases

---

### RISK-002: ML Models Train on Simulated Random Data

**Category:** Operational / Misleading Claims
**Severity:** CRITICAL
**Status:** ACTIVE

**Description:**
The ML training pipeline generates random data instead of using real blockchain data, making any model outputs meaningless.

**Evidence:**
```python
# continuous_training_orchestrator.py:325-338
async def _load_training_data(self) -> pd.DataFrame:
    n_samples = 10000
    data = {
        'price': np.random.normal(3000, 300, n_samples),  # RANDOM
        'volume_1h': np.random.exponential(100000, n_samples),  # RANDOM
        'strategy_action': np.random.choice(['hold', 'compound', 'rebalance'], n_samples),  # RANDOM
    }
```

**Impact:**
- Models trained on random data have no predictive value
- Published accuracy metrics are meaningless
- Users relying on predictions could make poor decisions
- Claims of "production ML" are misleading

**Likelihood:** HIGH (already happening)
**Fixable:** YES

**Mitigation:**
1. Connect to real data sources (Alchemy API exists in code)
2. Implement proper data pipeline from Kafka
3. Validate model accuracy on real held-out data
4. Update claims to reflect actual state

---

### RISK-003: No Position Limit Enforcement

**Category:** Security / DoS
**Severity:** CRITICAL
**Status:** ACTIVE

**Description:**
`MAX_POSITIONS_PER_ADDRESS = 200` is declared but never enforced in deposit logic.

**Evidence:**
```solidity
// DexterMVP.sol:32
uint32 constant public MAX_POSITIONS_PER_ADDRESS = 200;

// DexterMVP.sol:134-152 - depositPosition()
// NO CHECK against MAX_POSITIONS_PER_ADDRESS
function depositPosition(uint256 tokenId, AutomationSettings memory settings) external nonReentrant {
    // Transfer NFT to this contract
    nonfungiblePositionManager.safeTransferFrom(msg.sender, address(this), tokenId);
    // ... NO LIMIT CHECK
}
```

**Impact:**
- Users can deposit unlimited positions
- Could lead to gas limit issues in batch operations
- Potential DoS vector for keeper functions

**Likelihood:** MEDIUM
**Fixable:** YES

**Mitigation:**
```solidity
function depositPosition(...) external nonReentrant {
    require(accountTokens[msg.sender].length < MAX_POSITIONS_PER_ADDRESS, "Position limit exceeded");
    // ...
}
```

---

### RISK-004: API Endpoints Return Mock Data

**Category:** Operational / Misleading
**Severity:** CRITICAL
**Status:** ACTIVE

**Description:**
Several API endpoints return hardcoded mock data, making the API unreliable for real use.

**Evidence:**
```python
# api_server.py:321-345
pool_data = {
    'current_tick': 100000,  # HARDCODED
    'current_price': 3000,  # HARDCODED
    'liquidity': 1000000,  # HARDCODED
    ...
}

vault_metrics = {
    'total_value_locked': 2000000,  # HARDCODED
    'apr': 0.15,  # HARDCODED
    ...
}
```

**Impact:**
- Clients receive false data
- Automation based on API will malfunction
- User trust destroyed when discovered

**Likelihood:** HIGH if API used
**Fixable:** YES

**Mitigation:**
1. Connect to real data sources
2. Return explicit errors when data unavailable
3. Add "mock: true" flag to responses during development

---

## 2. HIGH RISKS

### RISK-005: No TWAP Protection Integration

**Category:** Security / MEV
**Severity:** HIGH
**Status:** ACTIVE

**Description:**
TWAP oracle code exists but is not integrated into compound/rebalance operations, leaving positions vulnerable to MEV attacks.

**Evidence:**
- `contracts/libraries/TWAPOracle.sol` exists
- `contracts/oracles/MLValidationOracle.sol` exists
- Neither imported or used in DexterMVP.sol
- No price validation before swaps

**Impact:**
- Sandwich attacks during rebalancing
- Price manipulation during compounds
- Loss of user funds to MEV bots

**Likelihood:** HIGH if deployed on mainnet
**Fixable:** YES

**Mitigation:**
1. Integrate TWAPOracle into DexterMVP
2. Add slippage protection to all swaps
3. Add MEV protection patterns
4. Test with MEV simulation

---

### RISK-006: Keeper Authorization Without Revocation Tracking

**Category:** Security / Access Control
**Severity:** HIGH
**Status:** ACTIVE

**Description:**
Keepers can be authorized/revoked, but no event history or emergency pause exists.

**Evidence:**
```solidity
// Only basic authorization
mapping(address => bool) public authorizedKeepers;

function setKeeperAuthorization(address keeper, bool authorized) external onlyOwner {
    authorizedKeepers[keeper] = authorized;
    emit KeeperAuthorized(keeper, authorized);
}
```

**Missing:**
- No time-locked authorization
- No multi-sig requirement for keepers
- No emergency pause
- No keeper action limits

**Impact:**
- Compromised keeper can drain positions
- No way to quickly freeze malicious activity
- Single owner can add malicious keepers

**Likelihood:** MEDIUM
**Fixable:** YES

**Mitigation:**
1. Add timelock for keeper changes
2. Implement emergency pause
3. Add per-keeper action limits
4. Consider multi-sig for owner

---

### RISK-007: No Database Migration Strategy

**Category:** Operational
**Severity:** HIGH
**Status:** ACTIVE

**Description:**
Database schema exists but no migration strategy or versioning.

**Evidence:**
- `backend/db/schema.sql` exists
- No migration tool (Alembic, etc.)
- No version tracking
- Docker init.sql used directly

**Impact:**
- Schema changes will break production
- Data loss during upgrades
- No rollback capability

**Likelihood:** HIGH when iterating
**Fixable:** YES

**Mitigation:**
1. Implement Alembic migrations
2. Version all schema changes
3. Test upgrade/downgrade paths

---

### RISK-008: Unverified External Dependencies

**Category:** Security / Supply Chain
**Severity:** HIGH
**Status:** ACTIVE

**Description:**
No lock files committed, dependencies not pinned to exact versions.

**Evidence:**
- No `package-lock.json` in contracts/mvp (though npm ci works)
- No `poetry.lock` for Python
- Requirements use ranges: `Flask==3.0.0` (some pinned, some not)
- 13 npm vulnerabilities reported

**Impact:**
- Non-reproducible builds
- Supply chain attacks possible
- Dependency drift between environments

**Likelihood:** MEDIUM
**Fixable:** YES

**Mitigation:**
1. Commit all lock files
2. Pin all dependencies to exact versions
3. Run `npm audit fix` and document unfixable
4. Use dependabot/renovate for updates

---

## 3. MEDIUM RISKS

### RISK-009: Documentation Claims Exceed Implementation

**Category:** Reputational / Legal
**Severity:** MEDIUM
**Status:** ACTIVE

**Description:**
README and documentation make claims that cannot be verified from the codebase.

**Evidence:**
- "99.9% uptime" - no monitoring data
- ">85% accuracy" - no test results
- "4 models in production" - trained on random data
- "16 services running" - docker-compose has 14

**Impact:**
- Loss of user trust
- Potential legal issues if marketed
- Contributor confusion

**Likelihood:** HIGH (already exists)
**Fixable:** YES

**Mitigation:**
1. Audit all claims against evidence
2. Add "Status: Planned/Development/Production" badges
3. Clearly separate vision from current state

---

### RISK-010: Python Version Inconsistency

**Category:** Operational
**Severity:** MEDIUM
**Status:** ACTIVE

**Description:**
Different Python versions specified across documentation and CI.

**Evidence:**
- CLAUDE.md: "Python 3.9+"
- CI: Uses Python 3.11
- Local development: Python 3.13 available
- Some packages incompatible with 3.13

**Impact:**
- "Works on my machine" issues
- CI passes but local fails
- Package compatibility problems

**Likelihood:** MEDIUM
**Fixable:** YES

**Mitigation:**
1. Standardize on Python 3.11
2. Update all documentation
3. Add `.python-version` file
4. Test on specified version only

---

### RISK-011: No Rate Limiting on Keeper Functions

**Category:** Security / DoS
**Severity:** MEDIUM
**Status:** ACTIVE

**Description:**
Keeper functions can be called repeatedly without throttling.

**Evidence:**
```solidity
// Only checks if should compound, no rate limit
function executeCompound(uint256 tokenId) external onlyAuthorizedKeeper {
    require(shouldCompound(tokenId), "Compound not needed");
    // No per-block or per-time limit
}
```

**Impact:**
- Gas griefing possible
- Excessive function calls
- Blockchain spam

**Likelihood:** LOW
**Fixable:** YES

**Mitigation:**
1. Add block-based cooldowns
2. Implement gas price checks
3. Add call frequency limits

---

### RISK-012: Minimal Test Coverage

**Category:** Quality
**Severity:** MEDIUM
**Status:** ACTIVE

**Description:**
Test coverage is minimal across all components.

**Evidence:**
- Contracts: 17 tests (unit only, no integration)
- Backend: 4 passing tests (smoke tests only)
- dexter-liquidity: Tests exist but import errors

**Impact:**
- Bugs go undetected
- Refactoring is risky
- No regression protection

**Likelihood:** HIGH
**Fixable:** YES

**Mitigation:**
1. Add integration tests for contracts
2. Add API endpoint tests
3. Add ML pipeline tests with mock data
4. Target 80% coverage for critical paths

---

## 4. LOW RISKS

### RISK-013: Hardcoded IP Addresses in Documentation

**Category:** Security / Operational
**Severity:** LOW
**Status:** ACTIVE

**Description:**
VPS IP address (5.78.71.231) mentioned in README.

**Evidence:**
- README.md mentions production IP
- Could enable targeted attacks
- IP may change

**Impact:**
- Security exposure (minor)
- Outdated documentation

**Likelihood:** LOW
**Fixable:** YES

**Mitigation:**
1. Remove specific IPs from public docs
2. Use domain names instead
3. Document infra privately

---

### RISK-014: Missing Pre-commit Hooks

**Category:** Quality
**Severity:** LOW
**Status:** ACTIVE

**Description:**
No pre-commit hooks for linting, formatting, or secrets scanning.

**Evidence:**
- No `.pre-commit-config.yaml`
- No git hooks configured
- Secrets scanning mentioned but not implemented

**Impact:**
- Inconsistent code style
- Risk of committing secrets
- Quality issues slip through

**Likelihood:** MEDIUM
**Fixable:** YES

**Mitigation:**
1. Add pre-commit configuration
2. Include: black, isort, eslint, prettier
3. Add gitleaks or similar for secrets

---

### RISK-015: No Error Monitoring/Alerting

**Category:** Operational
**Severity:** LOW
**Status:** ACTIVE

**Description:**
No error tracking or alerting system configured.

**Evidence:**
- No Sentry or similar
- No PagerDuty integration
- No Slack/Discord alerts
- Only basic logging

**Impact:**
- Errors go unnoticed
- Slow incident response
- Poor debugging capability

**Likelihood:** MEDIUM (when running)
**Fixable:** YES

**Mitigation:**
1. Integrate Sentry for error tracking
2. Add alerting for critical failures
3. Configure log aggregation

---

## 5. Risks That Require De-scoping

### RISK-016: Claiming "Production" Status Prematurely

**Recommendation:** MUST DE-SCOPE

**Current State:** README claims production deployment
**Reality:** Development/testing phase at best

**Action:** Change all "production" claims to "development" or "beta" until:
- Contracts deployed to mainnet
- ML trained on real data
- Infrastructure verified operational
- Security audit completed

---

### RISK-017: Multi-Chain Roadmap Without Single-Chain Working

**Recommendation:** DE-SCOPE ROADMAP

**Current State:** Roadmap includes Arbitrum, Optimism, Polygon, Solana
**Reality:** Base Network not fully operational

**Action:** Remove multi-chain roadmap until:
- Base Network deployment verified
- Core functionality proven
- Architecture supports multi-chain

---

## 6. Risk Matrix Summary

| Risk ID | Category | Severity | Likelihood | Fixable | Priority |
|---------|----------|----------|------------|---------|----------|
| RISK-001 | Security | CRITICAL | HIGH | YES | P0 |
| RISK-002 | Operational | CRITICAL | HIGH | YES | P0 |
| RISK-003 | Security | CRITICAL | MEDIUM | YES | P0 |
| RISK-004 | Operational | CRITICAL | HIGH | YES | P0 |
| RISK-005 | Security | HIGH | HIGH | YES | P1 |
| RISK-006 | Security | HIGH | MEDIUM | YES | P1 |
| RISK-007 | Operational | HIGH | HIGH | YES | P1 |
| RISK-008 | Security | HIGH | MEDIUM | YES | P1 |
| RISK-009 | Reputational | MEDIUM | HIGH | YES | P2 |
| RISK-010 | Operational | MEDIUM | MEDIUM | YES | P2 |
| RISK-011 | Security | MEDIUM | LOW | YES | P2 |
| RISK-012 | Quality | MEDIUM | HIGH | YES | P2 |
| RISK-013 | Security | LOW | LOW | YES | P3 |
| RISK-014 | Quality | LOW | MEDIUM | YES | P3 |
| RISK-015 | Operational | LOW | MEDIUM | YES | P3 |
| RISK-016 | Reputational | N/A | N/A | DE-SCOPE | - |
| RISK-017 | Planning | N/A | N/A | DE-SCOPE | - |

---

## 7. Recommended Fix Order

### Phase 1: Block Deployment (P0)
1. Fix RISK-001: Implement real contract functions
2. Fix RISK-003: Add position limit enforcement
3. Fix RISK-004: Return errors instead of mock data
4. Update RISK-016: Change production claims

### Phase 2: Enable Safe Operation (P1)
5. Fix RISK-005: Integrate TWAP protection
6. Fix RISK-006: Add emergency pause
7. Fix RISK-007: Implement migrations
8. Fix RISK-008: Lock dependencies
9. Fix RISK-002: Connect real data (parallel with above)

### Phase 3: Improve Quality (P2)
10. Fix RISK-009: Update documentation
11. Fix RISK-010: Standardize Python
12. Fix RISK-011: Add rate limiting
13. Fix RISK-012: Expand tests

### Phase 4: Polish (P3)
14. Fix RISK-013: Remove hardcoded IPs
15. Fix RISK-014: Add pre-commit
16. Fix RISK-015: Add monitoring
