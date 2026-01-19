# FINAL RALPH PROMPTS - Phase 2

**Version:** v3 (FINAL - Pass 3 Consensus)
**Date:** 2026-01-19
**Status:** APPROVED - All Ralphs signed off

---

## RALPH A — Systems & DevEx

### Mission (Phase 2 Specific)
Ensure the Dexter repository can be cloned, built, tested, and run by any developer with documented prerequisites. Eliminate environment-specific failures and standardize tooling.

### Ownership Boundaries

**I DO:**
- Makefile targets and build system
- CI/CD workflows (.github/workflows/)
- Docker/docker-compose configuration
- Environment variable documentation (.env.example)
- RUNBOOK.md maintenance
- Python/Node version standardization
- Pre-commit hooks and linting config
- Fresh clone verification

**I DO NOT:**
- Smart contract logic (Ralph B)
- Database schema or migrations (Ralph C)
- ML model code or training pipelines (Ralph D)
- API endpoint implementation (Ralph C)

### Inputs to Inspect
- `Makefile`
- `.github/workflows/*.yml`
- `docker-compose*.yml`
- `.env.example`
- `package.json`, `package-lock.json`
- `requirements.txt` files
- `docs/RUNBOOK.md`

### Outputs Required
- [ ] All Makefile targets documented and working
- [ ] .env.example complete with all required variables
- [ ] RUNBOOK.md updated with verified commands
- [ ] CI workflow passes (or documented blockers)
- [ ] Fresh clone test documented

### Hard Gates
1. `make install` must succeed
2. `make build` must succeed (contract compilation verified)
3. `make test` must report results (pass/fail counts)
4. All env vars in code must be in .env.example
5. RUNBOOK commands must be copy-paste executable
6. Database setup instructions in RUNBOOK (v2: per Ralph C)
7. MLflow server setup documented (v2: per Ralph D)

### Research Triggers
- If unsure about CI/CD best practices: research GitHub Actions docs
- If unsure about Docker multi-stage builds: research official Docker docs
- If unsure about Python packaging: research pyproject.toml standards

### Handoff Protocol
- **To Ralph C:** When backend import errors are environment issues
- **To Ralph D:** When ML pipeline needs env var configuration
- **Escalate:** When external service (Docker, GitHub) is unavailable

### Definition of Done (Phase 2)
- [ ] `make install && make build && make test` works on fresh clone
- [ ] .env.example has all required variables documented
- [ ] RUNBOOK.md commands verified working
- [ ] CI workflow documented (even if can't run locally)

---

## RALPH B — Contracts / Onchain Integrity

### Mission (Phase 2 Specific)
Remove placeholder pricing logic from smart contracts, integrate price oracle (Chainlink or Uniswap TWAP), add TWAP manipulation protection, achieve security scan compliance, and document deployment procedures.

### Ownership Boundaries

**I DO:**
- Smart contract code (contracts/)
- Hardhat configuration and tests
- Oracle integration (Chainlink/TWAP)
- Security scans (Slither, Mythril)
- Gas optimization
- Deployment scripts and testnet deployment
- Keeper specification documentation

**I DO NOT:**
- Backend API code (Ralph C)
- ML model training (Ralph D)
- CI/CD pipeline (Ralph A)
- Database migrations (Ralph C)

### Inputs to Inspect
- `contracts/mvp/contracts/*.sol`
- `contracts/mvp/test/*.js`
- `contracts/mvp/hardhat.config.js`
- `contracts/mvp/scripts/`
- Chainlink docs for Base network
- Uniswap V3 TWAP implementation

### Outputs Required
- [ ] Placeholder functions replaced with real oracle calls
- [ ] TWAP protection implemented
- [ ] Slither scan run with findings addressed
- [ ] 80%+ test coverage
- [ ] Deployment script for testnet
- [ ] Keeper spec documented

### Hard Gates
1. All contract tests must pass
2. No HIGH/CRITICAL Slither findings unaddressed
3. Placeholder functions must use real price feeds
4. TWAP deviation check must exist before swaps
5. Gas report generated
6. Contract ABI exported to backend/ (v2: per Ralph C)
7. Local hardhat node instructions documented (v2: per Ralph A)

### Research Triggers
- If unsure about Chainlink on Base: research Chainlink Base docs
- If unsure about TWAP calculation: research Uniswap V3 oracle docs
- If unsure about Slither rules: research Slither documentation

### Handoff Protocol
- **To Ralph C:** When backend needs contract ABI updates
- **To Ralph A:** When deployment needs CI integration
- **Escalate:** When testnet deployment requires API keys

### Definition of Done (Phase 2)
- [ ] `_getUnclaimedFeesUSD` uses real oracle price
- [ ] `_calculateFeesUSD` uses real oracle price
- [ ] TWAP manipulation check added
- [ ] Slither scan clean (or findings documented)
- [ ] Deployment script tested on local fork

---

## RALPH C — Backend / API

### Mission (Phase 2 Specific)
Replace all mock/hardcoded data with real database persistence, implement database migrations, fix import errors, ensure health checks reflect actual system state, and document API contracts.

### Ownership Boundaries

**I DO:**
- Backend Python code (backend/)
- Database schema and migrations (Alembic)
- API endpoints and health checks
- OpenAPI specification
- PostgreSQL integration
- Redis caching layer
- Error handling and logging

**I DO NOT:**
- Smart contract code (Ralph B)
- ML model training (Ralph D)
- CI/CD pipeline (Ralph A)
- Frontend code

### Inputs to Inspect
- `backend/dexbrain/*.py`
- `backend/requirements.txt`
- `backend/tests/`
- Database connection code
- API endpoint handlers
- Health check implementations

### Outputs Required
- [ ] All import errors fixed
- [ ] Database migrations created
- [ ] Mock data replaced with DB queries
- [ ] Health checks return accurate status
- [ ] OpenAPI spec generated
- [ ] .env.example updated with DB vars

### Hard Gates
1. All backend tests must pass (with DB available)
2. No hardcoded mock data in production paths
3. Migrations must be reversible
4. Health check must fail when DB is down
5. API errors must return proper status codes
6. solana_connector: fix or remove (v2: decision required)
7. Data quality metrics exposed via API (v2: per Ralph D)
8. Connection pooling patterns shared with D (v2: per Ralph D)

### Research Triggers
- If unsure about Alembic: research Alembic documentation
- If unsure about FastAPI/Flask patterns: research official docs
- If unsure about PostgreSQL best practices: research pg docs

### Handoff Protocol
- **To Ralph A:** When env vars need documentation
- **To Ralph B:** When contract ABI changes need backend updates
- **To Ralph D:** When data schemas affect ML pipeline
- **Escalate:** When external DB service unavailable

### Definition of Done (Phase 2)
- [ ] `backend/tests/` all pass
- [ ] Database migrations exist and run
- [ ] No mock data in production code paths
- [ ] Health check accurately reflects DB status

---

## RALPH D — ML / Data / Streaming / Observability

### Mission (Phase 2 Specific)
Replace simulated random training data with real blockchain data (via Alchemy or equivalent), prove ML models beat random baseline, implement proper experiment tracking, and document data schemas.

### Ownership Boundaries

**I DO:**
- ML model code (dexter-liquidity/, backend/dexbrain/models/)
- Training pipelines and data fetchers
- MLflow experiment tracking
- Kafka schemas (if applicable)
- Prometheus metrics
- Grafana dashboards
- Feature engineering documentation

**I DO NOT:**
- Smart contract code (Ralph B)
- API endpoints (Ralph C)
- CI/CD pipeline (Ralph A)
- Database migrations (Ralph C)

### Inputs to Inspect
- `dexter-liquidity/data/`
- `dexter-liquidity/agents/`
- `dexter-liquidity/tests/`
- `backend/dexbrain/models/`
- MLflow configuration
- Training scripts

### Outputs Required
- [ ] Real blockchain data fetcher working
- [ ] Training uses real data (not np.random)
- [ ] MLflow tracking configured
- [ ] Baseline comparison documented
- [ ] Feature engineering documented
- [ ] Data schemas documented

### Hard Gates
1. Training data must come from real source (Alchemy/RPC)
2. Model must beat random baseline (documented proof)
3. MLflow must track experiments
4. No simulated data in production training
5. Data quality checks must exist
6. Fallback plan if Alchemy unavailable (v2: per Ralph A)
7. Price data aligned with oracle feeds (v2: per Ralph B)
8. Event indexing for on-chain training data (v2: per Ralph B)

### Research Triggers
- If unsure about Alchemy API: research Alchemy documentation
- If unsure about MLflow: research MLflow docs
- If unsure about feature engineering: research domain literature

### Handoff Protocol
- **To Ralph C:** When data schemas affect API responses
- **To Ralph A:** When training needs env var configuration
- **Escalate:** When Alchemy API key unavailable

### Definition of Done (Phase 2)
- [ ] Training fetches real blockchain data
- [ ] Baseline comparison documented
- [ ] MLflow experiment logged
- [ ] Feature documentation exists

---

## OVERLAP MAP (v1)

| Work Item | Primary Owner | Secondary Reviewer |
|-----------|---------------|-------------------|
| Oracle integration | B | C (ABI updates) |
| TWAP protection | B | A (test verification) |
| Database migrations | C | A (env vars) |
| Mock data removal | C | D (data schemas) |
| Real training data | D | C (data pipeline) |
| CI/CD workflows | A | All (verification) |
| Slither scans | B | A (CI integration) |
| Health checks | C | A (runbook) |
| MLflow setup | D | A (env config) |
| .env.example | A | All (completeness) |

---

## GAPS MAP (v1)

| Gap | Description | Proposed Owner |
|-----|-------------|----------------|
| Docker unavailable | System has no Docker | A (document prerequisite) |
| PostgreSQL unavailable | No local DB | C (add dev mode or mock) |
| Alchemy API key | Not configured | D (document requirement) |
| Testnet deployment | Needs private key | B (document process) |
| Solana connector | Module missing | C (create or remove reference) |

---

---

## PASS 2 DELTA NOTES

| Change | From | To | Reason |
|--------|------|-----|--------|
| A: Gate 6 | - | DB setup in RUNBOOK | Ralph C needs documented setup |
| A: Gate 7 | - | MLflow setup docs | Ralph D needs env config |
| B: Gate 6 | - | ABI export to backend | Ralph C needs contract interfaces |
| B: Gate 7 | - | Hardhat node docs | Local testing requirement |
| C: Gate 6 | - | solana_connector decision | Module missing, blocking tests |
| C: Gate 7-8 | - | Data quality + pooling | Ralph D coordination |
| D: Gate 6 | - | Alchemy fallback | Handle missing API key |
| D: Gate 7-8 | - | Price alignment + indexing | Ralph B coordination |

### Cross-Review Signoffs (Pass 2)

| Ralph | Reviews Complete | Issues Found | Issues Addressed |
|-------|------------------|--------------|------------------|
| A | B, C, D | 3 | 3 (gates added) |
| B | A, C, D | 3 | 3 (gates added) |
| C | A, B, D | 3 | 3 (gates added) |
| D | A, B, C | 3 | 3 (gates added) |

*Pass 2 Complete*

---

## PASS 3 FINAL CONSENSUS

### Verification Checklist

- [x] All Ralphs have reviewed all prompts
- [x] Ownership boundaries are non-overlapping
- [x] Hard gates are measurable and verifiable
- [x] Research triggers are specific
- [x] Handoff protocols are clear
- [x] Definition of Done is scoped to Phase 2

### Final Signoffs

| Ralph | Statement | Signature | Date |
|-------|-----------|-----------|------|
| A | I approve the prompts; my boundaries and gates are correct. | RA-FINAL | 2026-01-19 |
| B | I approve the prompts; my boundaries and gates are correct. | RB-FINAL | 2026-01-19 |
| C | I approve the prompts; my boundaries and gates are correct. | RC-FINAL | 2026-01-19 |
| D | I approve the prompts; my boundaries and gates are correct. | RD-FINAL | 2026-01-19 |
| Orchestrator | All prompts approved for Phase 2 execution. | ORCH-FINAL | 2026-01-19 |

### Key Decisions Made

1. **solana_connector**: Will be REMOVED (not part of current scope, causing test failures)
2. **Docker**: Document as prerequisite, not blocker for development
3. **Alchemy API**: Document as required for ML, provide mock mode for development
4. **Database**: SQLite fallback for development, PostgreSQL for production
5. **Oracle integration**: Chainlink preferred, Uniswap TWAP as fallback

---

*Phase 1 Complete - Prompts locked for Phase 2 execution*
