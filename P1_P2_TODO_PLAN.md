# P1/P2 TODO Plan - Test Coverage & Developer Experience

**Date:** 2025-01-27  
**Status:** P1 In Progress

---

## P1: High Priority (Must Fix)

### P1-0: Backend Install Fix ✅ COMPLETED
- **Component:** Backend
- **Task:** Fix Python 3.13 compatibility issue blocking `make install`
- **Status:** ✅ Fixed Makefile to use `python3 -m pip` and upgrade setuptools
- **Acceptance Criteria:** `make install-backend` completes without errors
- **Effort:** S
- **Risk:** Low - Makefile change only

### P1-1: Contract Test Suite ✅ COMPLETED
- **Component:** Contracts
- **Task:** Create meaningful test suite for MVP contracts
- **Status:** ✅ Created 17 passing tests covering:
  - Deployment tests (3 contracts)
  - Access control (owner/keeper authorization)
  - Constants validation
  - Concentration level management
- **Acceptance Criteria:** 
  - ✅ `npm run test` runs and passes
  - ✅ Tests cover deployment, access control, constants
  - ✅ Tests fail when invariants are broken
- **Effort:** M
- **Risk:** Low - Tests are deterministic and fast

### P1-2: Backend Smoke Tests ✅ COMPLETED
- **Component:** Backend
- **Task:** Create smoke tests for imports, env, and API structure
- **Status:** ✅ Created 3 test files:
  - `test_imports.py` - Module import validation
  - `test_env.py` - Environment configuration safety
  - `test_api_smoke.py` - API structure validation
- **Acceptance Criteria:**
  - ✅ `pytest tests/test_imports.py` runs (skips if deps missing)
  - ✅ Tests handle missing dependencies gracefully
  - ✅ Tests validate .env.example structure
- **Effort:** S
- **Risk:** Low - Smoke tests are non-blocking

### P1-3: Environment File Hardening ✅ VERIFIED
- **Component:** Config
- **Task:** Ensure .env.example files exist and are tracked
- **Status:** ✅ Verified:
  - Root `.env.example` exists
  - `contracts/mvp/.env.example` exists
  - Both are tracked in git (not ignored)
  - `.gitignore` correctly allows `.env.example` via `!.env.example`
- **Acceptance Criteria:**
  - ✅ `.env.example` files exist and are committed
  - ✅ `make setup` can copy them to `.env`
- **Effort:** S
- **Risk:** None

### P1-4: CI Alignment ⚠️ IN PROGRESS
- **Component:** CI/CD
- **Task:** Ensure CI uses canonical commands and fails deterministically
- **Status:** ⚠️ Partially complete:
  - ✅ Contracts testing added to CI
  - ✅ Backend smoke tests added to CI
  - ⚠️ Need to verify CI uses Makefile targets
  - ⚠️ Need to ensure Python version is pinned
- **Acceptance Criteria:**
  - CI runs `make build` and `make test`
  - CI fails loudly when tests fail
  - Python version is pinned (3.9-3.11, not 3.13)
- **Effort:** M
- **Risk:** Medium - CI changes affect all PRs

---

## P2: Nice to Have (Future Improvements)

### P2-1: Contract Test Coverage Expansion
- **Component:** Contracts
- **Task:** Add integration tests with mock Uniswap contracts
- **Acceptance Criteria:** 
  - Tests for actual position management flows
  - Tests for compound/rebalance logic with mocks
  - Gas usage benchmarks
- **Effort:** L
- **Risk:** Low

### P2-2: Backend Integration Tests
- **Component:** Backend
- **Task:** Add tests for API endpoints with test database
- **Acceptance Criteria:**
  - Tests for `/health`, `/api/register`, key endpoints
  - Tests use in-memory or test database
  - Coverage > 60%
- **Effort:** L
- **Risk:** Medium - Requires database setup

### P2-3: Security Tooling
- **Component:** All
- **Task:** Add automated security scanning
- **Acceptance Criteria:**
  - Slither for Solidity contracts
  - Bandit for Python code
  - Runs in CI on every PR
- **Effort:** M
- **Risk:** Low

### P2-4: Linting & Formatting
- **Component:** All
- **Task:** Add consistent linting/formatting configs
- **Acceptance Criteria:**
  - `.eslintrc` for contracts
  - `pyproject.toml` with black/isort config
  - Pre-commit hooks
- **Effort:** M
- **Risk:** Low

### P2-5: Performance Benchmarks
- **Component:** Contracts
- **Task:** Document actual gas costs vs. claims
- **Acceptance Criteria:**
  - Gas reports in CI
  - Benchmarks for key operations
  - Comparison with manual strategies
- **Effort:** M
- **Risk:** Low

### P2-6: Documentation Expansion
- **Component:** Docs
- **Task:** Expand API docs, deployment guides
- **Acceptance Criteria:**
  - OpenAPI/Swagger for backend API
  - Contract deployment guide
  - Architecture diagrams
- **Effort:** L
- **Risk:** Low

---

## Execution Summary

### Completed (P1)
1. ✅ **Contract Tests** - 17 passing tests created
2. ✅ **Backend Smoke Tests** - 3 test files created (7 tests, 4 passing, 3 skipping gracefully)
3. ✅ **Makefile Fix** - Backend install uses `python3 -m pip`
4. ✅ **Env Files** - Verified .env.example files exist and are tracked

### In Progress (P1)
1. ⚠️ **CI Alignment** - Need to verify CI uses Makefile and pins Python version

### Remaining (P2)
- All P2 items are documented and ready for future work

---

## Verification Commands

```bash
# Contracts
cd contracts/mvp && npm run test
# Expected: 17 passing

# Backend (with deps installed)
cd backend && pytest tests/test_imports.py tests/test_env.py tests/test_api_smoke.py -v
# Expected: Tests pass or skip gracefully if deps missing

# Full suite
make test
# Expected: Contracts pass, backend tests run
```

