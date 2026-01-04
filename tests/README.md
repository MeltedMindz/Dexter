# Tests

## Purpose
The tests directory contains cross-component integration tests and test utilities that verify the interaction between different parts of the Dexter Protocol system.

## What Lives Here
- **Integration tests** - Tests that verify interactions between contracts, backend, and other components
- **Test utilities** - Shared testing helpers and fixtures
- **MVP tests** - Tests specific to the MVP implementation

## How It Fits Into the System
- **Tests**: Integration between `contracts/`, `backend/`, and other system components
- **Depends on**: All system components being properly configured
- **Provides**: Regression protection and validation of cross-component functionality
- **Part of**: The overall test suite alongside component-specific tests (see `contracts/mvp/test/` and `backend/tests/`)

## Current Status
🚧 **In development** - Test structure exists, coverage expansion in progress

## What This Is NOT
- This is not the contract unit tests (those are in `contracts/mvp/test/`)
- This is not the backend unit tests (those are in `backend/tests/`)
- This is not the dexter-liquidity tests (those are in `dexter-liquidity/tests/`)

## Relevant Docs / Entry Points
- **Contract tests**: See `contracts/mvp/test/`
- **Backend tests**: See `backend/tests/`
- **Root documentation**: See `../README.md`
- **Testing guide**: See `../CONTRIBUTING.md` for testing standards

## Running Tests
```bash
# Run all tests from root
make test

# Run specific test suites
cd contracts/mvp && npm test
cd backend && pytest tests/
```

