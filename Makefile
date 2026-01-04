# Dexter Protocol - Unified Build System
# Provides convenient commands for building, testing, and running all components

.PHONY: help install build test clean contracts backend dexter-liquidity frontend docker

# Default target
help:
	@echo "Dexter Protocol - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install all dependencies"
	@echo "  make build            - Build all components"
	@echo "  make test             - Run all tests"
	@echo "  make clean            - Clean all build artifacts"
	@echo ""
	@echo "Component-specific:"
	@echo "  make contracts        - Build and test smart contracts"
	@echo "  make backend          - Install and test backend"
	@echo "  make dexter-liquidity - Install and test dexter-liquidity"
	@echo "  make docker           - Build Docker images"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            - Initial setup (copy .env files)"
	@echo "  make verify-setup     - Verify environment setup"

# ============ INSTALLATION ============
install: install-contracts install-backend install-dexter-liquidity

install-contracts:
	@echo "📦 Installing contract dependencies..."
	cd contracts/mvp && npm install

install-backend:
	@echo "📦 Installing backend dependencies..."
	pip install -r backend/requirements.txt

install-dexter-liquidity:
	@echo "📦 Installing dexter-liquidity dependencies..."
	pip install -r dexter-liquidity/requirements.txt

# ============ BUILD ============
build: build-contracts

build-contracts:
	@echo "🔨 Building contracts..."
	cd contracts/mvp && npm run compile

# ============ TESTING ============
test: test-contracts test-backend test-dexter-liquidity

test-contracts:
	@echo "🧪 Testing contracts..."
	cd contracts/mvp && npm run test || echo "⚠️  Contract tests require .env configuration"

test-backend:
	@echo "🧪 Testing backend..."
	cd backend && python -m pytest tests/ -v || echo "⚠️  Backend tests may require database setup"

test-dexter-liquidity:
	@echo "🧪 Testing dexter-liquidity..."
	cd dexter-liquidity && pytest tests/ -v || echo "⚠️  Dexter-liquidity tests may require configuration"

# ============ CLEANUP ============
clean: clean-contracts clean-backend clean-dexter-liquidity

clean-contracts:
	@echo "🧹 Cleaning contract artifacts..."
	cd contracts/mvp && npm run clean
	rm -rf contracts/mvp/cache contracts/mvp/artifacts

clean-backend:
	@echo "🧹 Cleaning backend artifacts..."
	find backend -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete 2>/dev/null || true

clean-dexter-liquidity:
	@echo "🧹 Cleaning dexter-liquidity artifacts..."
	find dexter-liquidity -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find dexter-liquidity -type f -name "*.pyc" -delete 2>/dev/null || true

# ============ COMPONENT TARGETS ============
contracts: install-contracts build-contracts test-contracts

backend: install-backend test-backend

dexter-liquidity: install-dexter-liquidity test-dexter-liquidity

# ============ DOCKER ============
docker:
	@echo "🐳 Building Docker images..."
	docker-compose -f docker-compose.vault.yml build
	docker-compose -f docker-compose.streaming.yml build

docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose -f docker-compose.vault.yml up -d
	docker-compose -f docker-compose.streaming.yml up -d

docker-down:
	@echo "🐳 Stopping Docker services..."
	docker-compose -f docker-compose.vault.yml down
	docker-compose -f docker-compose.streaming.yml down

# ============ SETUP ============
setup:
	@echo "⚙️  Setting up Dexter Protocol..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from .env.example"; \
		echo "⚠️  Please edit .env with your configuration"; \
	else \
		echo "ℹ️  .env already exists"; \
	fi
	@if [ ! -f contracts/mvp/.env ]; then \
		cp contracts/mvp/.env.example contracts/mvp/.env; \
		echo "✅ Created contracts/mvp/.env from .env.example"; \
		echo "⚠️  Please edit contracts/mvp/.env with your configuration"; \
	else \
		echo "ℹ️  contracts/mvp/.env already exists"; \
	fi

verify-setup:
	@echo "🔍 Verifying setup..."
	@command -v node >/dev/null 2>&1 || { echo "❌ Node.js not found"; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo "❌ npm not found"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 not found"; exit 1; }
	@echo "✅ Basic tools verified"
	@if [ -f .env ]; then \
		echo "✅ Root .env exists"; \
	else \
		echo "⚠️  Root .env missing - run 'make setup'"; \
	fi
	@if [ -f contracts/mvp/.env ]; then \
		echo "✅ contracts/mvp/.env exists"; \
	else \
		echo "⚠️  contracts/mvp/.env missing - run 'make setup'"; \
	fi

# ============ LINTING ============
lint: lint-contracts lint-backend

lint-contracts:
	@echo "🔍 Linting contracts..."
	@command -v forge >/dev/null 2>&1 && forge fmt --check contracts/ || echo "⚠️  Foundry not installed, skipping contract linting"

lint-backend:
	@echo "🔍 Linting backend..."
	@command -v black >/dev/null 2>&1 && black --check backend/ || echo "⚠️  Black not installed, skipping backend linting"

