import pytest
import psycopg2

try:
    from dexbrain.db_manager import DexBrainDB
    DB_MANAGER_AVAILABLE = True
except ImportError:
    DB_MANAGER_AVAILABLE = False


@pytest.fixture
def db():
    """Create DB connection, skip if PostgreSQL not available"""
    if not DB_MANAGER_AVAILABLE:
        pytest.skip("db_manager not available")
    try:
        return DexBrainDB()
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available - skipping DB tests")


@pytest.mark.skipif(not DB_MANAGER_AVAILABLE, reason="db_manager not available")
def test_query_strategies(db):
    """Test strategy query - requires PostgreSQL connection"""
    if db is None:
        pytest.skip("DB fixture returned None")
    strategy = db.query_strategies("SOL/USDC")
    assert strategy is None or isinstance(strategy, dict)
