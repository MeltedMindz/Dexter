import pytest
from dexbrain.db_manager import DexBrainDB

@pytest.fixture
def db():
    return DexBrainDB()

def test_query_strategies(db):
    strategy = db.query_strategies("SOL/USDC")
    assert strategy is None or isinstance(strategy, dict)
