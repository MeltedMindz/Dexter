"""
Smoke tests for API server.
Tests basic API structure without requiring full database setup.
"""
import pytest
import sys
import os

# Add parent directory to path
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)


def test_api_server_import():
    """Test that api_server can be imported (may require deps)."""
    try:
        import dexbrain.api_server
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        # Expected if dependencies aren't installed
        if any(dep in str(e).lower() for dep in ['flask', 'psycopg2', 'redis']):
            pytest.skip(f"Dependencies not installed: {e}")
        else:
            pytest.fail(f"Unexpected import error: {e}")


def test_api_structure_exists():
    """Test that API structure is defined (if import succeeds)."""
    try:
        from dexbrain import api_server
        # Check that it's a module
        assert api_server is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Cannot test API structure - dependencies missing: {e}")

