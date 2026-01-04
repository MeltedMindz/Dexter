"""
Smoke tests for backend imports.
Verifies that core modules can be imported without crashing.
"""
import pytest
import sys
import os

# Add parent directory to path so we can import dexbrain directly
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)


def test_import_dexbrain_core():
    """Test that core dexbrain module can be imported."""
    try:
        import dexbrain.core
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        # If dependencies are missing, that's expected in CI
        if any(dep in str(e).lower() for dep in ['psycopg2', 'flask', 'redis', 'numpy', 'aiofiles', 'sqlalchemy', 'loguru', 'pydantic']):
            pytest.skip(f"Dependencies not installed (expected): {e}")
        else:
            pytest.fail(f"Failed to import dexbrain.core: {e}")


def test_import_dexbrain_config():
    """Test that config module can be imported."""
    try:
        import dexbrain.config
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        if any(dep in str(e).lower() for dep in ['psycopg2', 'flask', 'redis']):
            pytest.skip(f"Dependencies not installed (expected): {e}")
        else:
            pytest.fail(f"Failed to import dexbrain.config: {e}")


def test_import_dexbrain_schemas():
    """Test that schemas module can be imported."""
    try:
        import dexbrain.schemas
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        if any(dep in str(e).lower() for dep in ['pydantic', 'flask']):
            pytest.skip(f"Dependencies not installed (expected): {e}")
        else:
            pytest.fail(f"Failed to import dexbrain.schemas: {e}")


def test_import_dexbrain_models():
    """Test that models module can be imported."""
    try:
        import dexbrain.models
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        if any(dep in str(e).lower() for dep in ['numpy', 'sklearn', 'torch', 'aiofiles', 'sqlalchemy']):
            pytest.skip(f"Dependencies not installed (expected): {e}")
        else:
            pytest.fail(f"Failed to import dexbrain.models: {e}")


def test_import_api_server_structure():
    """Test that api_server module structure exists (may fail if deps missing)."""
    try:
        # This may fail if Flask-CORS is not installed, which is expected
        import dexbrain.api_server
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        # Expected if dependencies aren't installed - this is a smoke test
        # The test passes if the module structure is correct
        if 'flask_cors' in str(e).lower() or 'psycopg2' in str(e).lower():
            pytest.skip(f"Dependencies not installed (expected in CI): {e}")
        else:
            pytest.fail(f"Unexpected import error: {e}")

