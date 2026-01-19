"""
Environment configuration tests.
Verifies that environment parsing is safe and defaults work.
"""
import pytest
import os
import sys

# Add parent directory to path
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)


def test_env_defaults_safe():
    """Test that missing environment variables don't crash the app."""
    # Clear any existing env vars for this test
    test_vars = ['DATABASE_URL', 'REDIS_URL', 'ALCHEMY_API_KEY']
    original_values = {}
    
    for var in test_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    try:
        # Try to import config - it should handle missing env gracefully
        import dexbrain.config
        # If we get here, config loaded without crashing
        assert True
    except (ImportError, ModuleNotFoundError) as e:
        # If dependencies are missing, skip
        if any(dep in str(e).lower() for dep in ['psycopg2', 'flask', 'redis', 'dotenv']):
            pytest.skip(f"Dependencies not installed: {e}")
        else:
            pytest.fail(f"Config import failed: {e}")
    except Exception as e:
        # Config should handle missing env vars gracefully
        # If it crashes, that's a bug we need to fix
        pytest.fail(f"Config crashed with missing env vars: {e}")
    finally:
        # Restore original values
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value


def test_env_file_parsing():
    """Test that .env.example structure is valid."""
    env_example_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '.env.example'
    )
    
    if os.path.exists(env_example_path):
        with open(env_example_path, 'r') as f:
            content = f.read()
            # Basic validation: should have some key=value pairs
            assert '=' in content, ".env.example should contain key=value pairs"
            # Should use placeholder values (check for common placeholders)
            has_placeholders = any(
                placeholder in content.lower() 
                for placeholder in ['your_', 'example', 'placeholder', 'here', 'change_me']
            )
            # If it mentions password generation, that's fine (it's instructions)
            if 'password' in content.lower() and 'generate' in content.lower():
                has_placeholders = True
            assert has_placeholders, ".env.example should use placeholder values, not real secrets"
    else:
        pytest.skip(".env.example not found (may be in different location)")

