"""API Key Authentication System for DexBrain Network"""

import secrets
import hashlib
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
from .config import Config


class APIKeyManager:
    """Manages API keys for external agents"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Config.KNOWLEDGE_DB_PATH / "api_keys.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load existing API keys from storage"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                self.keys_db = json.load(f)
        else:
            self.keys_db = {}
    
    def _save_keys(self) -> None:
        """Save API keys to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.keys_db, f, indent=2)
    
    def generate_api_key(self, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Generate a new API key for an agent
        
        Args:
            agent_id: Unique identifier for the agent
            metadata: Additional information about the agent
            
        Returns:
            Generated API key
        """
        # Generate secure random API key
        raw_key = secrets.token_urlsafe(32)
        api_key = f"dx_{raw_key}"
        
        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store key metadata
        self.keys_db[key_hash] = {
            'agent_id': agent_id,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'request_count': 0,
            'data_submissions': 0,
            'is_active': True,
            'metadata': metadata or {}
        }
        
        self._save_keys()
        return api_key
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return agent info
        
        Args:
            api_key: API key to validate
            
        Returns:
            Agent information if valid, None otherwise
        """
        if not api_key or not api_key.startswith('dx_'):
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.keys_db:
            return None
        
        key_info = self.keys_db[key_hash]
        
        if not key_info.get('is_active', False):
            return None
        
        # Update usage statistics
        key_info['last_used'] = datetime.now().isoformat()
        key_info['request_count'] += 1
        self._save_keys()
        
        return key_info
    
    def record_data_submission(self, api_key: str) -> bool:
        """Record a data submission from an agent
        
        Args:
            api_key: API key of the submitting agent
            
        Returns:
            True if recorded successfully
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.keys_db:
            self.keys_db[key_hash]['data_submissions'] += 1
            self._save_keys()
            return True
        
        return False
    
    def deactivate_key(self, api_key: str) -> bool:
        """Deactivate an API key
        
        Args:
            api_key: API key to deactivate
            
        Returns:
            True if deactivated successfully
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.keys_db:
            self.keys_db[key_hash]['is_active'] = False
            self._save_keys()
            return True
        
        return False
    
    def get_agent_stats(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an agent
        
        Args:
            api_key: API key of the agent
            
        Returns:
            Agent statistics if found
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.keys_db:
            return self.keys_db[key_hash].copy()
        
        return None
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents
        
        Returns:
            Dictionary of agent information keyed by agent_id
        """
        agents = {}
        for key_hash, key_info in self.keys_db.items():
            agent_id = key_info.get('agent_id')
            if agent_id:
                agents[agent_id] = key_info.copy()
        
        return agents


class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # {api_key: [timestamp, ...]}
    
    def is_allowed(self, api_key: str) -> bool:
        """Check if request is allowed under rate limit
        
        Args:
            api_key: API key making the request
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Initialize if not exists
        if api_key not in self.requests:
            self.requests[api_key] = []
        
        # Remove old requests
        self.requests[api_key] = [
            timestamp for timestamp in self.requests[api_key]
            if timestamp > minute_ago
        ]
        
        # Check if under limit
        if len(self.requests[api_key]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[api_key].append(current_time)
        return True


def require_api_key(rate_limiter: RateLimiter, api_key_manager: APIKeyManager):
    """Decorator for API endpoints requiring authentication"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract API key from request headers
            api_key = kwargs.get('api_key') or (args[0].headers.get('Authorization', '').replace('Bearer ', '') if args else None)
            
            if not api_key:
                return {'error': 'API key required', 'code': 401}, 401
            
            # Validate API key
            agent_info = api_key_manager.validate_key(api_key)
            if not agent_info:
                return {'error': 'Invalid API key', 'code': 401}, 401
            
            # Check rate limit
            if not rate_limiter.is_allowed(api_key):
                return {'error': 'Rate limit exceeded', 'code': 429}, 429
            
            # Add agent info to kwargs
            kwargs['agent_info'] = agent_info
            
            return func(*args, **kwargs)
        return wrapper
    return decorator