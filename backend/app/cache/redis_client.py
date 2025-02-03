from typing import Optional, Any, Union, List, AsyncIterator, Dict
import redis.asyncio as redis
from datetime import timedelta
import logging
import json
import asyncio
from functools import wraps
from backend.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass

class ConnectionError(CacheError):
    """Raised when Redis connection fails"""
    pass

class SerializationError(CacheError):
    """Raised when data serialization/deserialization fails"""
    pass

def handle_redis_errors(func):
    """Decorator to handle Redis operation errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error in {func.__name__}: {e}")
            raise ConnectionError(f"Failed to connect to Redis: {e}")
        except redis.RedisError as e:
            logger.error(f"Redis error in {func.__name__}: {e}")
            raise CacheError(f"Redis operation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper

class CacheManager:
    """
    Manages Redis caching operations with connection pooling and error handling.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: Optional[timedelta] = None,
        max_pool_size: Optional[int] = None,
        encoding: str = 'utf-8'
    ):
        """Initialize cache manager with connection settings"""
        # Use settings from config if not provided
        self.redis_url = redis_url or str(settings.cache.REDIS_URL)
        self.default_ttl = default_ttl or timedelta(seconds=settings.cache.DEFAULT_TTL)
        self.max_pool_size = max_pool_size or settings.cache.POOL_SIZE
        self.encoding = encoding
        self.key_prefix = settings.cache.KEY_PREFIX
        
        # Initialize connection pool
        self.redis = redis.from_url(
            self.redis_url,
            max_connections=self.max_pool_size,
            decode_responses=True,
            encoding=encoding
        )
        
        logger.info(f"Initialized Redis cache manager with URL: {self.redis_url}")

    def _make_key(self, key: str, namespace: Optional[str] = None) -> str:
        """
        Create prefixed cache key
        
        Args:
            key: Base key
            namespace: Optional namespace (e.g., 'ai', 'metrics')
        """
        if namespace:
            return f"{self.key_prefix}{namespace}:{key}"
        return f"{self.key_prefix}{key}"

    async def initialize(self) -> None:
        """Initialize connection and verify Redis is accessible"""
        try:
            await self.redis.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")

    @handle_redis_errors
    async def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve value from cache"""
        value = await self.redis.get(self._make_key(key, namespace))
        if value is None:
            return None
            
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize cache value for key {key}: {e}")
            raise SerializationError(f"Failed to deserialize value: {e}")

    @handle_redis_errors
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Store value in cache with optional TTL"""
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize value for key {key}: {e}")
            raise SerializationError(f"Failed to serialize value: {e}")
            
        ttl = ttl or self.default_ttl
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
            
        return await self.redis.setex(
            self._make_key(key, namespace),
            ttl,
            serialized
        )

    @handle_redis_errors
    async def delete(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> bool:
        """Remove key from cache"""
        return bool(await self.redis.delete(self._make_key(key, namespace)))

    @handle_redis_errors
    async def exists(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> bool:
        """Check if key exists in cache"""
        return bool(await self.redis.exists(self._make_key(key, namespace)))

    @handle_redis_errors  
    async def get_or_set(
        self,
        key: str,
        getter_func,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: Optional[str] = None
    ) -> Any:
        """Get cached value or compute and cache if missing"""
        value = await self.get(key, namespace)
        if value is not None:
            return value
            
        value = await getter_func()
        if value is not None:
            await self.set(key, value, ttl, namespace)
        return value

    @handle_redis_errors
    async def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: Optional[str] = None
    ) -> int:
        """Increment counter by amount"""
        key = self._make_key(key, namespace)
        pipe = self.redis.pipeline()
        pipe.incrby(key, amount)
        
        if ttl:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            pipe.expire(key, ttl)
            
        results = await pipe.execute()
        return results[0]

    @handle_redis_errors
    async def mget(
        self,
        keys: List[str],
        namespace: Optional[str] = None
    ) -> List[Optional[Any]]:
        """Get multiple values at once"""
        prefixed_keys = [self._make_key(key, namespace) for key in keys]
        values = await self.redis.mget(prefixed_keys)
        
        return [
            json.loads(v) if v is not None else None
            for v in values
        ]

    @handle_redis_errors
    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[Union[int, timedelta]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set multiple key-value pairs at once"""
        if not mapping:
            return True
            
        pipe = self.redis.pipeline()
        
        for key, value in mapping.items():
            serialized = json.dumps(value)
            prefixed_key = self._make_key(key, namespace)
            
            if ttl:
                _ttl = int(ttl.total_seconds()) if isinstance(ttl, timedelta) else ttl
                pipe.setex(prefixed_key, _ttl, serialized)
            else:
                pipe.set(prefixed_key, serialized)
                
        await pipe.execute()
        return True

    @handle_redis_errors
    async def scan_keys(
        self,
        pattern: str,
        namespace: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Scan keys matching pattern"""
        pattern = self._make_key(pattern, namespace)
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            for key in keys:
                # Remove prefix when yielding
                clean_key = key.replace(self.key_prefix, '', 1)
                if namespace:
                    clean_key = clean_key.replace(f"{namespace}:", '', 1)
                yield clean_key
            if cursor == 0:
                break

    @handle_redis_errors
    async def clear_pattern(
        self,
        pattern: str,
        namespace: Optional[str] = None
    ) -> int:
        """Delete all keys matching pattern"""
        deleted = 0
        async for key in self.scan_keys(pattern, namespace):
            if await self.delete(key, namespace):
                deleted += 1
        return deleted

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connections"""
        try:
            await self.redis.close()
            logger.info("Closed Redis connections")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")