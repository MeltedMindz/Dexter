"""
Authentication and Rate Limiting System for DexBrain
"""
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import jwt
import redis.asyncio as redis
import logging
from enum import Enum
from pydantic import BaseModel
import hashlib
import secrets

# Models
class UserRole(str, Enum):
    ADMIN = "admin"
    PRO = "pro"
    BASIC = "basic"

class RateLimitTier(BaseModel):
    """Rate limit configuration for user tier"""
    requests_per_minute: int
    burst_size: int
    monthly_request_limit: Optional[int]

class User(BaseModel):
    """User information"""
    id: str
    email: str
    role: UserRole
    api_key: str
    rate_limit_tier: RateLimitTier

class AuthConfig:
    """Authentication configuration"""
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    API_KEY_HEADER: str = "X-API-Key"

class RateLimiter:
    """Token bucket rate limiter with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Default rate limit tiers
        self.rate_limit_tiers = {
            UserRole.BASIC: RateLimitTier(
                requests_per_minute=60,
                burst_size=10,
                monthly_request_limit=10000
            ),
            UserRole.PRO: RateLimitTier(
                requests_per_minute=300,
                burst_size=50,
                monthly_request_limit=100000
            ),
            UserRole.ADMIN: RateLimitTier(
                requests_per_minute=1000,
                burst_size=100,
                monthly_request_limit=None
            )
        }
        
    async def check_rate_limit(
        self,
        user_id: str,
        tier: RateLimitTier
    ) -> bool:
        """Check if request is within rate limits"""
        now = datetime.utcnow()
        bucket_key = f"ratelimit:bucket:{user_id}"
        monthly_key = f"ratelimit:monthly:{user_id}:{now.strftime('%Y%m')}"
        
        async with self.redis.pipeline() as pipe:
            try:
                # Get current bucket tokens
                tokens = await pipe.get(bucket_key)
                tokens = float(tokens) if tokens else tier.burst_size
                
                # Calculate token refill
                last_update = await pipe.get(f"{bucket_key}:ts")
                last_update = (
                    datetime.fromisoformat(last_update)
                    if last_update else now
                )
                
                delta = (now - last_update).total_seconds()
                tokens = min(
                    tier.burst_size,
                    tokens + (delta * tier.requests_per_minute / 60)
                )
                
                # Check if request can be allowed
                if tokens < 1:
                    return False
                    
                # Check monthly limit if applicable
                if tier.monthly_request_limit:
                    monthly_count = await pipe.incr(monthly_key)
                    if int(monthly_count) > tier.monthly_request_limit:
                        return False
                    # Set expiry for monthly counter
                    await pipe.expire(
                        monthly_key,
                        timedelta(days=35)  # Ensure it covers full month
                    )
                
                # Update bucket
                await pipe.set(bucket_key, tokens - 1)
                await pipe.set(f"{bucket_key}:ts", now.isoformat())
                await pipe.expire(bucket_key, 60)  # 1 minute TTL
                
                return True
                
            except Exception as e:
                logging.error(f"Rate limit check failed: {e}")
                return False

class AuthManager:
    """Manages authentication and API keys"""
    
    def __init__(
        self,
        config: AuthConfig,
        redis_client: redis.Redis
    ):
        self.config = config
        self.redis = redis_client
        self.rate_limiter = RateLimiter(redis_client)
        
    def generate_api_key(self) -> str:
        """Generate new API key"""
        return secrets.token_urlsafe(32)
        
    async def create_user(
        self,
        email: str,
        role: UserRole = UserRole.BASIC
    ) -> User:
        """Create new user with API key"""
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
        api_key = self.generate_api_key()
        
        user = User(
            id=user_id,
            email=email,
            role=role,
            api_key=api_key,
            rate_limit_tier=self.rate_limiter.rate_limit_tiers[role]
        )
        
        # Store user data
        await self.redis.set(
            f"user:{user_id}",
            user.json(),
            ex=timedelta(days=90)
        )
        await self.redis.set(
            f"apikey:{api_key}",
            user_id,
            ex=timedelta(days=90)
        )
        
        return user
        
    async def get_user(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        user_id = await self.redis.get(f"apikey:{api_key}")
        if not user_id:
            return None
            
        user_data = await self.redis.get(f"user:{user_id}")
        if not user_data:
            return None
            
        return User.parse_raw(user_data)
        
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            if not user_id:
                return None
                
            user_data = await self.redis.get(f"user:{user_id}")
            if not user_data:
                return None
                
            return User.parse_raw(user_data)
            
        except jwt.PyJWTError:
            return None
            
    def create_token(self, user: User) -> str:
        """Create JWT token for user"""
        expire = datetime.utcnow() + timedelta(
            minutes=self.config.JWT_EXPIRE_MINUTES
        )
        
        return jwt.encode(
            {
                "sub": user.id,
                "exp": expire,
                "role": user.role
            },
            self.config.SECRET_KEY,
            algorithm=self.config.JWT_ALGORITHM
        )

# FastAPI integration
api_key_header = APIKeyHeader(name="X-API-Key")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    api_key: str = Security(api_key_header),
    auth_manager: AuthManager = Depends()
) -> User:
    """Dependency to get current user from API key"""
    user = await auth_manager.get_user(api_key)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return user

async def check_rate_limit(
    user: User = Depends(get_current_user),
    rate_limiter: RateLimiter = Depends()
) -> None:
    """Dependency to check rate limits"""
    if not await rate_limiter.check_rate_limit(
        user.id,
        user.rate_limit_tier
    ):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

def requires_role(allowed_roles: List[UserRole]) -> Callable:
    """Decorator to check user role"""
    async def role_checker(user: User = Depends(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        return user
    return role_checker

# Usage example:
app = FastAPI()

@app.get(
    "/api/v1/pools",
    dependencies=[Depends(check_rate_limit)]
)
async def get_pools(user: User = Depends(get_current_user)):
    """Rate limited endpoint example"""
    return {"message": "Success"}

@app.get(
    "/api/v1/admin/users",
    dependencies=[
        Depends(requires_role([UserRole.ADMIN])),
        Depends(check_rate_limit)]
)
async def list_users(user: User = Depends(get_current_user)):
    """Admin endpoint to list users"""
    return {"message": "Success"}

# Add monitoring endpoints
@app.get("/api/v1/admin/rate-limits", dependencies=[
    Depends(requires_role([UserRole.ADMIN]))
])
async def get_rate_limit_stats(
    auth_manager: AuthManager = Depends(),
    rate_limiter: RateLimiter = Depends()
):
    """Get rate limiting statistics"""
    stats = {}
    async for key in rate_limiter.redis.scan_iter("ratelimit:*"):
        stats[key] = await rate_limiter.redis.get(key)
    return stats

# Add user management endpoints
class UserCreate(BaseModel):
    """User creation request"""
    email: str
    role: UserRole = UserRole.BASIC

@app.post("/api/v1/admin/users", dependencies=[
    Depends(requires_role([UserRole.ADMIN]))
])
async def create_user(
    user_data: UserCreate,
    auth_manager: AuthManager = Depends()
):
    """Create new user with API key"""
    user = await auth_manager.create_user(
        email=user_data.email,
        role=user_data.role
    )
    return {
        "user_id": user.id,
        "api_key": user.api_key
    }

@app.delete("/api/v1/admin/users/{user_id}", dependencies=[
    Depends(requires_role([UserRole.ADMIN]))
])
async def delete_user(
    user_id: str,
    auth_manager: AuthManager = Depends()
):
    """Delete user"""
    # Get user data
    user_data = await auth_manager.redis.get(f"user:{user_id}")
    if not user_data:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    user = User.parse_raw(user_data)
    
    # Remove user data and API key
    async with auth_manager.redis.pipeline() as pipe:
        await pipe.delete(f"user:{user_id}")
        await pipe.delete(f"apikey:{user.api_key}")
        await pipe.execute()
        
    return {"status": "success"}

# Add rate limit configuration endpoints
@app.put("/api/v1/admin/rate-limits/{role}", dependencies=[
    Depends(requires_role([UserRole.ADMIN]))
])
async def update_rate_limits(
    role: UserRole,
    limits: RateLimitTier,
    rate_limiter: RateLimiter = Depends()
):
    """Update rate limits for user role"""
    rate_limiter.rate_limit_tiers[role] = limits
    return {"status": "success"}

# Add monitoring middleware
@app.middleware("http")
async def monitor_requests(request, call_next):
    """Monitor API requests and response times"""
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Record metrics
    path = request.url.path
    method = request.method
    status_code = response.status_code
    
    # Update Prometheus metrics
    REQUEST_DURATION.labels(
        path=path,
        method=method,
        status=status_code
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        path=path,
        method=method,
        status=status_code
    ).inc()
    
    return response

# Initialize Prometheus metrics
from prometheus_client import Counter, Histogram

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['path', 'method', 'status']
)

REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total requests',
    ['path', 'method', 'status']
)

# Add rate limit metrics
RATE_LIMIT_EXCEEDED = Counter(
    'rate_limit_exceeded_total',
    'Number of rate limit exceeded errors',
    ['user_id', 'tier']
)

# Add authentication failure metrics
AUTH_FAILURES = Counter(
    'auth_failures_total',
    'Number of authentication failures',
    ['reason']
)

# Initialize services
def init_auth_services(
    redis_url: str,
    secret_key: str
) -> Tuple[AuthManager, RateLimiter]:
    """Initialize authentication and rate limiting services"""
    redis_client = redis.from_url(redis_url)
    
    config = AuthConfig(
        SECRET_KEY=secret_key
    )
    
    auth_manager = AuthManager(config, redis_client)
    rate_limiter = RateLimiter(redis_client)
    
    return auth_manager, rate_limiter

# Add startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    app.state.auth_manager, app.state.rate_limiter = init_auth_services(
        redis_url=settings.REDIS_URL,
        secret_key=settings.SECRET_KEY
    )

# Add graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await app.state.auth_manager.redis.close()
