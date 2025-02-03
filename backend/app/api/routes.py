"""
API Routes Module - FastAPI implementation for DLMM data access.

This module provides REST API endpoints for accessing DLMM pool data 
and AI-generated strategies.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pydantic import BaseModel, Field

from app.core.agent import DexterAgent
from app.protocols.meteora.adapter import MeteoraAdapter
from app.core.metrics import MetricsManager
from app.cache.redis_client import CacheManager
from backend.config.settings import Settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DexBrain DLMM API",
    description="API for DLMM pool data and AI strategies",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PoolResponse(BaseModel):
    """DLMM pool data response"""
    pool_id: str
    token_a: str
    token_b: str
    tvl_usd: str
    fee_rate: str
    apy: str
    range_lower: str
    range_upper: str
    status: str
    last_updated: datetime
    warnings: List[str] = Field(default_factory=list)

class StrategyResponse(BaseModel):
    """AI strategy response"""
    pool_id: str
    token_pair: str
    optimal_range: tuple[float, float]
    suggested_fee: float
    confidence_score: float
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

# Initialize components
settings = Settings()
meteora = MeteoraAdapter(settings.SOLANA_RPC_URL)
ai_agent = DexterAgent()

# Dependencies
async def get_cache():
    """Get cache manager instance"""
    cache = CacheManager(settings.REDIS_URL)
    try:
        await cache.initialize()
        yield cache
    finally:
        await cache.close()

@app.on_event("startup")
async def startup():
    """Initialize components on startup"""
    try:
        await meteora.initialize()
        logger.info("Initialized Meteora protocol")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get(
    "/api/v1/pools",
    response_model=List[PoolResponse],
    tags=["Pools"]
)
async def get_pools(
    cache: CacheManager = Depends(get_cache)
) -> List[PoolResponse]:
    """Get all DLMM pools"""
    try:
        # Check cache
        cached = await cache.get("pools:all")
        if cached:
            return [PoolResponse(**pool) for pool in cached]

        # Fetch pools
        pools = await meteora.get_pools()
        formatted = [await meteora.format_for_api(pool) for pool in pools]
        
        # Cache results
        await cache.set("pools:all", formatted, ttl=timedelta(minutes=5))
        
        return [PoolResponse(**pool) for pool in formatted]
        
    except Exception as e:
        logger.error(f"Error retrieving pools: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve pool data"
        )

@app.get(
    "/api/v1/pools/{pool_id}",
    response_model=PoolResponse,
    tags=["Pools"]
)
async def get_pool(
    pool_id: str,
    cache: CacheManager = Depends(get_cache)
) -> PoolResponse:
    """Get specific DLMM pool details"""
    try:
        # Check cache
        cache_key = f"pool:{pool_id}"
        cached = await cache.get(cache_key)
        if cached:
            return PoolResponse(**cached)
        
        # Fetch pool
        pool = await meteora.get_pool(pool_id)
        if not pool:
            raise HTTPException(
                status_code=404,
                detail=f"Pool {pool_id} not found"
            )
            
        # Format and cache response
        response = await meteora.format_for_api(pool)
        await cache.set(cache_key, response, ttl=timedelta(minutes=5))
        
        return PoolResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pool {pool_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve pool data"
        )

@app.get(
    "/api/v1/pools/{pool_id}/strategy",
    response_model=StrategyResponse,
    tags=["Strategy"]
)
async def get_pool_strategy(
    pool_id: str,
    risk_level: float = Query(0.5, ge=0, le=1),
    cache: CacheManager = Depends(get_cache)
) -> StrategyResponse:
    """Get AI-generated strategy for pool"""
    try:
        # Check cache
        cache_key = f"strategy:{pool_id}:{risk_level}"
        cached = await cache.get(cache_key)
        if cached:
            return StrategyResponse(**cached)

        # Get pool data
        pool = await meteora.get_pool(pool_id)
        if not pool:
            raise HTTPException(
                status_code=404,
                detail=f"Pool {pool_id} not found"
            )

        # Generate strategy
        strategy = await ai_agent.suggest_strategy(
            token_pair=f"{pool.token_a.symbol}/{pool.token_b.symbol}",
            user_risk=risk_level
        )
        
        response = {
            "pool_id": pool_id,
            "token_pair": f"{pool.token_a.symbol}/{pool.token_b.symbol}",
            "optimal_range": strategy["range"],
            "suggested_fee": strategy.get("fee", pool.fees.base_fee_rate),
            "confidence_score": strategy.get("confidence", 0.8)
        }
        
        # Cache strategy
        await cache.set(cache_key, response, ttl=timedelta(minutes=15))
        
        return StrategyResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating strategy for pool {pool_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate strategy"
        )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc)
        ).dict()
    )