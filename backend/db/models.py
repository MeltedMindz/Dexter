"""
Database Models Module

Provides SQLAlchemy models and database initialization for DLMM data storage.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, TIMESTAMP, JSON, Float, Boolean
from datetime import datetime
import logging
from typing import Optional, Dict, Any

from backend.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.db.POSTGRES_URL,
    echo=settings.DEBUG
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

class Strategy(Base):
    """DLMM strategy model"""
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    token_pair = Column(String(50), nullable=False)
    pool_id = Column(String(100), nullable=False)
    range_lower = Column(Float, nullable=False)
    range_upper = Column(Float, nullable=False)
    fee_tier = Column(Float, nullable=False)
    ai_generated = Column(Boolean, default=True)
    performance_data = Column(JSON, nullable=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

class PoolMetrics(Base):
    """Historical pool metrics"""
    __tablename__ = "pool_metrics"

    id = Column(Integer, primary_key=True)
    pool_id = Column(String(100), nullable=False)
    tvl_usd = Column(Float, nullable=False)
    daily_volume = Column(Float, nullable=False)
    fees_24h = Column(Float, nullable=False)
    metrics = Column(JSON, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

async def init_db():
    """Initialize database"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def get_db():
    """Dependency for database sessions"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def add_strategy(
    session: AsyncSession,
    token_pair: str,
    pool_id: str,
    range_lower: float,
    range_upper: float,
    fee_tier: float,
    performance_data: Optional[Dict] = None
) -> Strategy:
    """Add new strategy"""
    strategy = Strategy(
        token_pair=token_pair,
        pool_id=pool_id,
        range_lower=range_lower,
        range_upper=range_upper,
        fee_tier=fee_tier,
        performance_data=performance_data
    )
    session.add(strategy)
    await session.commit()
    return strategy

async def add_pool_metrics(
    session: AsyncSession,
    pool_id: str,
    tvl_usd: float,
    daily_volume: float,
    fees_24h: float,
    metrics: Dict[str, Any]
) -> PoolMetrics:
    """Add pool metrics"""
    pool_metrics = PoolMetrics(
        pool_id=pool_id,
        tvl_usd=tvl_usd,
        daily_volume=daily_volume,
        fees_24h=fees_24h,
        metrics=metrics
    )
    session.add(pool_metrics)
    await session.commit()
    return pool_metrics

async def get_latest_pool_metrics(
    session: AsyncSession,
    pool_id: str
) -> Optional[PoolMetrics]:
    """Get latest metrics for pool"""
    result = await session.execute(
        f"SELECT * FROM pool_metrics WHERE pool_id = '{pool_id}' "
        "ORDER BY timestamp DESC LIMIT 1"
    )
    row = result.fetchone()
    return PoolMetrics(**dict(row)) if row else None

async def get_pool_strategies(
    session: AsyncSession,
    pool_id: str,
    limit: int = 10
) -> list[Strategy]:
    """Get strategies for pool"""
    result = await session.execute(
        f"SELECT * FROM strategies WHERE pool_id = '{pool_id}' "
        f"ORDER BY timestamp DESC LIMIT {limit}"
    )
    return [Strategy(**dict(row)) for row in result.fetchall()]