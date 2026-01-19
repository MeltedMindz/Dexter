"""SQLAlchemy models for DexBrain database.

These models define the database schema and are used by Alembic for migrations.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Strategy(Base):
    """Strategy records for token pairs."""
    __tablename__ = 'strategies'

    id = Column(Integer, primary_key=True)
    token_pair = Column(String(50), nullable=False, index=True)
    strategy_details = Column(JSONB, nullable=False)
    performance_metrics = Column(JSONB, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class PerformanceMetric(Base):
    """Performance metrics for agents."""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    token_pair = Column(String(50), nullable=False, index=True)
    agent_id = Column(String(50), nullable=False, index=True)
    metrics = Column(JSONB, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    """Dataset registry for ML training data."""
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    file_path = Column(Text, nullable=False)
    added_on = Column(DateTime, default=datetime.utcnow)


class AgentRegistration(Base):
    """Agent registration records for the intelligence network."""
    __tablename__ = 'agent_registrations'

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), unique=True, nullable=False, index=True)
    api_key_hash = Column(String(256), nullable=False)
    agent_metadata = Column('metadata', JSONB)  # Column named 'metadata' in DB, 'agent_metadata' in Python
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime)
    is_active = Column(Integer, default=1)  # Boolean as int for compatibility


class Insight(Base):
    """Knowledge base insights."""
    __tablename__ = 'insights'

    id = Column(Integer, primary_key=True)
    category = Column(String(100), nullable=False, index=True)
    data = Column(JSONB, nullable=False)
    quality_score = Column(Integer, default=0)
    source_agent_id = Column(String(100), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class VaultMetric(Base):
    """Vault performance metrics."""
    __tablename__ = 'vault_metrics'

    id = Column(Integer, primary_key=True)
    vault_address = Column(String(42), nullable=False, index=True)
    total_value_locked = Column(Integer, default=0)  # In smallest units
    total_fees_24h = Column(Integer, default=0)
    apr = Column(Integer, default=0)  # Basis points
    compounds_executed = Column(Integer, default=0)
    recorded_at = Column(DateTime, default=datetime.utcnow)
