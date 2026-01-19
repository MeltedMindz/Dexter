"""Initial database schema

Revision ID: 0001
Revises:
Create Date: 2026-01-19

This migration establishes the initial database schema for DexBrain,
matching the existing schema.sql definitions plus new tables for the
intelligence network.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create strategies table (matches existing schema.sql)
    op.create_table(
        'strategies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token_pair', sa.String(50), nullable=False),
        sa.Column('strategy_details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('performance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_strategies_token_pair', 'strategies', ['token_pair'])

    # Create performance_metrics table (matches existing schema.sql)
    op.create_table(
        'performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token_pair', sa.String(50), nullable=False),
        sa.Column('agent_id', sa.String(50), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_performance_metrics_token_pair', 'performance_metrics', ['token_pair'])
    op.create_index('ix_performance_metrics_agent_id', 'performance_metrics', ['agent_id'])

    # Create datasets table (matches existing schema.sql)
    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('added_on', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create agent_registrations table (new for intelligence network)
    op.create_table(
        'agent_registrations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sa.String(100), nullable=False),
        sa.Column('api_key_hash', sa.String(256), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_active', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Integer(), server_default='1', nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('agent_id')
    )
    op.create_index('ix_agent_registrations_agent_id', 'agent_registrations', ['agent_id'])

    # Create insights table (new for knowledge base)
    op.create_table(
        'insights',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('quality_score', sa.Integer(), server_default='0', nullable=True),
        sa.Column('source_agent_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_insights_category', 'insights', ['category'])
    op.create_index('ix_insights_source_agent_id', 'insights', ['source_agent_id'])

    # Create vault_metrics table (new for vault tracking)
    op.create_table(
        'vault_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('vault_address', sa.String(42), nullable=False),
        sa.Column('total_value_locked', sa.Integer(), server_default='0', nullable=True),
        sa.Column('total_fees_24h', sa.Integer(), server_default='0', nullable=True),
        sa.Column('apr', sa.Integer(), server_default='0', nullable=True),
        sa.Column('compounds_executed', sa.Integer(), server_default='0', nullable=True),
        sa.Column('recorded_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_vault_metrics_vault_address', 'vault_metrics', ['vault_address'])


def downgrade() -> None:
    op.drop_table('vault_metrics')
    op.drop_table('insights')
    op.drop_table('agent_registrations')
    op.drop_table('datasets')
    op.drop_table('performance_metrics')
    op.drop_table('strategies')
