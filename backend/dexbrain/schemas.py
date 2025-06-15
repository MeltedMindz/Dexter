"""Data Standardization Schemas for DexBrain Network"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import json


class Blockchain(Enum):
    """Supported blockchains"""
    ETHEREUM = "ethereum"
    BASE = "base"
    SOLANA = "solana"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


class DEXProtocol(Enum):
    """Supported DEX protocols"""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    UNISWAP_V4 = "uniswap_v4"
    SUSHISWAP = "sushiswap"
    PANCAKESWAP = "pancakeswap"
    METEORA = "meteora"
    RAYDIUM = "raydium"
    ORCA = "orca"
    CURVE = "curve"
    BALANCER = "balancer"


class PositionStatus(Enum):
    """Liquidity position status"""
    ACTIVE = "active"
    CLOSED = "closed"
    PENDING = "pending"
    FAILED = "failed"


@dataclass
class TokenInfo:
    """Standardized token information"""
    address: str
    symbol: str
    name: str
    decimals: int
    blockchain: str
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PoolInfo:
    """Standardized pool information"""
    address: str
    blockchain: str
    dex_protocol: str
    token0: TokenInfo
    token1: TokenInfo
    fee_tier: float
    total_liquidity_usd: float
    volume_24h_usd: float
    volume_7d_usd: Optional[float] = None
    fees_24h_usd: Optional[float] = None
    apr: Optional[float] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['token0'] = self.token0.to_dict()
        data['token1'] = self.token1.to_dict()
        return data


@dataclass
class LiquidityPosition:
    """Standardized liquidity position"""
    position_id: str
    agent_id: str
    pool: PoolInfo
    status: str
    liquidity_amount: float
    token0_amount: float
    token1_amount: float
    position_value_usd: float
    entry_price: float
    current_price: Optional[float] = None
    price_range_lower: Optional[float] = None
    price_range_upper: Optional[float] = None
    fees_earned_usd: float = 0.0
    impermanent_loss_usd: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    closed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['pool'] = self.pool.to_dict()
        return data


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics"""
    position_id: str
    agent_id: str
    total_return_usd: float
    total_return_percent: float
    fees_earned_usd: float
    impermanent_loss_usd: float
    net_profit_usd: float
    apr: float
    duration_hours: float
    gas_costs_usd: float
    slippage_percent: float
    win: bool
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketConditions:
    """Standardized market conditions"""
    blockchain: str
    timestamp: str
    gas_price_gwei: Optional[float] = None
    gas_price_usd: Optional[float] = None
    block_number: Optional[int] = None
    network_congestion: Optional[str] = None  # "low", "medium", "high"
    total_value_locked_usd: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSubmission:
    """Standardized agent data submission"""
    agent_id: str
    submission_id: str
    timestamp: str
    blockchain: str
    dex_protocol: str
    positions: List[LiquidityPosition]
    performance_metrics: List[PerformanceMetrics]
    market_conditions: Optional[MarketConditions] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'submission_id': self.submission_id,
            'timestamp': self.timestamp,
            'blockchain': self.blockchain,
            'dex_protocol': self.dex_protocol,
            'positions': [pos.to_dict() for pos in self.positions],
            'performance_metrics': [perf.to_dict() for perf in self.performance_metrics],
            'market_conditions': self.market_conditions.to_dict() if self.market_conditions else None,
            'metadata': self.metadata or {}
        }


class DataValidator:
    """Validates data submissions against schemas"""
    
    @staticmethod
    def validate_token_info(data: Dict[str, Any]) -> bool:
        """Validate token information"""
        required_fields = ['address', 'symbol', 'name', 'decimals', 'blockchain']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate types
        if not isinstance(data['decimals'], int) or data['decimals'] < 0:
            return False
        
        if data['blockchain'].lower() not in [b.value for b in Blockchain]:
            return False
        
        return True
    
    @staticmethod
    def validate_pool_info(data: Dict[str, Any]) -> bool:
        """Validate pool information"""
        required_fields = ['address', 'blockchain', 'dex_protocol', 'token0', 'token1', 'fee_tier']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate nested token info
        if not DataValidator.validate_token_info(data['token0']):
            return False
        if not DataValidator.validate_token_info(data['token1']):
            return False
        
        # Validate blockchain and DEX protocol
        if data['blockchain'].lower() not in [b.value for b in Blockchain]:
            return False
        
        if data['dex_protocol'].lower() not in [d.value for d in DEXProtocol]:
            return False
        
        # Validate numeric fields
        numeric_fields = ['fee_tier', 'total_liquidity_usd', 'volume_24h_usd']
        for field in numeric_fields:
            if field in data and not isinstance(data[field], (int, float)):
                return False
        
        return True
    
    @staticmethod
    def validate_liquidity_position(data: Dict[str, Any]) -> bool:
        """Validate liquidity position"""
        required_fields = [
            'position_id', 'agent_id', 'pool', 'status', 'liquidity_amount',
            'token0_amount', 'token1_amount', 'position_value_usd', 'entry_price'
        ]
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate pool info
        if not DataValidator.validate_pool_info(data['pool']):
            return False
        
        # Validate status
        if data['status'].lower() not in [s.value for s in PositionStatus]:
            return False
        
        # Validate numeric fields
        numeric_fields = [
            'liquidity_amount', 'token0_amount', 'token1_amount',
            'position_value_usd', 'entry_price', 'fees_earned_usd', 'impermanent_loss_usd'
        ]
        for field in numeric_fields:
            if field in data and not isinstance(data[field], (int, float)):
                return False
        
        return True
    
    @staticmethod
    def validate_performance_metrics(data: Dict[str, Any]) -> bool:
        """Validate performance metrics"""
        required_fields = [
            'position_id', 'agent_id', 'total_return_usd', 'total_return_percent',
            'fees_earned_usd', 'impermanent_loss_usd', 'net_profit_usd',
            'apr', 'duration_hours', 'timestamp'
        ]
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate boolean field
        if 'win' in data and not isinstance(data['win'], bool):
            return False
        
        # Validate numeric fields
        numeric_fields = [
            'total_return_usd', 'total_return_percent', 'fees_earned_usd',
            'impermanent_loss_usd', 'net_profit_usd', 'apr', 'duration_hours'
        ]
        for field in numeric_fields:
            if field in data and not isinstance(data[field], (int, float)):
                return False
        
        return True
    
    @staticmethod
    def validate_agent_submission(data: Dict[str, Any]) -> bool:
        """Validate complete agent submission"""
        required_fields = [
            'agent_id', 'submission_id', 'timestamp', 'blockchain',
            'dex_protocol', 'positions', 'performance_metrics'
        ]
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate blockchain and DEX protocol
        if data['blockchain'].lower() not in [b.value for b in Blockchain]:
            return False
        
        if data['dex_protocol'].lower() not in [d.value for d in DEXProtocol]:
            return False
        
        # Validate positions array
        if not isinstance(data['positions'], list):
            return False
        
        for position in data['positions']:
            if not DataValidator.validate_liquidity_position(position):
                return False
        
        # Validate performance metrics array
        if not isinstance(data['performance_metrics'], list):
            return False
        
        for metrics in data['performance_metrics']:
            if not DataValidator.validate_performance_metrics(metrics):
                return False
        
        return True


class DataTransformer:
    """Transforms data between different formats"""
    
    @staticmethod
    def normalize_blockchain_name(blockchain: str) -> str:
        """Normalize blockchain name to standard format"""
        blockchain_mapping = {
            'eth': 'ethereum',
            'matic': 'polygon',
            'arb': 'arbitrum',
            'op': 'optimism',
            'sol': 'solana'
        }
        
        normalized = blockchain.lower().strip()
        return blockchain_mapping.get(normalized, normalized)
    
    @staticmethod
    def normalize_dex_protocol(protocol: str) -> str:
        """Normalize DEX protocol name to standard format"""
        protocol_mapping = {
            'uniswap': 'uniswap_v3',  # Default to v3
            'uni': 'uniswap_v3',
            'sushi': 'sushiswap',
            'pancake': 'pancakeswap',
            'ray': 'raydium'
        }
        
        normalized = protocol.lower().strip()
        return protocol_mapping.get(normalized, normalized)
    
    @staticmethod
    def convert_to_standard_units(
        amount: float,
        decimals: int,
        to_human_readable: bool = True
    ) -> float:
        """Convert between raw blockchain units and human readable units"""
        if to_human_readable:
            return amount / (10 ** decimals)
        else:
            return amount * (10 ** decimals)
    
    @staticmethod
    def create_submission_from_raw_data(
        agent_id: str,
        raw_data: Dict[str, Any]
    ) -> Optional[AgentSubmission]:
        """Create standardized submission from raw agent data"""
        try:
            # Generate submission ID
            submission_id = f"{agent_id}_{int(datetime.now().timestamp())}"
            
            # Transform raw data to standardized format
            blockchain = DataTransformer.normalize_blockchain_name(
                raw_data.get('blockchain', 'ethereum')
            )
            
            dex_protocol = DataTransformer.normalize_dex_protocol(
                raw_data.get('dex_protocol', 'uniswap_v3')
            )
            
            # Convert positions (this would be more complex in practice)
            positions = []
            if 'positions' in raw_data:
                for pos_data in raw_data['positions']:
                    # This is a simplified conversion - real implementation would be more robust
                    positions.append(pos_data)
            
            # Convert performance metrics
            performance_metrics = []
            if 'performance_metrics' in raw_data:
                for perf_data in raw_data['performance_metrics']:
                    performance_metrics.append(perf_data)
            
            submission = AgentSubmission(
                agent_id=agent_id,
                submission_id=submission_id,
                timestamp=datetime.now().isoformat(),
                blockchain=blockchain,
                dex_protocol=dex_protocol,
                positions=positions,
                performance_metrics=performance_metrics,
                metadata=raw_data.get('metadata', {})
            )
            
            return submission
            
        except Exception as e:
            print(f"Error creating submission: {e}")
            return None


# Schema definitions for API documentation
SCHEMAS = {
    'TokenInfo': {
        'type': 'object',
        'required': ['address', 'symbol', 'name', 'decimals', 'blockchain'],
        'properties': {
            'address': {'type': 'string'},
            'symbol': {'type': 'string'},
            'name': {'type': 'string'},
            'decimals': {'type': 'integer', 'minimum': 0},
            'blockchain': {'type': 'string', 'enum': [b.value for b in Blockchain]},
            'price_usd': {'type': 'number'},
            'market_cap': {'type': 'number'}
        }
    },
    'PoolInfo': {
        'type': 'object',
        'required': ['address', 'blockchain', 'dex_protocol', 'token0', 'token1', 'fee_tier'],
        'properties': {
            'address': {'type': 'string'},
            'blockchain': {'type': 'string', 'enum': [b.value for b in Blockchain]},
            'dex_protocol': {'type': 'string', 'enum': [d.value for d in DEXProtocol]},
            'token0': {'$ref': '#/components/schemas/TokenInfo'},
            'token1': {'$ref': '#/components/schemas/TokenInfo'},
            'fee_tier': {'type': 'number'},
            'total_liquidity_usd': {'type': 'number'},
            'volume_24h_usd': {'type': 'number'},
            'volume_7d_usd': {'type': 'number'},
            'fees_24h_usd': {'type': 'number'},
            'apr': {'type': 'number'},
            'created_at': {'type': 'string', 'format': 'date-time'}
        }
    },
    'LiquidityPosition': {
        'type': 'object',
        'required': [
            'position_id', 'agent_id', 'pool', 'status', 'liquidity_amount',
            'token0_amount', 'token1_amount', 'position_value_usd', 'entry_price'
        ],
        'properties': {
            'position_id': {'type': 'string'},
            'agent_id': {'type': 'string'},
            'pool': {'$ref': '#/components/schemas/PoolInfo'},
            'status': {'type': 'string', 'enum': [s.value for s in PositionStatus]},
            'liquidity_amount': {'type': 'number'},
            'token0_amount': {'type': 'number'},
            'token1_amount': {'type': 'number'},
            'position_value_usd': {'type': 'number'},
            'entry_price': {'type': 'number'},
            'current_price': {'type': 'number'},
            'price_range_lower': {'type': 'number'},
            'price_range_upper': {'type': 'number'},
            'fees_earned_usd': {'type': 'number'},
            'impermanent_loss_usd': {'type': 'number'},
            'created_at': {'type': 'string', 'format': 'date-time'},
            'updated_at': {'type': 'string', 'format': 'date-time'},
            'closed_at': {'type': 'string', 'format': 'date-time'}
        }
    }
}