"""Configuration settings for Dexter Liquidity Management System"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Centralized configuration management"""
    
    # Environment
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # API Keys and RPC URLs
    ALCHEMY_API_KEY: str = os.getenv('ALCHEMY_API_KEY', '')
    BASESCAN_API_KEY: str = os.getenv('BASESCAN_API_KEY', '')
    BASE_RPC_URL: str = os.getenv('BASE_RPC_URL', f'https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}')
    
    # Subgraph URLs
    UNISWAP_V3_SUBGRAPH_URL: str = os.getenv(
        'UNISWAP_V3_SUBGRAPH_URL', 
        'https://api.thegraph.com/subgraphs/name/messari/uniswap-v3-base'
    )
    METEORA_SUBGRAPH_URL: str = os.getenv('METEORA_SUBGRAPH_URL', '')
    
    # Database Configuration
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://dexter:dexter@localhost:5432/dexter')
    DB_POOL_SIZE: int = int(os.getenv('DB_POOL_SIZE', '10'))
    DB_MAX_OVERFLOW: int = int(os.getenv('DB_MAX_OVERFLOW', '20'))
    
    # Redis Configuration
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
    
    # Performance and Execution
    PARALLEL_WORKERS: int = int(os.getenv('PARALLEL_WORKERS', '4'))
    MAX_RETRY_ATTEMPTS: int = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
    RETRY_DELAY: float = float(os.getenv('RETRY_DELAY', '1.0'))
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # Risk Management
    MAX_SLIPPAGE: float = float(os.getenv('MAX_SLIPPAGE', '0.005'))  # 0.5%
    MAX_GAS_PRICE_GWEI: int = int(os.getenv('MAX_GAS_PRICE_GWEI', '50'))
    MIN_LIQUIDITY_USD: float = float(os.getenv('MIN_LIQUIDITY_USD', '1000'))
    
    # Agent Configuration
    CONSERVATIVE_MIN_LIQUIDITY: float = float(os.getenv('CONSERVATIVE_MIN_LIQUIDITY', '100000'))
    CONSERVATIVE_MAX_VOLATILITY: float = float(os.getenv('CONSERVATIVE_MAX_VOLATILITY', '0.15'))
    
    AGGRESSIVE_MIN_LIQUIDITY: float = float(os.getenv('AGGRESSIVE_MIN_LIQUIDITY', '50000'))
    AGGRESSIVE_MAX_VOLATILITY: float = float(os.getenv('AGGRESSIVE_MAX_VOLATILITY', '0.30'))
    
    HYPER_AGGRESSIVE_MIN_LIQUIDITY: float = float(os.getenv('HYPER_AGGRESSIVE_MIN_LIQUIDITY', '25000'))
    HYPER_AGGRESSIVE_MAX_VOLATILITY: float = float(os.getenv('HYPER_AGGRESSIVE_MAX_VOLATILITY', '1.0'))
    
    # Monitoring and Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    PROMETHEUS_PORT: int = int(os.getenv('PROMETHEUS_PORT', '9092'))
    GRAFANA_PORT: int = int(os.getenv('GRAFANA_PORT', '3002'))
    
    # File Paths
    LOG_FILE_PATH: Path = Path(os.getenv('LOG_FILE_PATH', './logs/dexter.log'))
    DATA_DIR: Path = Path(os.getenv('DATA_DIR', './data'))
    
    # Validation
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings"""
        errors = []
        
        # Check required API keys
        if not cls.ALCHEMY_API_KEY:
            errors.append("ALCHEMY_API_KEY is required")
        
        if not cls.BASESCAN_API_KEY:
            errors.append("BASESCAN_API_KEY is required for production")
        
        # Validate numeric ranges
        if cls.PARALLEL_WORKERS < 1 or cls.PARALLEL_WORKERS > 32:
            errors.append("PARALLEL_WORKERS must be between 1 and 32")
        
        if cls.MAX_SLIPPAGE < 0 or cls.MAX_SLIPPAGE > 0.1:
            errors.append("MAX_SLIPPAGE must be between 0 and 0.1 (10%)")
        
        # Create directories
        cls.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @classmethod
    def get_agent_config(cls, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        configs = {
            'conservative': {
                'min_liquidity': cls.CONSERVATIVE_MIN_LIQUIDITY,
                'max_volatility': cls.CONSERVATIVE_MAX_VOLATILITY,
                'fee_tier': 0.0001,  # 0.01%
                'rebalance_threshold': 0.02,  # 2%
            },
            'aggressive': {
                'min_liquidity': cls.AGGRESSIVE_MIN_LIQUIDITY,
                'max_volatility': cls.AGGRESSIVE_MAX_VOLATILITY,
                'fee_tier': 0.0005,  # 0.05%
                'rebalance_threshold': 0.05,  # 5%
            },
            'hyper_aggressive': {
                'min_liquidity': cls.HYPER_AGGRESSIVE_MIN_LIQUIDITY,
                'max_volatility': cls.HYPER_AGGRESSIVE_MAX_VOLATILITY,
                'fee_tier': 0.003,  # 0.3%
                'rebalance_threshold': 0.10,  # 10%
            }
        }
        
        return configs.get(agent_type.lower(), {})
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return cls.ENVIRONMENT.lower() == 'production'
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return cls.ENVIRONMENT.lower() == 'development'


# Initialize and validate settings on import
try:
    Settings.validate()
except ValueError as e:
    if Settings.is_production():
        raise
    else:
        print(f"Configuration warning (development mode): {e}")

# Legacy compatibility
ALCHEMY_KEY = Settings.ALCHEMY_API_KEY
BASE_RPC_URL = Settings.BASE_RPC_URL
SUBGRAPH_URL = Settings.UNISWAP_V3_SUBGRAPH_URL