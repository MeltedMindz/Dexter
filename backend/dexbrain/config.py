import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration for DexBrain system"""
    
    # Blockchain RPC Endpoints  
    ETHEREUM_RPC: str = os.getenv('ETHEREUM_RPC', 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID')
    BASE_RPC: str = os.getenv('BASE_RPC', 'https://mainnet-rpc.linkpool.io/')
    
    # ML and Data Configuration
    MODEL_STORAGE_PATH: Path = Path(os.getenv('MODEL_STORAGE_PATH', './model_storage/'))
    KNOWLEDGE_DB_PATH: Path = Path(os.getenv('KNOWLEDGE_DB_PATH', './knowledge_base/'))
    
    # Database Configuration
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://dexter:dexter@localhost:5432/dexter')
    
    # Redis Configuration
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    
    # Logging and Monitoring
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8080'))
    API_DEBUG: bool = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    # Global Intelligence Network Settings
    MAX_AGENTS_PER_REQUEST: int = int(os.getenv('MAX_AGENTS_PER_REQUEST', '1000'))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    DATA_QUALITY_THRESHOLD: float = float(os.getenv('DATA_QUALITY_THRESHOLD', '60.0'))
    PERFORMANCE_SCORING_ENABLED: bool = os.getenv('PERFORMANCE_SCORING_ENABLED', 'True').lower() == 'true'
    
    # API Key Settings
    API_KEY_LENGTH: int = int(os.getenv('API_KEY_LENGTH', '32'))
    API_KEY_PREFIX: str = os.getenv('API_KEY_PREFIX', 'dx_')
    
    # Performance Settings
    PARALLEL_WORKERS: int = int(os.getenv('PARALLEL_WORKERS', '4'))
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))
    
    # Intelligence Feed Settings
    DEFAULT_INSIGHTS_LIMIT: int = int(os.getenv('DEFAULT_INSIGHTS_LIMIT', '100'))
    MAX_INSIGHTS_LIMIT: int = int(os.getenv('MAX_INSIGHTS_LIMIT', '1000'))
    PREDICTION_POOL_LIMIT: int = int(os.getenv('PREDICTION_POOL_LIMIT', '5'))
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings"""
        # Ensure required directories exist
        cls.MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        cls.KNOWLEDGE_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Validate RPC endpoints
        if 'YOUR_PROJECT_ID' in cls.ETHEREUM_RPC:
            raise ValueError("Please configure a valid Ethereum RPC endpoint")
            
    @classmethod
    def get_env(cls) -> str:
        """Get current environment"""
        return os.getenv('ENVIRONMENT', 'development')