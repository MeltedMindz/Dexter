"""
Configuration Settings Module

Uses Pydantic for settings management with validation.
Loads from environment variables with sane defaults.
"""

from typing import Any, Dict, List
from pydantic import (
    BaseSettings,
    PostgresDsn,
    RedisDsn,
    HttpUrl,
    Field,
    validator
)
from functools import lru_cache
import logging
from pathlib import Path

class DatabaseSettings(BaseSettings):
    """Database connection settings"""
    URL: PostgresDsn = Field(
        "postgresql://postgres:password123@localhost:5432/dexbrain",
        description="PostgreSQL connection URL"
    )
    POOL_SIZE: int = Field(5, ge=1, le=20)
    MAX_OVERFLOW: int = Field(10, ge=0)
    POOL_TIMEOUT: int = Field(30, ge=1)

    class Config:
        env_prefix = "DB_"

class RedisSettings(BaseSettings):
    """Redis cache settings"""
    URL: RedisDsn = Field(
        "redis://localhost:6379/0",
        description="Redis connection URL"
    )
    POOL_SIZE: int = Field(5, ge=1, le=20)
    DEFAULT_TTL: int = Field(3600, ge=1)  # 1 hour default
    PREFIX: str = "dexbrain:"

    class Config:
        env_prefix = "REDIS_"

class SolanaSettings(BaseSettings):
    """Solana connection settings"""
    RPC_URL: HttpUrl = Field(
        "https://api.mainnet-beta.solana.com",
        description="Solana RPC endpoint"
    )
    COMMITMENT: str = Field("confirmed")
    TIMEOUT: int = Field(30, ge=1)
    MAX_RETRIES: int = Field(3, ge=0)

    class Config:
        env_prefix = "SOLANA_"

class APISettings(BaseSettings):
    """API server settings"""
    HOST: str = Field("0.0.0.0")
    PORT: int = Field(8000, ge=1, le=65535)
    WORKERS: int = Field(4, ge=1)
    DEBUG: bool = Field(False)
    PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_prefix = "API_"

class MeteoraSettings(BaseSettings):
    """Meteora protocol settings"""
    POOL_CONFIGS: Dict[str, Dict[str, Any]] = Field(
        default={
            "3W2HKgUa96Z69zzG3LK1g8KdcRAWzAttiLiHfYnKuPw5": {
                "token_a_symbol": "SOL",
                "token_b_symbol": "USDC",
                "token_a_decimals": 9,
                "token_b_decimals": 6
            }
        }
    )
    UPDATE_INTERVAL: int = Field(60, ge=1)  # seconds
    
    class Config:
        env_prefix = "METEORA_"

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    LEVEL: str = Field("INFO")
    FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    DIRECTORY: Path = Field("logs")

    @validator("LEVEL")
    def validate_level(cls, v: str) -> str:
        """Validate logging level"""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    class Config:
        env_prefix = "LOG_"

class Settings(BaseSettings):
    """Main settings class combining all configurations"""
    # Environment
    ENV: str = Field("development")
    DEBUG: bool = Field(False)
    
    # Component configurations
    db: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    solana: SolanaSettings = SolanaSettings()
    api: APISettings = APISettings()
    meteora: MeteoraSettings = MeteoraSettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def configure_logging(self) -> None:
        """Configure logging based on settings"""
        # Create logs directory if it doesn't exist
        self.logging.DIRECTORY.mkdir(exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.logging.LEVEL),
            format=self.logging.FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.logging.DIRECTORY / "dexbrain.log"
                )
            ]
        )

@lru_cache()
def get_settings() -> Settings:
    """Create cached settings instance"""
    settings = Settings()
    settings.configure_logging()
    return settings

# Create global settings instance
settings = get_settings()