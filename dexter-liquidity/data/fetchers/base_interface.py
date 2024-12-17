from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, NamedTuple
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FetcherError(Exception):
    """Base class for fetcher exceptions"""
    pass

class NetworkError(FetcherError):
    """Raised when network operations fail"""
    pass

class DataValidationError(FetcherError):
    """Raised when data validation fails"""
    pass

@dataclass
class PoolStats:
    """Common pool statistics across different DEXes"""
    address: str
    token0_address: str
    token1_address: str
    token0_symbol: str
    token1_symbol: str
    tvl: Decimal
    volume_24h: Decimal
    fee_rate: Decimal
    liquidity: Decimal
    price: Decimal
    timestamp: datetime

@dataclass
class PositionInfo:
    """Position information for liquidity provision"""
    lower_price: Decimal
    upper_price: Decimal
    optimal_liquidity: Decimal
    estimated_apy: Decimal
    risk_score: Decimal

class LiquidityPoolFetcher(ABC):
    """Abstract base class for liquidity pool fetchers"""
    
    def __init__(self, retry_attempts: int = 3, timeout: int = 30):
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self._last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests
        logger.info(f"{self.__class__.__name__}: Initialized with {retry_attempts} retry attempts")

    @abstractmethod
    async def get_pool_data(self, pool_address: str) -> Optional[PoolStats]:
        """Fetch current pool data"""
        pass

    @abstractmethod
    async def get_all_pools(self) -> List[PoolStats]:
        """Fetch all available pools"""
        pass

    @abstractmethod
    async def calculate_optimal_position(
        self,
        pool_address: str,
        amount: Decimal,
        max_slippage: Decimal
    ) -> Optional[PositionInfo]:
        """Calculate optimal position for given pool and amount"""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        pool_address: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Fetch historical pool data"""
        pass

    async def _make_request(self, request_func, *args, **kwargs) -> Optional[Dict]:
        """Make a request with retry logic and rate limiting"""
        for attempt in range(self.retry_attempts):
            try:
                # Rate limiting
                await self._respect_rate_limit()
                
                # Execute request with timeout
                async with asyncio.timeout(self.timeout):
                    result = await request_func(*args, **kwargs)
                    self._last_request_time = asyncio.get_event_loop().time()
                    return result
                    
            except asyncio.TimeoutError:
                logger.warning(f"{self.__class__.__name__}: Request timeout on attempt {attempt + 1}")
                if attempt == self.retry_attempts - 1:
                    raise NetworkError("Request timed out after all retry attempts")
                    
            except Exception as e:
                logger.error(f"{self.__class__.__name__}: Request failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise
                    
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)

    async def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = asyncio.get_event_loop().time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)

    def _validate_pool_data(self, data: Dict) -> bool:
        """Validate pool data"""
        required_fields = {'address', 'tvl', 'volume_24h', 'liquidity'}
        return all(field in data for field in required_fields)

    async def _handle_error(self, error: Exception, context: str):
        """Centralized error handling"""
        logger.error(f"{self.__class__.__name__}: Error in {context}: {str(error)}")
        
        if isinstance(error, (asyncio.TimeoutError, aiohttp.ClientError)):
            raise NetworkError(f"Network error in {context}: {str(error)}")
        elif isinstance(error, (ValueError, KeyError)):
            raise DataValidationError(f"Data validation error in {context}: {str(error)}")
        else:
            raise FetcherError(f"Unknown error in {context}: {str(error)}")
