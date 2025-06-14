from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging


class BlockchainConnector(ABC):
    """Abstract base class for blockchain connectors"""
    
    def __init__(self, rpc_endpoint: str):
        self.rpc_endpoint = rpc_endpoint
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = None
    
    @abstractmethod
    async def connect(self) -> Any:
        """Establish connection to blockchain network
        
        Returns:
            Connected client instance
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to blockchain network"""
        pass
    
    @abstractmethod
    async def fetch_liquidity_data(self, pool_address: str) -> Optional[Dict[str, Any]]:
        """Retrieve liquidity data for a specific pool
        
        Args:
            pool_address: Address of the liquidity pool
            
        Returns:
            Dictionary containing liquidity metrics or None if error
        """
        pass
    
    @abstractmethod
    async def fetch_token_prices(self, tokens: List[str]) -> Dict[str, float]:
        """Get current token prices
        
        Args:
            tokens: List of token addresses/symbols
            
        Returns:
            Dictionary mapping token to price in USD
        """
        pass
    
    @abstractmethod
    async def fetch_pool_volume(self, pool_address: str, period: str = '24h') -> float:
        """Get trading volume for a pool
        
        Args:
            pool_address: Address of the liquidity pool
            period: Time period for volume calculation
            
        Returns:
            Volume in USD
        """
        pass
    
    @abstractmethod
    async def fetch_pool_fees(self, pool_address: str) -> Dict[str, Any]:
        """Get fee information for a pool
        
        Args:
            pool_address: Address of the liquidity pool
            
        Returns:
            Dictionary containing fee tier and collected fees
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()