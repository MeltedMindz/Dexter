import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import time
from .base_interface import LiquidityPoolFetcher

logger = logging.getLogger(__name__)

@dataclass
class MeteoraPoolData:
    pool_address: str
    name: str
    token_mints: List[str]
    token_amounts: List[float]
    tvl: float
    volume_24h: float
    fee_volume: float
    apy: float

class MeteoraFetcher(LiquidityPoolFetcher):
    def __init__(self, api_url: str = "https://api.meteora.ag/"):
        super().__init__()
        self.api_url = api_url
        
    async def get_pool_data(self, pool_address: str) -> Optional[MeteoraPoolData]:
        """Fetch data for a specific pool"""
        logger.info(f"meteora_fetcher.py: Fetching data for pool {pool_address}")
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/pool/{pool_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._convert_to_pool_data(data)
                    else:
                        logger.error(f"meteora_fetcher.py: Failed to fetch pool data: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"meteora_fetcher.py: Error fetching pool data: {str(e)}")
            return None
            
    async def get_all_pools(self) -> List[MeteoraPoolData]:
        """Fetch all available pools"""
        logger.info("meteora_fetcher.py: Fetching all pools")
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/pools"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [self._convert_to_pool_data(pool) for pool in data]
                    else:
                        logger.error(f"meteora_fetcher.py: Failed to fetch pools: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"meteora_fetcher.py: Error fetching pools: {str(e)}")
            return []
            
    def _convert_to_pool_data(self, data: Dict) -> MeteoraPoolData:
        """Convert API response to PoolData object"""
        return MeteoraPoolData(
            pool_address=data["pool_address"],
            name=data.get("pool_name", "Unknown"),
            token_mints=data.get("pool_token_mints", []),
            token_amounts=data.get("pool_token_amounts", []),
            tvl=float(data.get("pool_tvl", 0)),
            volume_24h=float(data.get("trading_volume", 0)),
            fee_volume=float(data.get("fee_volume", 0)),
            apy=float(data.get("trade_apy", 0))
        )