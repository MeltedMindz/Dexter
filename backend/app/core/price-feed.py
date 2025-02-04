"""
Price Feed Service with multiple provider support and failover
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional, List
import aiohttp
import asyncio
from enum import Enum

class PriceSource(Enum):
    PYTH = "pyth"
    SWITCHBOARD = "switchboard"
    CHAINLINK = "chainlink"

class BasePriceFeed(ABC):
    @abstractmethod
    async def get_price(self, token_address: str) -> Optional[Decimal]:
        pass

class PythPriceFeed(BasePriceFeed):
    def __init__(self, network_url: str):
        self.url = network_url
        
    async def get_price(self, token_address: str) -> Optional[Decimal]:
        async with aiohttp.ClientSession() as session:
            try:
                response = await session.get(f"{self.url}/price/{token_address}")
                data = await response.json()
                return Decimal(str(data['price']))
            except Exception as e:
                logger.error(f"Pyth price fetch failed: {e}")
                return None

class SwitchboardPriceFeed(BasePriceFeed):
    def __init__(self, rpc_url: str):
        self.url = rpc_url
        
    async def get_price(self, token_address: str) -> Optional[Decimal]:
        # Implement Switchboard price fetching
        pass

class PriceFeedService:
    def __init__(self, config: Dict[str, str]):
        self.feeds: List[BasePriceFeed] = [
            PythPriceFeed(config['pyth_url']),
            SwitchboardPriceFeed(config['switchboard_url'])
        ]
        self.cache = {}
        self.cache_ttl = 30  # seconds

    async def get_price(self, token_address: str) -> Optional[Decimal]:
        # Check cache first
        cached = self.cache.get(token_address)
        if cached and (time.time() - cached['timestamp']) < self.cache_ttl:
            return cached['price']

        # Try each feed in order until we get a price
        for feed in self.feeds:
            price = await feed.get_price(token_address)
            if price:
                self.cache[token_address] = {
                    'price': price,
                    'timestamp': time.time()
                }
                return price

        return None

    async def get_multiple_prices(
        self,
        token_addresses: List[str]
    ) -> Dict[str, Optional[Decimal]]:
        tasks = [self.get_price(addr) for addr in token_addresses]
        prices = await asyncio.gather(*tasks)
        return dict(zip(token_addresses, prices))

    def register_feed(self, feed: BasePriceFeed) -> None:
        self.feeds.append(feed)
