import asyncio
import json
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from web3.middleware import async_cache_middleware
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import aiohttp
from dataclasses import dataclass
from cachetools import TTLCache, LRUCache
import logging
import websockets
import time
from prometheus_client import Counter, Gauge, Histogram
from decimal import Decimal

# Prometheus metrics
POOL_REQUESTS = Counter('pool_requests_total', 'Total number of pool data requests')
CACHE_HITS = Counter('cache_hits_total', 'Total number of cache hits')
FETCH_DURATION = Histogram('fetch_duration_seconds', 'Time spent fetching pool data')
POOL_TVL = Gauge('pool_tvl_dollars', 'Pool TVL in USD', ['pool_address'])
POOL_VOLUME = Gauge('pool_volume_24h_dollars', 'Pool 24h volume in USD', ['pool_address'])

logger = logging.getLogger(__name__)

@dataclass
class BasePoolData:
    address: str
    token0: str
    token1: str
    fee: int
    liquidity: int
    sqrt_price_x96: int
    tick: int
    token0_price: float
    token1_price: float
    tvl_usd: float
    volume_24h: float
    fee_tier: int
    token0_decimals: int
    token1_decimals: int

class BaseFetcher:
    def __init__(
        self,
        alchemy_key: str,
        subgraph_url: str,
        cache_ttl: int = 300
    ):
        logger.info("base_fetcher.py: Initializing BaseFetcher")
        
        # Initialize Web3
        self.w3 = Web3(Web3.AsyncHTTPProvider(
            f"https://base-mainnet.g.alchemy.com/v2/{alchemy_key}"
        ))
        self.subgraph_url = subgraph_url
        
        # Enhanced caching system
        self.pool_cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.token_decimals_cache = LRUCache(maxsize=1000)  # Longer-term cache for token decimals
        self.fee_tier_cache = LRUCache(maxsize=1000)  # Cache for fee tiers
        
        # Price cache with shorter TTL for frequent updates
        self.price_cache = TTLCache(maxsize=100, ttl=60)  # 1-minute cache for prices
        
        self.w3.middleware_onion.add(async_cache_middleware)
        logger.info(f"base_fetcher.py: Initialized with cache TTL {cache_ttl}s")

    async def get_pool_data(self, pool_address: str) -> Optional[BasePoolData]:
        """Fetch comprehensive pool data with caching and metrics"""
        POOL_REQUESTS.inc()
        start_time = time.time()
        
        try:
            if pool_address in self.pool_cache:
                CACHE_HITS.inc()
                return self.pool_cache[pool_address]

            pool_contract = self._get_pool_contract(pool_address)
            data = await self._fetch_pool_data(pool_contract)
            
            if data:
                self.pool_cache[pool_address] = data
                
                # Update metrics
                POOL_TVL.labels(pool_address=pool_address).set(data.tvl_usd)
                POOL_VOLUME.labels(pool_address=pool_address).set(data.volume_24h)
                
                FETCH_DURATION.observe(time.time() - start_time)
                return data
                
            return None
            
        except Exception as e:
            logger.error(f"base_fetcher.py: Error fetching pool data: {str(e)}")
            return None

    async def _fetch_pool_data(self, contract) -> Optional[BasePoolData]:
        """Enhanced pool data fetching with all metrics"""
        try:
            # Fetch contract data concurrently
            slot0, liquidity, token0, token1, fee_tier = await asyncio.gather(
                contract.functions.slot0().call(),
                contract.functions.liquidity().call(),
                contract.functions.token0().call(),
                contract.functions.token1().call(),
                self._get_fee_tier(contract),
            )
            
            # Fetch token decimals (cached)
            token0_decimals, token1_decimals = await asyncio.gather(
                self._get_token_decimals(token0),
                self._get_token_decimals(token1)
            )
            
            # Calculate prices
            sqrt_price_x96 = slot0[0]
            tick = slot0[1]
            
            token0_price = self._calculate_price(sqrt_price_x96, True)
            token1_price = self._calculate_price(sqrt_price_x96, False)
            
            # Fetch TVL and volume from subgraph
            pool_metrics = await self._fetch_pool_metrics(contract.address)
            
            pool_data = BasePoolData(
                address=contract.address,
                token0=token0,
                token1=token1,
                fee=fee_tier,
                liquidity=liquidity,
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                token0_price=token0_price,
                token1_price=token1_price,
                tvl_usd=pool_metrics['tvl'],
                volume_24h=pool_metrics['volume'],
                fee_tier=fee_tier,
                token0_decimals=token0_decimals,
                token1_decimals=token1_decimals
            )
            
            return pool_data
            
        except Exception as e:
            logger.error(f"base_fetcher.py: Error in _fetch_pool_data: {str(e)}")
            return None

    async def _get_fee_tier(self, contract) -> int:
        """Get pool fee tier with caching"""
        address = contract.address
        
        if address in self.fee_tier_cache:
            return self.fee_tier_cache[address]
            
        try:
            fee = await contract.functions.fee().call()
            self.fee_tier_cache[address] = fee
            return fee
        except Exception as e:
            logger.error(f"base_fetcher.py: Error fetching fee tier: {str(e)}")
            return 0

    async def _get_token_decimals(self, token_address: str) -> int:
        """Get token decimals with caching"""
        if token_address in self.token_decimals_cache:
            return self.token_decimals_cache[token_address]
            
        try:
            token_contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(token_address),
                abi=[{
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"type": "uint8", "name": ""}],
                    "stateMutability": "view",
                    "type": "function"
                }]
            )
            
            decimals = await token_contract.functions.decimals().call()
            self.token_decimals_cache[token_address] = decimals
            return decimals
            
        except Exception as e:
            logger.error(f"base_fetcher.py: Error fetching token decimals: {str(e)}")
            return 18  # Default to 18 decimals

    async def _fetch_pool_metrics(self, pool_address: str) -> Dict[str, float]:
        """Fetch pool metrics from subgraph"""
        query = gql("""
            query PoolMetrics($id: String!) {
                pool(id: $id) {
                    totalValueLockedUSD
                    volumeUSD
                    feesUSD
                }
            }
        """)
        
        try:
            transport = AIOHTTPTransport(url=self.subgraph_url)
            async with Client(transport=transport) as client:
                result = await client.execute_async(query, {"id": pool_address.lower()})
                pool_data = result.get("pool", {})
                
                return {
                    'tvl': float(pool_data.get("totalValueLockedUSD", 0)),
                    'volume': float(pool_data.get("volumeUSD", 0))
                }
                
        except Exception as e:
            logger.error(f"base_fetcher.py: Error fetching pool metrics: {str(e)}")
            return {'tvl': 0, 'volume': 0}