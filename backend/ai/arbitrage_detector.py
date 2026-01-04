"""
Cross-Chain Arbitrage Detection and Opportunity Scanning System
Advanced arbitrage detection across multiple chains and DEXs with real-time monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import json
import math
from decimal import Decimal, getcontext
import web3
from web3 import Web3
import numpy as np

# Set decimal precision for financial calculations
getcontext().prec = 50

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChainId(Enum):
    """Supported blockchain networks"""
    ETHEREUM = 1
    BASE = 8453
    ARBITRUM = 42161
    OPTIMISM = 10
    POLYGON = 137
    BSC = 56
    AVALANCHE = 43114

class DEXType(Enum):
    """Supported DEX types"""
    UNISWAP_V3 = "uniswap_v3"
    UNISWAP_V2 = "uniswap_v2"
    SUSHISWAP = "sushiswap"
    CURVE = "curve"
    BALANCER = "balancer"
    PANCAKESWAP = "pancakeswap"
    TRADERJOE = "traderjoe"

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    SIMPLE = "simple"  # Basic price difference
    TRIANGULAR = "triangular"  # Three-asset cycle
    CROSS_CHAIN = "cross_chain"  # Cross-chain price difference
    FLASH_LOAN = "flash_loan"  # Flash loan arbitrage
    MEV = "mev"  # MEV opportunity

@dataclass
class TokenInfo:
    """Token information across chains"""
    symbol: str
    name: str
    addresses: Dict[ChainId, str]
    decimals: Dict[ChainId, int]
    is_stable: bool = False
    coingecko_id: Optional[str] = None

@dataclass
class PoolInfo:
    """DEX pool information"""
    pool_address: str
    dex_type: DEXType
    chain_id: ChainId
    token0: str
    token1: str
    fee: int
    liquidity: Decimal
    reserve0: Decimal
    reserve1: Decimal
    price: Decimal
    last_updated: datetime

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity details"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    profit_usd: Decimal
    profit_percentage: Decimal
    confidence_score: float
    
    # Path information
    buy_chain: ChainId
    sell_chain: ChainId
    buy_dex: DEXType
    sell_dex: DEXType
    buy_pool: str
    sell_pool: str
    
    # Token and amount details
    base_token: str
    quote_token: str
    optimal_amount: Decimal
    
    # Pricing information
    buy_price: Decimal
    sell_price: Decimal
    gas_cost_usd: Decimal
    bridge_cost_usd: Decimal
    
    # Timing and execution
    execution_time: datetime
    expiry_time: datetime
    complexity: int  # 1-10 scale
    
    # Risk assessment
    slippage_risk: float
    liquidity_risk: float
    bridge_risk: float
    overall_risk: float

@dataclass
class ChainConfig:
    """Configuration for each blockchain"""
    chain_id: ChainId
    rpc_url: str
    gas_price_gwei: float
    gas_token_price_usd: float
    bridge_contracts: Dict[ChainId, str]
    average_block_time: float
    bridge_time_minutes: Dict[ChainId, int]

class PriceOracle:
    """Multi-source price oracle for accurate pricing"""
    
    def __init__(self):
        self.price_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self.cache_duration = timedelta(minutes=1)
        
    async def get_token_price(self, token_address: str, chain_id: ChainId) -> Optional[Decimal]:
        """Get token price from multiple sources"""
        cache_key = f"{token_address}_{chain_id.value}"
        
        # Check cache
        if cache_key in self.price_cache:
            price, timestamp = self.price_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return price
        
        # Fetch from multiple sources
        price = await self._fetch_price_from_sources(token_address, chain_id)
        
        if price:
            self.price_cache[cache_key] = (price, datetime.now())
        
        return price
    
    async def _fetch_price_from_sources(self, token_address: str, chain_id: ChainId) -> Optional[Decimal]:
        """Fetch price from multiple sources and return median"""
        prices = []
        
        # Source 1: CoinGecko
        try:
            coingecko_price = await self._fetch_coingecko_price(token_address, chain_id)
            if coingecko_price:
                prices.append(coingecko_price)
        except Exception as e:
            logger.warning(f"CoinGecko price fetch failed: {e}")
        
        # Source 2: On-chain TWAP
        try:
            twap_price = await self._fetch_twap_price(token_address, chain_id)
            if twap_price:
                prices.append(twap_price)
        except Exception as e:
            logger.warning(f"TWAP price fetch failed: {e}")
        
        # Source 3: DEX aggregator
        try:
            dex_price = await self._fetch_dex_price(token_address, chain_id)
            if dex_price:
                prices.append(dex_price)
        except Exception as e:
            logger.warning(f"DEX price fetch failed: {e}")
        
        if not prices:
            return None
        
        # Return median price for robustness
        prices.sort()
        n = len(prices)
        if n % 2 == 0:
            return (prices[n//2 - 1] + prices[n//2]) / 2
        else:
            return prices[n//2]
    
    async def _fetch_coingecko_price(self, token_address: str, chain_id: ChainId) -> Optional[Decimal]:
        """Fetch price from CoinGecko API"""
        # Implementation would use actual CoinGecko API
        return Decimal("1.0")  # Placeholder
    
    async def _fetch_twap_price(self, token_address: str, chain_id: ChainId) -> Optional[Decimal]:
        """Fetch TWAP price from Uniswap V3"""
        # Implementation would use actual TWAP oracle
        return Decimal("1.0")  # Placeholder
    
    async def _fetch_dex_price(self, token_address: str, chain_id: ChainId) -> Optional[Decimal]:
        """Fetch price from DEX aggregator"""
        # Implementation would use 1inch or similar
        return Decimal("1.0")  # Placeholder

class GasPriceEstimator:
    """Gas price estimation across chains"""
    
    def __init__(self):
        self.gas_cache: Dict[ChainId, Tuple[float, datetime]] = {}
        self.cache_duration = timedelta(minutes=2)
    
    async def estimate_gas_cost(self, chain_id: ChainId, gas_units: int) -> Decimal:
        """Estimate gas cost in USD"""
        gas_price_gwei = await self.get_gas_price(chain_id)
        gas_token_price = await self.get_gas_token_price(chain_id)
        
        gas_cost_eth = (gas_price_gwei * gas_units) / 1e9  # Convert to ETH
        gas_cost_usd = gas_cost_eth * gas_token_price
        
        return Decimal(str(gas_cost_usd))
    
    async def get_gas_price(self, chain_id: ChainId) -> float:
        """Get current gas price in gwei"""
        cache_key = chain_id
        
        if cache_key in self.gas_cache:
            price, timestamp = self.gas_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return price
        
        # Fetch current gas price
        gas_price = await self._fetch_gas_price(chain_id)
        self.gas_cache[cache_key] = (gas_price, datetime.now())
        
        return gas_price
    
    async def _fetch_gas_price(self, chain_id: ChainId) -> float:
        """Fetch current gas price from network"""
        # Implementation would use actual RPC calls
        base_prices = {
            ChainId.ETHEREUM: 20.0,
            ChainId.BASE: 0.1,
            ChainId.ARBITRUM: 0.1,
            ChainId.OPTIMISM: 0.1,
            ChainId.POLYGON: 30.0,
            ChainId.BSC: 3.0,
            ChainId.AVALANCHE: 25.0
        }
        return base_prices.get(chain_id, 10.0)
    
    async def get_gas_token_price(self, chain_id: ChainId) -> float:
        """Get gas token price in USD"""
        # Implementation would fetch actual prices
        prices = {
            ChainId.ETHEREUM: 2000.0,
            ChainId.BASE: 2000.0,
            ChainId.ARBITRUM: 2000.0,
            ChainId.OPTIMISM: 2000.0,
            ChainId.POLYGON: 0.8,
            ChainId.BSC: 300.0,
            ChainId.AVALANCHE: 25.0
        }
        return prices.get(chain_id, 2000.0)

class LiquidityAnalyzer:
    """Analyze liquidity depth and slippage across DEXs"""
    
    async def analyze_slippage(self, 
                              pool_address: str, 
                              amount: Decimal, 
                              chain_id: ChainId,
                              dex_type: DEXType) -> Tuple[Decimal, float]:
        """Analyze price slippage for given trade size"""
        try:
            # Get pool liquidity data
            pool_data = await self._get_pool_data(pool_address, chain_id, dex_type)
            
            if not pool_data:
                return Decimal("0"), 1.0  # No data, assume high slippage
            
            # Calculate slippage based on pool type
            if dex_type == DEXType.UNISWAP_V3:
                slippage = await self._calculate_v3_slippage(pool_data, amount)
            elif dex_type == DEXType.CURVE:
                slippage = await self._calculate_curve_slippage(pool_data, amount)
            else:
                slippage = await self._calculate_v2_slippage(pool_data, amount)
            
            # Convert slippage to risk score (0-1)
            risk_score = min(1.0, float(slippage) / 0.05)  # 5% = max risk
            
            return slippage, risk_score
            
        except Exception as e:
            logger.error(f"Slippage analysis failed: {e}")
            return Decimal("0.05"), 1.0  # Default to 5% slippage, max risk
    
    async def _get_pool_data(self, pool_address: str, chain_id: ChainId, dex_type: DEXType) -> Optional[Dict]:
        """Get pool data from blockchain"""
        # Implementation would use actual RPC calls
        return {
            "liquidity": Decimal("1000000"),
            "reserve0": Decimal("500000"),
            "reserve1": Decimal("500000"),
            "fee": 3000
        }
    
    async def _calculate_v3_slippage(self, pool_data: Dict, amount: Decimal) -> Decimal:
        """Calculate slippage for Uniswap V3 pool"""
        liquidity = pool_data["liquidity"]
        if liquidity == 0:
            return Decimal("1.0")  # 100% slippage
        
        # Simplified slippage calculation
        slippage = amount / liquidity
        return min(slippage, Decimal("1.0"))
    
    async def _calculate_curve_slippage(self, pool_data: Dict, amount: Decimal) -> Decimal:
        """Calculate slippage for Curve pool"""
        # Curve has different slippage characteristics
        total_liquidity = pool_data["reserve0"] + pool_data["reserve1"]
        if total_liquidity == 0:
            return Decimal("1.0")
        
        slippage = (amount / total_liquidity) * Decimal("0.5")  # Curve is more efficient
        return min(slippage, Decimal("1.0"))
    
    async def _calculate_v2_slippage(self, pool_data: Dict, amount: Decimal) -> Decimal:
        """Calculate slippage for Uniswap V2 style pool"""
        reserve = min(pool_data["reserve0"], pool_data["reserve1"])
        if reserve == 0:
            return Decimal("1.0")
        
        # V2 constant product formula
        slippage = amount / (reserve + amount)
        return min(slippage, Decimal("1.0"))

class ArbitrageDetector:
    """Main arbitrage detection and opportunity scanning engine"""
    
    def __init__(self):
        self.price_oracle = PriceOracle()
        self.gas_estimator = GasPriceEstimator()
        self.liquidity_analyzer = LiquidityAnalyzer()
        
        # Configuration
        self.min_profit_usd = Decimal("10")
        self.min_profit_percentage = Decimal("0.01")  # 1%
        self.max_execution_time = timedelta(minutes=10)
        self.supported_tokens = self._load_supported_tokens()
        self.chain_configs = self._load_chain_configs()
        
        # Monitoring data
        self.opportunities: List[ArbitrageOpportunity] = []
        self.pools: Dict[str, PoolInfo] = {}
        self.last_scan_time = datetime.now()
        
    def _load_supported_tokens(self) -> Dict[str, TokenInfo]:
        """Load supported token configurations"""
        return {
            "USDC": TokenInfo(
                symbol="USDC",
                name="USD Coin",
                addresses={
                    ChainId.ETHEREUM: "0xA0b86a33E6441b8C2D10D9b0C6C66E3Bb9e86d66",
                    ChainId.BASE: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    ChainId.ARBITRUM: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                    ChainId.POLYGON: "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
                },
                decimals={
                    ChainId.ETHEREUM: 6,
                    ChainId.BASE: 6,
                    ChainId.ARBITRUM: 6,
                    ChainId.POLYGON: 6
                },
                is_stable=True,
                coingecko_id="usd-coin"
            ),
            "WETH": TokenInfo(
                symbol="WETH",
                name="Wrapped Ether",
                addresses={
                    ChainId.ETHEREUM: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                    ChainId.BASE: "0x4200000000000000000000000000000000000006",
                    ChainId.ARBITRUM: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    ChainId.POLYGON: "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619"
                },
                decimals={
                    ChainId.ETHEREUM: 18,
                    ChainId.BASE: 18,
                    ChainId.ARBITRUM: 18,
                    ChainId.POLYGON: 18
                },
                is_stable=False,
                coingecko_id="ethereum"
            )
        }
    
    def _load_chain_configs(self) -> Dict[ChainId, ChainConfig]:
        """Load blockchain configuration"""
        return {
            ChainId.ETHEREUM: ChainConfig(
                chain_id=ChainId.ETHEREUM,
                rpc_url="https://eth-mainnet.alchemyapi.io/v2/YOUR_KEY",
                gas_price_gwei=20.0,
                gas_token_price_usd=2000.0,
                bridge_contracts={},
                average_block_time=12.0,
                bridge_time_minutes={}
            ),
            ChainId.BASE: ChainConfig(
                chain_id=ChainId.BASE,
                rpc_url="https://base-mainnet.alchemyapi.io/v2/YOUR_KEY",
                gas_price_gwei=0.1,
                gas_token_price_usd=2000.0,
                bridge_contracts={},
                average_block_time=2.0,
                bridge_time_minutes={ChainId.ETHEREUM: 10}
            )
        }
    
    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """Main opportunity scanning function"""
        logger.info("Starting arbitrage opportunity scan...")
        
        opportunities = []
        
        # Scan different types of arbitrage
        simple_ops = await self._scan_simple_arbitrage()
        triangular_ops = await self._scan_triangular_arbitrage()
        cross_chain_ops = await self._scan_cross_chain_arbitrage()
        
        opportunities.extend(simple_ops)
        opportunities.extend(triangular_ops)
        opportunities.extend(cross_chain_ops)
        
        # Filter and rank opportunities
        filtered_ops = self._filter_opportunities(opportunities)
        ranked_ops = self._rank_opportunities(filtered_ops)
        
        self.opportunities = ranked_ops
        self.last_scan_time = datetime.now()
        
        logger.info(f"Found {len(ranked_ops)} profitable arbitrage opportunities")
        return ranked_ops
    
    async def _scan_simple_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Scan for simple arbitrage opportunities within same chain"""
        opportunities = []
        
        for token_symbol, token_info in self.supported_tokens.items():
            for chain_id in token_info.addresses.keys():
                try:
                    chain_ops = await self._scan_chain_simple_arbitrage(token_symbol, chain_id)
                    opportunities.extend(chain_ops)
                except Exception as e:
                    logger.error(f"Simple arbitrage scan failed for {token_symbol} on {chain_id}: {e}")
        
        return opportunities
    
    async def _scan_chain_simple_arbitrage(self, token_symbol: str, chain_id: ChainId) -> List[ArbitrageOpportunity]:
        """Scan simple arbitrage on specific chain"""
        opportunities = []
        
        # Get all pools for this token on this chain
        pools = await self._get_token_pools(token_symbol, chain_id)
        
        # Compare prices between all pool pairs
        for i, pool1 in enumerate(pools):
            for pool2 in pools[i+1:]:
                if pool1.dex_type != pool2.dex_type:
                    opportunity = await self._analyze_simple_opportunity(pool1, pool2)
                    if opportunity and opportunity.profit_usd >= self.min_profit_usd:
                        opportunities.append(opportunity)
        
        return opportunities
    
    async def _scan_triangular_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Scan for triangular arbitrage opportunities"""
        opportunities = []
        
        # Triangular arbitrage: Token A -> Token B -> Token C -> Token A
        for chain_id in ChainId:
            try:
                chain_ops = await self._scan_chain_triangular_arbitrage(chain_id)
                opportunities.extend(chain_ops)
            except Exception as e:
                logger.error(f"Triangular arbitrage scan failed on {chain_id}: {e}")
        
        return opportunities
    
    async def _scan_chain_triangular_arbitrage(self, chain_id: ChainId) -> List[ArbitrageOpportunity]:
        """Scan triangular arbitrage on specific chain"""
        # Implementation would find profitable triangular paths
        return []  # Placeholder
    
    async def _scan_cross_chain_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Scan for cross-chain arbitrage opportunities"""
        opportunities = []
        
        for token_symbol, token_info in self.supported_tokens.items():
            chains = list(token_info.addresses.keys())
            
            for i, chain1 in enumerate(chains):
                for chain2 in chains[i+1:]:
                    try:
                        opportunity = await self._analyze_cross_chain_opportunity(
                            token_symbol, chain1, chain2
                        )
                        if opportunity and opportunity.profit_usd >= self.min_profit_usd:
                            opportunities.append(opportunity)
                    except Exception as e:
                        logger.error(f"Cross-chain analysis failed for {token_symbol}: {e}")
        
        return opportunities
    
    async def _analyze_simple_opportunity(self, pool1: PoolInfo, pool2: PoolInfo) -> Optional[ArbitrageOpportunity]:
        """Analyze simple arbitrage opportunity between two pools"""
        try:
            # Determine buy and sell pools
            if pool1.price < pool2.price:
                buy_pool, sell_pool = pool1, pool2
            else:
                buy_pool, sell_pool = pool2, pool1
            
            price_diff = sell_pool.price - buy_pool.price
            profit_percentage = price_diff / buy_pool.price
            
            if profit_percentage < self.min_profit_percentage:
                return None
            
            # Calculate optimal trade size
            optimal_amount = await self._calculate_optimal_amount(buy_pool, sell_pool)
            
            # Estimate costs
            gas_cost = await self._estimate_transaction_costs(buy_pool.chain_id, 2)  # 2 transactions
            
            # Calculate slippage risks
            buy_slippage, buy_risk = await self.liquidity_analyzer.analyze_slippage(
                buy_pool.pool_address, optimal_amount, buy_pool.chain_id, buy_pool.dex_type
            )
            sell_slippage, sell_risk = await self.liquidity_analyzer.analyze_slippage(
                sell_pool.pool_address, optimal_amount, sell_pool.chain_id, sell_pool.dex_type
            )
            
            # Calculate net profit
            gross_profit = optimal_amount * price_diff
            net_profit = gross_profit - gas_cost
            
            if net_profit <= 0:
                return None
            
            # Generate opportunity
            opportunity = ArbitrageOpportunity(
                opportunity_id=self._generate_opportunity_id(),
                arbitrage_type=ArbitrageType.SIMPLE,
                profit_usd=net_profit,
                profit_percentage=profit_percentage,
                confidence_score=self._calculate_confidence_score(buy_risk, sell_risk),
                buy_chain=buy_pool.chain_id,
                sell_chain=sell_pool.chain_id,
                buy_dex=buy_pool.dex_type,
                sell_dex=sell_pool.dex_type,
                buy_pool=buy_pool.pool_address,
                sell_pool=sell_pool.pool_address,
                base_token=buy_pool.token0,  # Simplified
                quote_token=buy_pool.token1,
                optimal_amount=optimal_amount,
                buy_price=buy_pool.price,
                sell_price=sell_pool.price,
                gas_cost_usd=gas_cost,
                bridge_cost_usd=Decimal("0"),
                execution_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(minutes=5),
                complexity=2,
                slippage_risk=max(buy_risk, sell_risk),
                liquidity_risk=self._assess_liquidity_risk(buy_pool, sell_pool),
                bridge_risk=0.0,
                overall_risk=self._calculate_overall_risk(buy_risk, sell_risk, 0.0)
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing simple opportunity: {e}")
            return None
    
    async def _analyze_cross_chain_opportunity(self, 
                                              token_symbol: str, 
                                              chain1: ChainId, 
                                              chain2: ChainId) -> Optional[ArbitrageOpportunity]:
        """Analyze cross-chain arbitrage opportunity"""
        try:
            # Get best prices on each chain
            price1 = await self._get_best_price(token_symbol, chain1)
            price2 = await self._get_best_price(token_symbol, chain2)
            
            if not price1 or not price2:
                return None
            
            # Determine direction
            if price1 < price2:
                buy_chain, sell_chain = chain1, chain2
                buy_price, sell_price = price1, price2
            else:
                buy_chain, sell_chain = chain2, chain1
                buy_price, sell_price = price2, price1
            
            price_diff = sell_price - buy_price
            profit_percentage = price_diff / buy_price
            
            if profit_percentage < self.min_profit_percentage:
                return None
            
            # Estimate bridge costs and time
            bridge_cost = await self._estimate_bridge_cost(buy_chain, sell_chain)
            bridge_time = self._get_bridge_time(buy_chain, sell_chain)
            
            # Calculate optimal amount considering bridge costs
            optimal_amount = Decimal("1000")  # Simplified
            
            # Calculate total costs
            gas_cost = await self._estimate_transaction_costs(buy_chain, 1)
            gas_cost += await self._estimate_transaction_costs(sell_chain, 1)
            total_cost = gas_cost + bridge_cost
            
            # Calculate net profit
            gross_profit = optimal_amount * price_diff
            net_profit = gross_profit - total_cost
            
            if net_profit <= 0:
                return None
            
            # Assess risks
            bridge_risk = self._assess_bridge_risk(buy_chain, sell_chain)
            time_risk = bridge_time / 60.0  # Convert to hours for risk calculation
            
            opportunity = ArbitrageOpportunity(
                opportunity_id=self._generate_opportunity_id(),
                arbitrage_type=ArbitrageType.CROSS_CHAIN,
                profit_usd=net_profit,
                profit_percentage=profit_percentage,
                confidence_score=self._calculate_confidence_score(0.2, 0.2, bridge_risk),
                buy_chain=buy_chain,
                sell_chain=sell_chain,
                buy_dex=DEXType.UNISWAP_V3,  # Simplified
                sell_dex=DEXType.UNISWAP_V3,
                buy_pool="",  # Would be filled with actual pool
                sell_pool="",
                base_token=token_symbol,
                quote_token="USDC",
                optimal_amount=optimal_amount,
                buy_price=buy_price,
                sell_price=sell_price,
                gas_cost_usd=gas_cost,
                bridge_cost_usd=bridge_cost,
                execution_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(minutes=bridge_time + 5),
                complexity=8,
                slippage_risk=0.3,
                liquidity_risk=0.3,
                bridge_risk=bridge_risk,
                overall_risk=self._calculate_overall_risk(0.3, 0.3, bridge_risk)
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing cross-chain opportunity: {e}")
            return None
    
    # Helper methods
    async def _get_token_pools(self, token_symbol: str, chain_id: ChainId) -> List[PoolInfo]:
        """Get all pools for a token on a specific chain"""
        # Implementation would query actual pools
        return []  # Placeholder
    
    async def _get_best_price(self, token_symbol: str, chain_id: ChainId) -> Optional[Decimal]:
        """Get best price for token on chain"""
        token_info = self.supported_tokens.get(token_symbol)
        if not token_info or chain_id not in token_info.addresses:
            return None
        
        token_address = token_info.addresses[chain_id]
        return await self.price_oracle.get_token_price(token_address, chain_id)
    
    async def _calculate_optimal_amount(self, buy_pool: PoolInfo, sell_pool: PoolInfo) -> Decimal:
        """Calculate optimal trade amount to maximize profit"""
        # Simplified calculation - would use more sophisticated optimization
        min_liquidity = min(buy_pool.liquidity, sell_pool.liquidity)
        return min_liquidity * Decimal("0.01")  # 1% of smaller pool
    
    async def _estimate_transaction_costs(self, chain_id: ChainId, tx_count: int) -> Decimal:
        """Estimate transaction costs for operations"""
        gas_per_tx = 200000  # Estimated gas per transaction
        total_gas = gas_per_tx * tx_count
        return await self.gas_estimator.estimate_gas_cost(chain_id, total_gas)
    
    async def _estimate_bridge_cost(self, from_chain: ChainId, to_chain: ChainId) -> Decimal:
        """Estimate bridge cost between chains"""
        # Implementation would use actual bridge cost estimates
        base_costs = {
            (ChainId.ETHEREUM, ChainId.BASE): Decimal("10"),
            (ChainId.BASE, ChainId.ETHEREUM): Decimal("5"),
            (ChainId.ETHEREUM, ChainId.ARBITRUM): Decimal("15"),
            (ChainId.ARBITRUM, ChainId.ETHEREUM): Decimal("8")
        }
        return base_costs.get((from_chain, to_chain), Decimal("20"))
    
    def _get_bridge_time(self, from_chain: ChainId, to_chain: ChainId) -> int:
        """Get bridge time in minutes"""
        # Implementation would use actual bridge time estimates
        times = {
            (ChainId.ETHEREUM, ChainId.BASE): 10,
            (ChainId.BASE, ChainId.ETHEREUM): 10,
            (ChainId.ETHEREUM, ChainId.ARBITRUM): 15,
            (ChainId.ARBITRUM, ChainId.ETHEREUM): 15
        }
        return times.get((from_chain, to_chain), 30)
    
    def _assess_bridge_risk(self, from_chain: ChainId, to_chain: ChainId) -> float:
        """Assess bridge risk between chains"""
        # Lower risk for well-established bridges
        risk_matrix = {
            (ChainId.ETHEREUM, ChainId.BASE): 0.1,
            (ChainId.BASE, ChainId.ETHEREUM): 0.1,
            (ChainId.ETHEREUM, ChainId.ARBITRUM): 0.15,
            (ChainId.ARBITRUM, ChainId.ETHEREUM): 0.15
        }
        return risk_matrix.get((from_chain, to_chain), 0.5)
    
    def _assess_liquidity_risk(self, pool1: PoolInfo, pool2: PoolInfo) -> float:
        """Assess liquidity risk for arbitrage"""
        min_liquidity = min(pool1.liquidity, pool2.liquidity)
        if min_liquidity > Decimal("1000000"):  # $1M+
            return 0.1
        elif min_liquidity > Decimal("100000"):  # $100K+
            return 0.3
        else:
            return 0.7
    
    def _calculate_confidence_score(self, *risk_factors: float) -> float:
        """Calculate confidence score from risk factors"""
        if not risk_factors:
            return 0.5
        
        avg_risk = sum(risk_factors) / len(risk_factors)
        return max(0.0, min(1.0, 1.0 - avg_risk))
    
    def _calculate_overall_risk(self, *risk_factors: float) -> float:
        """Calculate overall risk score"""
        if not risk_factors:
            return 0.5
        
        # Use maximum risk as overall risk (conservative approach)
        return min(1.0, max(risk_factors))
    
    def _generate_opportunity_id(self) -> str:
        """Generate unique opportunity ID"""
        timestamp = int(time.time() * 1000000)
        return f"ARB_{timestamp}"
    
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on criteria"""
        filtered = []
        
        for opp in opportunities:
            # Minimum profit filter
            if opp.profit_usd < self.min_profit_usd:
                continue
            
            # Minimum profit percentage filter
            if opp.profit_percentage < self.min_profit_percentage:
                continue
            
            # Risk filter (exclude very high risk)
            if opp.overall_risk > 0.8:
                continue
            
            # Time filter (exclude opportunities that take too long)
            if opp.expiry_time - opp.execution_time > self.max_execution_time:
                continue
            
            filtered.append(opp)
        
        return filtered
    
    def _rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Rank opportunities by attractiveness"""
        def opportunity_score(opp: ArbitrageOpportunity) -> float:
            # Weighted score: profit (50%), confidence (30%), speed (20%)
            profit_score = min(1.0, float(opp.profit_usd) / 1000.0)  # Normalize to $1000
            confidence_score = opp.confidence_score
            speed_score = 1.0 / (opp.complexity / 10.0)  # Lower complexity = higher speed score
            
            return (profit_score * 0.5 + 
                   confidence_score * 0.3 + 
                   speed_score * 0.2)
        
        return sorted(opportunities, key=opportunity_score, reverse=True)
    
    def get_opportunity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive opportunity statistics"""
        if not self.opportunities:
            return {}
        
        total_profit = sum(opp.profit_usd for opp in self.opportunities)
        avg_profit = total_profit / len(self.opportunities)
        
        arbitrage_types = {}
        chain_distribution = {}
        
        for opp in self.opportunities:
            # Type distribution
            arb_type = opp.arbitrage_type.value
            arbitrage_types[arb_type] = arbitrage_types.get(arb_type, 0) + 1
            
            # Chain distribution
            buy_chain = opp.buy_chain.name
            sell_chain = opp.sell_chain.name
            chain_key = f"{buy_chain}->{sell_chain}"
            chain_distribution[chain_key] = chain_distribution.get(chain_key, 0) + 1
        
        return {
            "total_opportunities": len(self.opportunities),
            "total_profit_usd": float(total_profit),
            "average_profit_usd": float(avg_profit),
            "arbitrage_types": arbitrage_types,
            "chain_distribution": chain_distribution,
            "last_scan_time": self.last_scan_time.isoformat(),
            "high_confidence_count": len([opp for opp in self.opportunities if opp.confidence_score > 0.7]),
            "low_risk_count": len([opp for opp in self.opportunities if opp.overall_risk < 0.3])
        }

# Singleton instance
_arbitrage_detector = None

def get_arbitrage_detector() -> ArbitrageDetector:
    """Get singleton arbitrage detector instance"""
    global _arbitrage_detector
    if _arbitrage_detector is None:
        _arbitrage_detector = ArbitrageDetector()
    return _arbitrage_detector

# Monitoring functions
async def continuous_arbitrage_monitoring(scan_interval: int = 60):
    """Continuously monitor for arbitrage opportunities"""
    detector = get_arbitrage_detector()
    
    while True:
        try:
            start_time = time.time()
            opportunities = await detector.scan_opportunities()
            scan_duration = time.time() - start_time
            
            if opportunities:
                logger.info(f"Found {len(opportunities)} opportunities in {scan_duration:.2f}s")
                for opp in opportunities[:3]:  # Log top 3
                    logger.info(f"  {opp.arbitrage_type.value}: ${float(opp.profit_usd):.2f} "
                              f"({float(opp.profit_percentage)*100:.2f}%) "
                              f"confidence: {opp.confidence_score:.2f}")
            
            await asyncio.sleep(scan_interval)
            
        except Exception as e:
            logger.error(f"Arbitrage monitoring error: {e}")
            await asyncio.sleep(scan_interval)

if __name__ == "__main__":
    # Example usage
    async def main():
        detector = get_arbitrage_detector()
        opportunities = await detector.scan_opportunities()
        
        print(f"Found {len(opportunities)} arbitrage opportunities")
        for opp in opportunities:
            print(f"  {opp.arbitrage_type.value}: ${float(opp.profit_usd):.2f}")
    
    asyncio.run(main())