from web3 import Web3
from web3.contract import Contract
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass
import logging
import math
from enum import Enum
from decimal import Decimal
import json
from eth_utils import to_checksum_address

logger = logging.getLogger(__name__)

# Uniswap V4 Contract Addresses on Base Network
CONTRACT_ADDRESSES = {
    "POOL_MANAGER": "0x498581ff718922c3f8e6a244956af099b2652b2b",
    "POSITION_MANAGER": "0x7c5f5a4bbd8fd63184577525326123b519429bdc", 
    "UNIVERSAL_ROUTER": "0x6ff5693b99212da76ad316178a184ab56d299b43",
    "QUOTER": "0x0d5e0f971ed27fbff6c2837bf31316121532048d",
    "STATE_VIEW": "0xa3c0c9b65bad0b08107aa264b0f3db444b867a71",
    "PERMIT2": "0x000000000022D473030F116dDEE9F6B43aC78BA3"
}

class VolatilityTimeframe(Enum):
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_24 = "24h"

@dataclass
class VolatilityWeight:
    timeframe: VolatilityTimeframe
    weight: float
    
    @staticmethod
    def get_default_weights() -> List['VolatilityWeight']:
        return [
            VolatilityWeight(VolatilityTimeframe.HOUR_1, 0.5),
            VolatilityWeight(VolatilityTimeframe.HOUR_4, 0.3),
            VolatilityWeight(VolatilityTimeframe.HOUR_24, 0.2)
        ]

class LiquidityRange(NamedTuple):
    lower_tick: int
    upper_tick: int
    optimal_liquidity: Decimal
    fee_estimate: Decimal

class UniswapV4Fetcher:
    """Enhanced Uniswap V4 data fetcher with singleton PoolManager architecture"""
    
    def __init__(
        self, 
        web3: Web3,
        volatility_weights: Optional[List[VolatilityWeight]] = None,
        k_factor: float = 2.5
    ):
        self.web3 = web3
        self.volatility_weights = volatility_weights or VolatilityWeight.get_default_weights()
        self.k_factor = k_factor
        
        # Initialize V4 contracts
        self.pool_manager = self._init_pool_manager()
        self.position_manager = self._init_position_manager() 
        self.universal_router = self._init_universal_router()
        self.quoter = self._init_quoter()
        
        logger.info(f"uniswap_v4_fetcher.py: Initialized UniswapV4Fetcher with V4 contracts")
        
    async def calculate_optimal_position(
        self,
        pool_address: str,
        current_price: Decimal,
        volatilities: Dict[VolatilityTimeframe, float],
        liquidity_depth: Decimal
    ) -> LiquidityRange:
        """Calculate optimal liquidity position range"""
        logger.info(f"uniswap_v4_fetcher.py: Calculating optimal position for pool {pool_address}")
        
        try:
            # Calculate weighted volatility
            weighted_variance = sum(
                weight.weight * (volatilities[weight.timeframe] ** 2)
                for weight in self.volatility_weights
            )
            weighted_volatility = math.sqrt(weighted_variance)
            logger.debug(f"uniswap_v4_fetcher.py: Calculated weighted volatility: {weighted_volatility}")
            
            # Calculate range
            range_size = weighted_volatility * self.k_factor
            tick_spacing = 60
            price_lower = current_price * Decimal(1 - range_size)
            price_upper = current_price * Decimal(1 + range_size)
            
            lower_tick = self._price_to_tick(price_lower, tick_spacing)
            upper_tick = self._price_to_tick(price_upper, tick_spacing)
            
            optimal_liquidity = self._calculate_optimal_liquidity(
                current_price,
                price_lower,
                price_upper,
                liquidity_depth
            )
            
            fee_estimate = self._estimate_fee_returns(
                weighted_volatility,
                optimal_liquidity,
                current_price
            )
            
            range_data = LiquidityRange(lower_tick, upper_tick, optimal_liquidity, fee_estimate)
            logger.info(f"uniswap_v4_fetcher.py: Calculated position range: ticks={lower_tick}-{upper_tick}")
            return range_data
            
        except Exception as e:
            logger.error(f"uniswap_v4_fetcher.py: Error calculating optimal position: {str(e)}")
            raise

    def _init_pool_manager(self) -> Contract:
        """Initialize PoolManager singleton contract"""
        abi = self._get_pool_manager_abi()
        address = to_checksum_address(CONTRACT_ADDRESSES["POOL_MANAGER"])
        return self.web3.eth.contract(address=address, abi=abi)
    
    def _init_position_manager(self) -> Contract:
        """Initialize PositionManager contract"""
        abi = self._get_position_manager_abi()
        address = to_checksum_address(CONTRACT_ADDRESSES["POSITION_MANAGER"])
        return self.web3.eth.contract(address=address, abi=abi)
    
    def _init_universal_router(self) -> Contract:
        """Initialize Universal Router contract"""
        abi = self._get_universal_router_abi()
        address = to_checksum_address(CONTRACT_ADDRESSES["UNIVERSAL_ROUTER"])
        return self.web3.eth.contract(address=address, abi=abi)
    
    def _init_quoter(self) -> Contract:
        """Initialize Quoter contract"""
        abi = self._get_quoter_abi()
        address = to_checksum_address(CONTRACT_ADDRESSES["QUOTER"])
        return self.web3.eth.contract(address=address, abi=abi)
    
    async def get_pool_state(self, pool_key: Dict[str, Any]) -> Dict[str, Any]:
        """Get pool state from PoolManager singleton"""
        try:
            # Call PoolManager.getSlot0() for pool state
            pool_id = self._compute_pool_id(pool_key)
            slot0 = await self.pool_manager.functions.getSlot0(pool_id).call()
            
            return {
                "sqrt_price_x96": slot0[0],
                "tick": slot0[1], 
                "protocol_fee": slot0[2],
                "lp_fee": slot0[3]
            }
        except Exception as e:
            logger.error(f"Error getting pool state: {str(e)}")
            raise
    
    async def mint_position(
        self,
        pool_key: Dict[str, Any],
        tick_lower: int,
        tick_upper: int,
        liquidity: int,
        amount0_max: int,
        amount1_max: int,
        recipient: str,
        deadline: int
    ) -> str:
        """Mint new liquidity position using PositionManager"""
        try:
            # Encode MINT_POSITION + SETTLE_PAIR actions
            actions = self._encode_actions(["MINT_POSITION", "SETTLE_PAIR"])
            
            # Encode parameters
            mint_params = [
                pool_key,
                tick_lower,
                tick_upper, 
                liquidity,
                amount0_max,
                amount1_max,
                recipient,
                b""  # hookData
            ]
            
            settle_params = [
                pool_key["currency0"],
                pool_key["currency1"]
            ]
            
            params = [mint_params, settle_params]
            
            # Submit transaction
            tx_hash = await self.position_manager.functions.modifyLiquidities(
                (actions, params),
                deadline
            ).transact()
            
            logger.info(f"Minted position with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error minting position: {str(e)}")
            raise
    
    async def increase_liquidity(
        self,
        token_id: int,
        liquidity: int,
        amount0_max: int,
        amount1_max: int,
        deadline: int
    ) -> str:
        """Increase liquidity in existing position"""
        try:
            actions = self._encode_actions(["INCREASE_LIQUIDITY", "SETTLE_PAIR"])
            
            increase_params = [token_id, liquidity, amount0_max, amount1_max, b""]
            settle_params = [0, 0]  # Will be filled with actual currencies
            
            params = [increase_params, settle_params]
            
            tx_hash = await self.position_manager.functions.modifyLiquidities(
                (actions, params),
                deadline
            ).transact()
            
            logger.info(f"Increased liquidity with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error increasing liquidity: {str(e)}")
            raise
    
    async def decrease_liquidity(
        self,
        token_id: int,
        liquidity: int,
        amount0_min: int,
        amount1_min: int,
        recipient: str,
        deadline: int
    ) -> str:
        """Decrease liquidity from position"""
        try:
            actions = self._encode_actions(["DECREASE_LIQUIDITY", "TAKE_PAIR"])
            
            decrease_params = [token_id, liquidity, amount0_min, amount1_min, b""]
            take_params = [0, 0, recipient]  # currencies filled later
            
            params = [decrease_params, take_params]
            
            tx_hash = await self.position_manager.functions.modifyLiquidities(
                (actions, params),
                deadline
            ).transact()
            
            logger.info(f"Decreased liquidity with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error decreasing liquidity: {str(e)}")
            raise
    
    async def collect_fees(
        self,
        token_id: int,
        recipient: str,
        deadline: int
    ) -> str:
        """Collect fees from position (decrease with 0 liquidity)"""
        try:
            actions = self._encode_actions(["DECREASE_LIQUIDITY", "TAKE_PAIR"])
            
            # Decrease with 0 liquidity to collect fees only
            decrease_params = [token_id, 0, 0, 0, b""]
            take_params = [0, 0, recipient]
            
            params = [decrease_params, take_params]
            
            tx_hash = await self.position_manager.functions.modifyLiquidities(
                (actions, params),
                deadline
            ).transact()
            
            logger.info(f"Collected fees with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error collecting fees: {str(e)}")
            raise
    
    async def burn_position(
        self,
        token_id: int,
        amount0_min: int,
        amount1_min: int,
        recipient: str,
        deadline: int
    ) -> str:
        """Burn entire position and withdraw funds"""
        try:
            actions = self._encode_actions(["BURN_POSITION", "TAKE_PAIR"])
            
            burn_params = [token_id, amount0_min, amount1_min, b""]
            take_params = [0, 0, recipient]
            
            params = [burn_params, take_params]
            
            tx_hash = await self.position_manager.functions.modifyLiquidities(
                (actions, params),
                deadline
            ).transact()
            
            logger.info(f"Burned position with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error burning position: {str(e)}")
            raise
    
    async def swap_exact_input(
        self,
        pool_key: Dict[str, Any],
        amount_in: int,
        amount_out_minimum: int,
        deadline: int
    ) -> str:
        """Execute swap using Universal Router"""
        try:
            # Encode V4_SWAP command
            commands = ["V4_SWAP"]
            
            swap_params = {
                "poolKey": pool_key,
                "zeroForOne": True,  # Swap token0 for token1
                "amountSpecified": amount_in,
                "sqrtPriceLimitX96": 0  # No limit
            }
            
            inputs = [swap_params]
            
            tx_hash = await self.universal_router.functions.execute(
                commands,
                inputs,
                deadline
            ).transact()
            
            logger.info(f"Executed swap with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error executing swap: {str(e)}")
            raise
    
    async def update_dynamic_fee(
        self,
        pool_key: Dict[str, Any],
        new_fee: int
    ) -> str:
        """Update dynamic fee for pool (if supported by hook)"""
        try:
            pool_id = self._compute_pool_id(pool_key)
            
            tx_hash = await self.pool_manager.functions.updateDynamicLPFee(
                pool_id,
                new_fee
            ).transact()
            
            logger.info(f"Updated dynamic fee with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error updating dynamic fee: {str(e)}")
            raise
    
    async def batch_liquidity_operations(
        self,
        operations: List[Dict[str, Any]],
        deadline: int
    ) -> str:
        """Execute multiple liquidity operations in batch"""
        try:
            all_actions = []
            all_params = []
            
            for op in operations:
                op_type = op["type"]
                
                if op_type == "mint":
                    all_actions.extend(["MINT_POSITION", "SETTLE_PAIR"])
                    all_params.extend([
                        [op["pool_key"], op["tick_lower"], op["tick_upper"], 
                         op["liquidity"], op["amount0_max"], op["amount1_max"],
                         op["recipient"], b""],
                        [op["pool_key"]["currency0"], op["pool_key"]["currency1"]]
                    ])
                    
                elif op_type == "increase":
                    all_actions.extend(["INCREASE_LIQUIDITY", "SETTLE_PAIR"])
                    all_params.extend([
                        [op["token_id"], op["liquidity"], op["amount0_max"], 
                         op["amount1_max"], b""],
                        [0, 0]  # currencies
                    ])
                    
                elif op_type == "decrease":
                    all_actions.extend(["DECREASE_LIQUIDITY", "TAKE_PAIR"])
                    all_params.extend([
                        [op["token_id"], op["liquidity"], op["amount0_min"],
                         op["amount1_min"], b""],
                        [0, 0, op["recipient"]]
                    ])
                    
                elif op_type == "burn":
                    all_actions.extend(["BURN_POSITION", "TAKE_PAIR"])
                    all_params.extend([
                        [op["token_id"], op["amount0_min"], op["amount1_min"], b""],
                        [0, 0, op["recipient"]]
                    ])
            
            # Add final CLOSE_CURRENCY action for efficiency
            all_actions.append("CLOSE_CURRENCY")
            all_params.append([])
            
            actions = self._encode_actions(all_actions)
            
            tx_hash = await self.position_manager.functions.modifyLiquidities(
                (actions, all_params),
                deadline
            ).transact()
            
            logger.info(f"Executed batch operations with tx: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error in batch operations: {str(e)}")
            raise
    
    async def quote_exact_input(
        self,
        pool_key: Dict[str, Any],
        amount_in: int,
        zero_for_one: bool
    ) -> Tuple[int, int]:
        """Get quote for exact input swap"""
        try:
            quote_params = {
                "poolKey": pool_key,
                "zeroForOne": zero_for_one,
                "exactAmount": amount_in,
                "sqrtPriceLimitX96": 0
            }
            
            result = await self.quoter.functions.quoteExactInputSingle(
                quote_params
            ).call()
            
            return result[0], result[1]  # amountOut, gasEstimate
            
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            raise
    
    def _compute_pool_id(self, pool_key: Dict[str, Any]) -> bytes:
        """Compute pool ID from pool key"""
        # Implementation depends on PoolKey struct encoding
        # This is a simplified version
        return self.web3.keccak(
            text=f"{pool_key['currency0']}{pool_key['currency1']}{pool_key['fee']}"
        )
    
    def _encode_actions(self, action_names: List[str]) -> bytes:
        """Encode action types for PositionManager"""
        action_map = {
            "MINT_POSITION": 0,
            "INCREASE_LIQUIDITY": 1,
            "DECREASE_LIQUIDITY": 2,
            "BURN_POSITION": 3,
            "SETTLE_PAIR": 4,
            "TAKE_PAIR": 5,
            "SWEEP": 6,
            "CLEAR_OR_TAKE": 7,
            "CLOSE_CURRENCY": 8
        }
        
        return b"".join(action_map[name].to_bytes(1, 'big') for name in action_names)
    
    def _get_pool_manager_abi(self) -> List[Dict]:
        """Get PoolManager ABI - simplified for core functions"""
        return [
            {
                "inputs": [{"name": "id", "type": "bytes32"}],
                "name": "getSlot0",
                "outputs": [
                    {"name": "sqrtPriceX96", "type": "uint160"},
                    {"name": "tick", "type": "int24"},
                    {"name": "protocolFee", "type": "uint24"},
                    {"name": "lpFee", "type": "uint24"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _get_position_manager_abi(self) -> List[Dict]:
        """Get PositionManager ABI - simplified for modifyLiquidities"""
        return [
            {
                "inputs": [
                    {"name": "unlockData", "type": "bytes"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "modifyLiquidities",
                "outputs": [],
                "stateMutability": "payable",
                "type": "function"
            }
        ]
    
    def _get_universal_router_abi(self) -> List[Dict]:
        """Get Universal Router ABI - simplified for execute"""
        return [
            {
                "inputs": [
                    {"name": "commands", "type": "bytes"},
                    {"name": "inputs", "type": "bytes[]"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "execute",
                "outputs": [],
                "stateMutability": "payable",
                "type": "function"
            }
        ]
    
    def _get_quoter_abi(self) -> List[Dict]:
        """Get Quoter ABI - simplified for quote functions"""
        return [
            {
                "inputs": [
                    {"name": "params", "type": "tuple"}
                ],
                "name": "quoteExactInputSingle",
                "outputs": [
                    {"name": "amountOut", "type": "uint256"},
                    {"name": "gasEstimate", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _price_to_tick(self, price: Decimal, tick_spacing: int) -> int:
        """Convert price to tick aligned to tick spacing"""
        tick = int(math.log(float(price)) / math.log(1.0001))
        return (tick // tick_spacing) * tick_spacing
    
    def _calculate_optimal_liquidity(
        self,
        current_price: Decimal,
        price_lower: Decimal, 
        price_upper: Decimal,
        liquidity_depth: Decimal
    ) -> Decimal:
        """Calculate optimal liquidity amount"""
        # Simplified calculation - should be enhanced with V4 math
        price_range = price_upper - price_lower
        return liquidity_depth * (current_price / price_range)
    
    def _estimate_fee_returns(
        self,
        volatility: float,
        liquidity: Decimal,
        current_price: Decimal
    ) -> Decimal:
        """Estimate fee returns based on volatility and liquidity"""
        # Base fee rate scaled by volatility
        base_fee_rate = Decimal("0.0005")  # 0.05%
        volatility_multiplier = Decimal(str(1 + volatility))
        return base_fee_rate * volatility_multiplier * liquidity
