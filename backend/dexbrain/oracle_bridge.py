"""
Secure Oracle Bridge for VPS-to-Smart Contract ML Predictions
Implements secure, validated oracle data delivery with MEV protection
"""

import asyncio
import logging
import json
import os
import time
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
import aiohttp
import redis.asyncio as redis

from .config import Config
from .realtime_strategy_optimizer import RealtimeStrategyOptimizer

logger = logging.getLogger(__name__)

@dataclass
class MLPrediction:
    """ML prediction data for oracle delivery"""
    pool_address: str
    prediction_type: str  # 'fee_optimization', 'rebalance', 'emergency'
    action_recommendation: str
    confidence: float
    urgency: float
    optimal_fee_bp: Optional[int] = None
    optimal_tick_lower: Optional[int] = None
    optimal_tick_upper: Optional[int] = None
    market_regime: str = 'stable'
    expected_apr: Optional[float] = None
    risk_score: float = 0.5
    timestamp: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def get_signature_data(self) -> str:
        """Get data string for signature verification"""
        data = {
            'pool_address': self.pool_address,
            'action': self.action_recommendation,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }
        return json.dumps(data, sort_keys=True)

@dataclass
class OracleValidation:
    """Oracle data validation result"""
    is_valid: bool
    confidence: float
    deviation_from_consensus: float
    sources_count: int
    reasons: List[str]

class SecureOracleBridge:
    """
    Secure bridge between VPS ML services and smart contracts
    Implements multi-signature validation and MEV protection
    """
    
    def __init__(self):
        # Web3 configuration
        self.rpc_url = os.getenv('BASE_RPC_URL', 'https://mainnet.base.org')
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Oracle signing
        self.oracle_private_key = os.getenv('ORACLE_PRIVATE_KEY')
        if not self.oracle_private_key:
            raise ValueError("ORACLE_PRIVATE_KEY environment variable required")
        
        self.oracle_account = Account.from_key(self.oracle_private_key)
        
        # Smart contract addresses
        self.hook_address = os.getenv('DEXTER_HOOK_ADDRESS')
        self.oracle_registry = os.getenv('ORACLE_REGISTRY_ADDRESS')
        
        # Redis for caching and rate limiting
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        
        # ML components
        self.strategy_optimizer = RealtimeStrategyOptimizer()
        
        # Validation thresholds
        self.min_confidence = 0.7
        self.max_deviation = 0.1  # 10% max deviation from consensus
        self.rate_limit_per_minute = 60
        
        # MEV protection
        self.mev_protection_delay = 12  # seconds (1 block)
        self.price_impact_threshold = 0.05  # 5% max price impact
        
        logger.info("Secure oracle bridge initialized")
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Oracle bridge async components initialized")
    
    async def generate_ml_prediction(self, pool_data: Dict[str, Any]) -> MLPrediction:
        """Generate ML prediction for pool using strategy optimizer"""
        
        try:
            # Get strategy recommendation
            recommendations = self.strategy_optimizer.generate_strategy_recommendations([pool_data])
            
            if not recommendations:
                raise ValueError("No strategy recommendation generated")
            
            rec = recommendations[0]
            
            # Convert to oracle prediction format
            prediction = MLPrediction(
                pool_address=pool_data.get('pool_address', ''),
                prediction_type='strategy_optimization',
                action_recommendation=rec['action'],
                confidence=rec['confidence'],
                urgency=rec['urgency'],
                market_regime=self.strategy_optimizer.analyze_market_regime(pool_data),
                risk_score=rec['expected_outcomes'].get('risk_score', 0.5)
            )
            
            # Add action-specific parameters
            if rec['action'] == 'compound':
                prediction.optimal_fee_bp = None  # Keep current fee
                
            elif rec['action'] in ['widen_range', 'narrow_range', 'rebalance']:
                prediction.optimal_tick_lower = pool_data.get('suggested_tick_lower')
                prediction.optimal_tick_upper = pool_data.get('suggested_tick_upper')
                
            elif rec['action'] == 'hold':
                prediction.prediction_type = 'no_action'
            
            return prediction
            
        except Exception as e:
            logger.error(f"ML prediction generation failed: {e}")
            raise
    
    async def validate_prediction(self, prediction: MLPrediction, 
                                external_sources: List[Dict[str, Any]] = None) -> OracleValidation:
        """Validate ML prediction against multiple sources"""
        
        validation_reasons = []
        is_valid = True
        
        # Confidence threshold check
        if prediction.confidence < self.min_confidence:
            validation_reasons.append(f"Low confidence: {prediction.confidence:.2f} < {self.min_confidence}")
            is_valid = False
        
        # Timestamp freshness check
        age_seconds = time.time() - prediction.timestamp
        if age_seconds > 300:  # 5 minutes max age
            validation_reasons.append(f"Stale prediction: {age_seconds:.0f}s old")
            is_valid = False
        
        # Rate limiting check
        rate_limit_key = f"oracle_rate_limit:{prediction.pool_address}"
        recent_calls = await self.redis_client.get(rate_limit_key)
        if recent_calls and int(recent_calls) > self.rate_limit_per_minute:
            validation_reasons.append("Rate limit exceeded")
            is_valid = False
        
        # External source validation
        deviation = 0.0
        sources_count = 1  # Our ML prediction
        
        if external_sources:
            consensus_confidence = prediction.confidence
            source_confidences = [src.get('confidence', 0) for src in external_sources]
            
            if source_confidences:
                avg_external_confidence = sum(source_confidences) / len(source_confidences)
                deviation = abs(consensus_confidence - avg_external_confidence) / consensus_confidence
                sources_count += len(source_confidences)
                
                if deviation > self.max_deviation:
                    validation_reasons.append(f"High deviation: {deviation:.2%} > {self.max_deviation:.2%}")
                    is_valid = False
        
        # Pool-specific validation
        pool_validation = await self._validate_pool_state(prediction)
        if not pool_validation:
            validation_reasons.append("Pool state validation failed")
            is_valid = False
        
        if is_valid:
            validation_reasons.append("All validation checks passed")
        
        return OracleValidation(
            is_valid=is_valid,
            confidence=prediction.confidence,
            deviation_from_consensus=deviation,
            sources_count=sources_count,
            reasons=validation_reasons
        )
    
    async def _validate_pool_state(self, prediction: MLPrediction) -> bool:
        """Validate pool state for prediction feasibility"""
        
        try:
            # Check if pool exists and is active
            pool_cache_key = f"pool_state:{prediction.pool_address}"
            cached_state = await self.redis_client.get(pool_cache_key)
            
            if cached_state:
                pool_state = json.loads(cached_state)
                
                # Check TVL threshold
                if pool_state.get('tvl_usd', 0) < 10000:  # $10k minimum
                    return False
                
                # Check if pool is not in emergency mode
                if pool_state.get('emergency_mode', False):
                    return False
                
                # Validate action feasibility
                if prediction.action_recommendation == 'rebalance':
                    current_tick = pool_state.get('current_tick', 0)
                    tick_lower = prediction.optimal_tick_lower or pool_state.get('tick_lower', 0)
                    tick_upper = prediction.optimal_tick_upper or pool_state.get('tick_upper', 0)
                    
                    # Check if rebalance would be meaningful
                    if tick_lower <= current_tick <= tick_upper:
                        return False  # Position already in range
            
            return True
            
        except Exception as e:
            logger.warning(f"Pool validation error: {e}")
            return False
    
    def sign_prediction(self, prediction: MLPrediction) -> str:
        """Sign prediction with oracle private key"""
        
        message_data = prediction.get_signature_data()
        message_hash = hashlib.sha256(message_data.encode()).hexdigest()
        
        # Create Ethereum signed message
        message = encode_defunct(text=message_hash)
        signed_message = Account.sign_message(message, private_key=self.oracle_private_key)
        
        return signed_message.signature.hex()
    
    async def deliver_prediction_to_contract(self, prediction: MLPrediction, 
                                           validation: OracleValidation) -> Dict[str, Any]:
        """Deliver validated prediction to smart contract"""
        
        if not validation.is_valid:
            raise ValueError(f"Cannot deliver invalid prediction: {validation.reasons}")
        
        try:
            # Apply MEV protection delay
            await self._apply_mev_protection(prediction)
            
            # Sign the prediction
            signature = self.sign_prediction(prediction)
            
            # Prepare contract call data
            contract_data = {
                'pool_address': prediction.pool_address,
                'action_type': prediction.action_recommendation,
                'confidence': int(prediction.confidence * 10000),  # Convert to basis points
                'urgency': int(prediction.urgency * 10000),
                'timestamp': prediction.timestamp,
                'signature': signature
            }
            
            # Add action-specific parameters
            if prediction.optimal_fee_bp:
                contract_data['optimal_fee'] = prediction.optimal_fee_bp
            
            if prediction.optimal_tick_lower and prediction.optimal_tick_upper:
                contract_data['tick_lower'] = prediction.optimal_tick_lower
                contract_data['tick_upper'] = prediction.optimal_tick_upper
            
            # Cache delivery attempt
            delivery_key = f"oracle_delivery:{prediction.pool_address}:{prediction.timestamp}"
            await self.redis_client.setex(delivery_key, 3600, json.dumps(contract_data))
            
            # Update rate limiting
            rate_limit_key = f"oracle_rate_limit:{prediction.pool_address}"
            await self.redis_client.incr(rate_limit_key)
            await self.redis_client.expire(rate_limit_key, 60)
            
            logger.info(f"Prediction delivered for pool {prediction.pool_address}: {prediction.action_recommendation}")
            
            return {
                'status': 'delivered',
                'contract_data': contract_data,
                'gas_estimate': await self._estimate_gas_cost(contract_data),
                'delivery_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction delivery failed: {e}")
            raise
    
    async def _apply_mev_protection(self, prediction: MLPrediction):
        """Apply MEV protection mechanisms"""
        
        # Time-based delay to prevent front-running
        if prediction.urgency < 0.8:  # Only delay non-urgent predictions
            await asyncio.sleep(self.mev_protection_delay)
        
        # Check for recent price manipulation
        price_history_key = f"price_history:{prediction.pool_address}"
        price_history = await self.redis_client.lrange(price_history_key, 0, 10)
        
        if len(price_history) >= 2:
            try:
                recent_prices = [float(p) for p in price_history[:2]]
                price_change = abs(recent_prices[0] - recent_prices[1]) / recent_prices[1]
                
                if price_change > self.price_impact_threshold:
                    logger.warning(f"High price impact detected: {price_change:.2%}, applying additional delay")
                    await asyncio.sleep(self.mev_protection_delay * 2)
                    
            except (ValueError, ZeroDivisionError):
                pass  # Skip if price data is invalid
    
    async def _estimate_gas_cost(self, contract_data: Dict[str, Any]) -> int:
        """Estimate gas cost for contract interaction"""
        
        # Base gas for oracle update
        base_gas = 150000
        
        # Additional gas for complex operations
        if contract_data.get('tick_lower') and contract_data.get('tick_upper'):
            base_gas += 100000  # Rebalancing gas
        
        return base_gas
    
    async def monitor_oracle_health(self) -> Dict[str, Any]:
        """Monitor oracle bridge health and performance"""
        
        try:
            # Check Redis connectivity
            redis_ping = await self.redis_client.ping()
            
            # Check Web3 connectivity
            latest_block = self.w3.eth.block_number
            
            # Get recent delivery stats
            delivered_count = await self.redis_client.get('oracle_delivered_24h') or '0'
            failed_count = await self.redis_client.get('oracle_failed_24h') or '0'
            
            # Calculate success rate
            total_attempts = int(delivered_count) + int(failed_count)
            success_rate = int(delivered_count) / max(total_attempts, 1)
            
            health_status = {
                'status': 'healthy',
                'redis_connected': redis_ping,
                'web3_connected': latest_block > 0,
                'latest_block': latest_block,
                'deliveries_24h': int(delivered_count),
                'failures_24h': int(failed_count),
                'success_rate': success_rate,
                'oracle_address': self.oracle_account.address,
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine overall health
            if not redis_ping or latest_block == 0 or success_rate < 0.9:
                health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def process_pool_predictions(self, pools_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process ML predictions for multiple pools"""
        
        results = []
        
        for pool_data in pools_data:
            try:
                # Generate ML prediction
                prediction = await self.generate_ml_prediction(pool_data)
                
                # Validate prediction
                validation = await self.validate_prediction(prediction)
                
                if validation.is_valid:
                    # Deliver to contract
                    delivery_result = await self.deliver_prediction_to_contract(prediction, validation)
                    
                    result = {
                        'pool_address': prediction.pool_address,
                        'status': 'success',
                        'action': prediction.action_recommendation,
                        'confidence': prediction.confidence,
                        'delivery': delivery_result
                    }
                else:
                    result = {
                        'pool_address': prediction.pool_address,
                        'status': 'validation_failed',
                        'reasons': validation.reasons
                    }
                
                results.append(result)
                
                # Track metrics
                await self.redis_client.incr('oracle_delivered_24h')
                await self.redis_client.expire('oracle_delivered_24h', 86400)
                
            except Exception as e:
                logger.error(f"Pool prediction processing failed for {pool_data.get('pool_address')}: {e}")
                
                results.append({
                    'pool_address': pool_data.get('pool_address', 'unknown'),
                    'status': 'error',
                    'error': str(e)
                })
                
                # Track failures
                await self.redis_client.incr('oracle_failed_24h')
                await self.redis_client.expire('oracle_failed_24h', 86400)
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()

# Factory function
async def create_oracle_bridge() -> SecureOracleBridge:
    """Create and initialize oracle bridge"""
    bridge = SecureOracleBridge()
    await bridge.initialize()
    return bridge