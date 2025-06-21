"""
Advanced AI-powered compounding service for Dexter Protocol.
Integrates with DexBrain AI models for optimal compounding strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from web3 import Web3
from web3.contract import Contract

from dexbrain.config import Config
from dexbrain.enhanced_ml_models import EnhancedMLModels, LSTMModel, TickRangePredictor
from data.fetchers.uniswap_v3_fetcher import UniswapV3Fetcher
from utils.performance_tracker import PerformanceTracker, PerformanceMetrics


class CompoundStrategy(Enum):
    """Available compounding strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    AI_OPTIMIZED = "ai_optimized"
    GAS_OPTIMIZED = "gas_optimized"
    FEE_MAXIMIZED = "fee_maximized"


@dataclass
class CompoundOpportunity:
    """Represents a compound opportunity for a position."""
    token_id: int
    owner: str
    current_fees_usd: float
    estimated_gas_cost: float
    profit_potential: float
    optimal_timing_score: float
    strategy: CompoundStrategy
    urgency_score: float
    risk_score: float
    ai_confidence: float
    estimated_apr_improvement: float
    
    @property
    def priority_score(self) -> float:
        """Calculate overall priority score for compounding."""
        return (
            self.profit_potential * 0.3 +
            self.optimal_timing_score * 0.25 +
            self.urgency_score * 0.2 +
            (1 - self.risk_score) * 0.15 +
            self.ai_confidence * 0.1
        )


@dataclass
class CompoundResult:
    """Result of a compound operation."""
    token_id: int
    success: bool
    transaction_hash: Optional[str]
    gas_used: int
    gas_cost_eth: float
    fees_compounded_usd: float
    profit_usd: float
    execution_time: float
    strategy_used: CompoundStrategy
    error_message: Optional[str]


class CompoundService:
    """Advanced compounding service with AI optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.web3 = Web3(Web3.HTTPProvider(config.BASE_RPC_URL))
        
        # Initialize AI models
        self.ml_models = EnhancedMLModels()
        self.lstm_model = LSTMModel()
        self.tick_predictor = TickRangePredictor()
        
        # Initialize data fetcher
        self.uniswap_fetcher = UniswapV3Fetcher()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Contract instances (would be initialized with actual addresses)
        self.compoundor_contract: Optional[Contract] = None
        self.multi_compoundor_contract: Optional[Contract] = None
        
        # Strategy configurations
        self.strategy_configs = {
            CompoundStrategy.CONSERVATIVE: {
                "min_fees_threshold": 50.0,  # $50 minimum fees
                "max_gas_cost_ratio": 0.1,   # Max 10% of fees as gas cost
                "risk_tolerance": 0.2,       # Low risk tolerance
                "timing_sensitivity": 0.3,   # Low timing sensitivity
            },
            CompoundStrategy.BALANCED: {
                "min_fees_threshold": 25.0,  # $25 minimum fees
                "max_gas_cost_ratio": 0.15,  # Max 15% of fees as gas cost
                "risk_tolerance": 0.5,       # Medium risk tolerance
                "timing_sensitivity": 0.6,   # Medium timing sensitivity
            },
            CompoundStrategy.AGGRESSIVE: {
                "min_fees_threshold": 10.0,  # $10 minimum fees
                "max_gas_cost_ratio": 0.25,  # Max 25% of fees as gas cost
                "risk_tolerance": 0.8,       # High risk tolerance
                "timing_sensitivity": 0.9,   # High timing sensitivity
            },
            CompoundStrategy.AI_OPTIMIZED: {
                "min_fees_threshold": 5.0,   # AI determines threshold
                "max_gas_cost_ratio": 0.3,   # AI optimizes gas efficiency
                "risk_tolerance": 1.0,       # AI manages risk
                "timing_sensitivity": 1.0,   # AI optimizes timing
            }
        }
        
        # Cache for compound opportunities
        self._opportunity_cache: Dict[int, CompoundOpportunity] = {}
        self._cache_expiry = timedelta(minutes=5)
        self._last_cache_update = datetime.min
        
    async def initialize(self):
        """Initialize the compound service."""
        try:
            # Load contract ABIs and addresses
            await self._load_contracts()
            
            # Initialize AI models
            await self.ml_models.initialize()
            await self.lstm_model.load_model()
            await self.tick_predictor.load_model()
            
            self.logger.info("CompoundService initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CompoundService: {e}")
            raise
    
    async def find_compound_opportunities(
        self,
        max_positions: int = 100,
        strategies: List[CompoundStrategy] = None,
        min_profit_usd: float = 5.0
    ) -> List[CompoundOpportunity]:
        """Find and rank compound opportunities across all positions."""
        if strategies is None:
            strategies = [CompoundStrategy.AI_OPTIMIZED]
        
        try:
            # Check cache first
            if self._is_cache_valid():
                cached_opportunities = list(self._opportunity_cache.values())
                return sorted(cached_opportunities, key=lambda x: x.priority_score, reverse=True)[:max_positions]
            
            opportunities = []
            
            # Get all positions from the compoundor contract
            positions = await self._get_all_positions()
            
            for position in positions[:max_positions]:
                for strategy in strategies:
                    opportunity = await self._analyze_compound_opportunity(position, strategy)
                    
                    if opportunity and opportunity.profit_potential >= min_profit_usd:
                        opportunities.append(opportunity)
            
            # Remove duplicates (keep highest scoring opportunity per position)
            unique_opportunities = {}
            for opp in opportunities:
                if opp.token_id not in unique_opportunities or \
                   opp.priority_score > unique_opportunities[opp.token_id].priority_score:
                    unique_opportunities[opp.token_id] = opp
            
            final_opportunities = list(unique_opportunities.values())
            
            # Sort by priority score
            final_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Update cache
            self._opportunity_cache = {opp.token_id: opp for opp in final_opportunities}
            self._last_cache_update = datetime.now()
            
            self.logger.info(f"Found {len(final_opportunities)} compound opportunities")
            return final_opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding compound opportunities: {e}")
            return []
    
    async def execute_compound(
        self,
        opportunity: CompoundOpportunity,
        max_gas_price: Optional[int] = None,
        slippage_tolerance: float = 0.01
    ) -> CompoundResult:
        """Execute a compound operation for a specific opportunity."""
        start_time = datetime.now()
        
        try:
            # Pre-execution validation
            if not await self._validate_compound_opportunity(opportunity):
                return CompoundResult(
                    token_id=opportunity.token_id,
                    success=False,
                    transaction_hash=None,
                    gas_used=0,
                    gas_cost_eth=0.0,
                    fees_compounded_usd=0.0,
                    profit_usd=0.0,
                    execution_time=0.0,
                    strategy_used=opportunity.strategy,
                    error_message="Opportunity validation failed"
                )
            
            # Get optimal compound parameters
            compound_params = await self._get_optimal_compound_params(opportunity, slippage_tolerance)
            
            # Execute compound transaction
            if opportunity.strategy == CompoundStrategy.AI_OPTIMIZED:
                tx_hash, gas_used = await self._execute_ai_optimized_compound(
                    opportunity, compound_params, max_gas_price
                )
            else:
                tx_hash, gas_used = await self._execute_standard_compound(
                    opportunity, compound_params, max_gas_price
                )
            
            # Calculate results
            execution_time = (datetime.now() - start_time).total_seconds()
            gas_cost_eth = self._calculate_gas_cost(gas_used, max_gas_price)
            
            # Track performance
            await self._track_compound_performance(opportunity, tx_hash, gas_used, execution_time)
            
            result = CompoundResult(
                token_id=opportunity.token_id,
                success=True,
                transaction_hash=tx_hash,
                gas_used=gas_used,
                gas_cost_eth=gas_cost_eth,
                fees_compounded_usd=opportunity.current_fees_usd,
                profit_usd=opportunity.profit_potential,
                execution_time=execution_time,
                strategy_used=opportunity.strategy,
                error_message=None
            )
            
            self.logger.info(f"Successfully compounded position {opportunity.token_id}")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to compound position {opportunity.token_id}: {e}")
            
            return CompoundResult(
                token_id=opportunity.token_id,
                success=False,
                transaction_hash=None,
                gas_used=0,
                gas_cost_eth=0.0,
                fees_compounded_usd=0.0,
                profit_usd=0.0,
                execution_time=execution_time,
                strategy_used=opportunity.strategy,
                error_message=str(e)
            )
    
    async def execute_batch_compound(
        self,
        opportunities: List[CompoundOpportunity],
        max_positions_per_batch: int = 20,
        max_gas_price: Optional[int] = None
    ) -> List[CompoundResult]:
        """Execute batch compounding for multiple positions."""
        try:
            # Group opportunities into batches
            batches = [
                opportunities[i:i + max_positions_per_batch]
                for i in range(0, len(opportunities), max_positions_per_batch)
            ]
            
            all_results = []
            
            for batch_idx, batch in enumerate(batches):
                self.logger.info(f"Executing batch {batch_idx + 1}/{len(batches)} with {len(batch)} positions")
                
                # Prepare batch parameters
                batch_params = await self._prepare_batch_params(batch)
                
                # Execute batch compound
                try:
                    tx_hash, gas_used = await self._execute_batch_compound_transaction(
                        batch_params, max_gas_price
                    )
                    
                    # Create results for successful batch
                    batch_results = []
                    for opportunity in batch:
                        result = CompoundResult(
                            token_id=opportunity.token_id,
                            success=True,
                            transaction_hash=tx_hash,
                            gas_used=gas_used // len(batch),  # Approximate gas per position
                            gas_cost_eth=self._calculate_gas_cost(gas_used // len(batch), max_gas_price),
                            fees_compounded_usd=opportunity.current_fees_usd,
                            profit_usd=opportunity.profit_potential,
                            execution_time=0.0,  # Would be measured properly
                            strategy_used=opportunity.strategy,
                            error_message=None
                        )
                        batch_results.append(result)
                    
                    all_results.extend(batch_results)
                    
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx + 1} failed: {e}")
                    
                    # Create failed results for entire batch
                    for opportunity in batch:
                        result = CompoundResult(
                            token_id=opportunity.token_id,
                            success=False,
                            transaction_hash=None,
                            gas_used=0,
                            gas_cost_eth=0.0,
                            fees_compounded_usd=0.0,
                            profit_usd=0.0,
                            execution_time=0.0,
                            strategy_used=opportunity.strategy,
                            error_message=str(e)
                        )
                        all_results.append(result)
                
                # Add delay between batches to avoid overwhelming the network
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(2)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Batch compound execution failed: {e}")
            return []
    
    async def optimize_compound_timing(self, token_id: int) -> Dict[str, Any]:
        """Use AI to determine optimal compound timing for a position."""
        try:
            # Get position data
            position_data = await self._get_position_data(token_id)
            
            # Get historical fee accumulation data
            fee_history = await self._get_fee_history(token_id, days=30)
            
            # Use LSTM model for timing prediction
            timing_prediction = await self.lstm_model.predict_optimal_timing(
                position_data, fee_history
            )
            
            # Get tick range prediction
            tick_prediction = await self.tick_predictor.predict_optimal_range(position_data)
            
            # Calculate comprehensive timing score
            timing_score = await self._calculate_timing_score(
                position_data, timing_prediction, tick_prediction
            )
            
            return {
                "optimal_timing_hours": timing_prediction.get("hours_until_optimal", 0),
                "confidence": timing_prediction.get("confidence", 0.5),
                "timing_score": timing_score,
                "recommended_action": timing_prediction.get("action", "wait"),
                "predicted_fee_accumulation": timing_prediction.get("predicted_fees", 0),
                "risk_assessment": timing_prediction.get("risk_level", "medium"),
                "gas_cost_forecast": timing_prediction.get("gas_forecast", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing compound timing for {token_id}: {e}")
            return {
                "optimal_timing_hours": 0,
                "confidence": 0.0,
                "timing_score": 0.0,
                "recommended_action": "error",
                "error": str(e)
            }
    
    async def get_compound_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics on compounding performance."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get performance metrics
            metrics = await self.performance_tracker.get_metrics(
                start_date=start_date,
                end_date=end_date,
                metric_types=[
                    "compound_success_rate",
                    "average_profit_per_compound",
                    "gas_efficiency",
                    "timing_accuracy",
                    "strategy_performance"
                ]
            )
            
            # Calculate strategy-specific analytics
            strategy_analytics = {}
            for strategy in CompoundStrategy:
                strategy_metrics = await self._get_strategy_analytics(strategy, start_date, end_date)
                strategy_analytics[strategy.value] = strategy_metrics
            
            # AI model performance
            ai_performance = await self._get_ai_performance_metrics(start_date, end_date)
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "overall_metrics": metrics,
                "strategy_analytics": strategy_analytics,
                "ai_performance": ai_performance,
                "compound_volume": await self._get_compound_volume(start_date, end_date),
                "gas_analytics": await self._get_gas_analytics(start_date, end_date),
                "profit_distribution": await self._get_profit_distribution(start_date, end_date)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating compound analytics: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    async def _analyze_compound_opportunity(
        self, position: Dict, strategy: CompoundStrategy
    ) -> Optional[CompoundOpportunity]:
        """Analyze a single position for compound opportunity."""
        try:
            token_id = position["token_id"]
            
            # Get current fees and position value
            fees_data = await self._get_position_fees(token_id)
            current_fees_usd = fees_data["total_fees_usd"]
            
            # Check minimum threshold
            strategy_config = self.strategy_configs[strategy]
            if current_fees_usd < strategy_config["min_fees_threshold"]:
                return None
            
            # Estimate gas cost
            estimated_gas_cost = await self._estimate_compound_gas_cost(token_id, strategy)
            
            # Check gas cost ratio
            gas_cost_ratio = estimated_gas_cost / current_fees_usd
            if gas_cost_ratio > strategy_config["max_gas_cost_ratio"]:
                return None
            
            # Calculate profit potential
            profit_potential = current_fees_usd - estimated_gas_cost
            
            # Get AI-based scores
            if strategy == CompoundStrategy.AI_OPTIMIZED:
                ai_scores = await self._get_ai_scores(position, fees_data)
                optimal_timing_score = ai_scores["timing_score"]
                urgency_score = ai_scores["urgency_score"]
                risk_score = ai_scores["risk_score"]
                ai_confidence = ai_scores["confidence"]
                apr_improvement = ai_scores["apr_improvement"]
            else:
                optimal_timing_score = self._calculate_basic_timing_score(fees_data)
                urgency_score = min(current_fees_usd / 100.0, 1.0)  # Based on fee size
                risk_score = strategy_config["risk_tolerance"]
                ai_confidence = 0.5  # Default confidence for non-AI strategies
                apr_improvement = self._estimate_apr_improvement(fees_data)
            
            return CompoundOpportunity(
                token_id=token_id,
                owner=position["owner"],
                current_fees_usd=current_fees_usd,
                estimated_gas_cost=estimated_gas_cost,
                profit_potential=profit_potential,
                optimal_timing_score=optimal_timing_score,
                strategy=strategy,
                urgency_score=urgency_score,
                risk_score=risk_score,
                ai_confidence=ai_confidence,
                estimated_apr_improvement=apr_improvement
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing compound opportunity for position {position.get('token_id')}: {e}")
            return None
    
    async def _get_ai_scores(self, position: Dict, fees_data: Dict) -> Dict[str, float]:
        """Get AI-based scores for compound opportunity analysis."""
        try:
            # Use enhanced ML models to score the opportunity
            position_features = await self._extract_position_features(position, fees_data)
            
            # Get predictions from different models
            timing_prediction = await self.lstm_model.predict(position_features)
            risk_assessment = await self.ml_models.assess_risk(position_features)
            apr_prediction = await self.ml_models.predict_apr_improvement(position_features)
            
            return {
                "timing_score": timing_prediction.get("timing_score", 0.5),
                "urgency_score": timing_prediction.get("urgency_score", 0.5),
                "risk_score": risk_assessment.get("risk_score", 0.5),
                "confidence": min(
                    timing_prediction.get("confidence", 0.5),
                    risk_assessment.get("confidence", 0.5),
                    apr_prediction.get("confidence", 0.5)
                ),
                "apr_improvement": apr_prediction.get("improvement", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting AI scores: {e}")
            return {
                "timing_score": 0.5,
                "urgency_score": 0.5,
                "risk_score": 0.5,
                "confidence": 0.0,
                "apr_improvement": 0.0
            }
    
    def _is_cache_valid(self) -> bool:
        """Check if the opportunity cache is still valid."""
        return datetime.now() - self._last_cache_update < self._cache_expiry
    
    async def _load_contracts(self):
        """Load smart contract instances."""
        # This would load the actual contract ABIs and create Web3 contract instances
        # Simplified for now
        pass
    
    async def _get_all_positions(self) -> List[Dict]:
        """Get all positions from the compoundor contract."""
        # This would query the contract for all positions
        # Simplified for now
        return []
    
    async def _get_position_fees(self, token_id: int) -> Dict:
        """Get current fees for a position."""
        # This would query the position's current fee data
        return {"total_fees_usd": 0.0}
    
    async def _estimate_compound_gas_cost(self, token_id: int, strategy: CompoundStrategy) -> float:
        """Estimate gas cost for compounding a position."""
        # Base gas cost estimation
        base_gas = 150000
        
        # Strategy-specific adjustments
        if strategy == CompoundStrategy.AI_OPTIMIZED:
            base_gas += 50000  # Additional gas for AI operations
        elif strategy == CompoundStrategy.GAS_OPTIMIZED:
            base_gas -= 20000  # Optimized for lower gas usage
        
        # Get current gas price
        gas_price_wei = self.web3.eth.gas_price
        gas_price_gwei = gas_price_wei / 1e9
        
        # Calculate cost in ETH
        gas_cost_eth = (base_gas * gas_price_wei) / 1e18
        
        # Convert to USD (would need price feed)
        eth_price_usd = 2000  # Simplified
        return gas_cost_eth * eth_price_usd
    
    def _calculate_basic_timing_score(self, fees_data: Dict) -> float:
        """Calculate basic timing score without AI."""
        # Simple heuristic based on fee accumulation rate
        return min(fees_data.get("total_fees_usd", 0) / 50.0, 1.0)
    
    def _estimate_apr_improvement(self, fees_data: Dict) -> float:
        """Estimate APR improvement from compounding."""
        # Simplified calculation
        return fees_data.get("total_fees_usd", 0) * 0.001  # 0.1% improvement per $100 fees
    
    def _calculate_gas_cost(self, gas_used: int, max_gas_price: Optional[int]) -> float:
        """Calculate gas cost in ETH."""
        gas_price = max_gas_price or self.web3.eth.gas_price
        return (gas_used * gas_price) / 1e18