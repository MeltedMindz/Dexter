"""
Advanced Uniswap V3/V4 Fetcher with ML-driven optimizations
Implements comprehensive feature extraction and position optimization
"""

import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from web3 import Web3
from eth_utils import to_checksum_address
import json
import math

from .uniswap_v4_fetcher import UniswapV4Fetcher
from .base_interface import LiquidityPoolFetcher

logger = logging.getLogger(__name__)

@dataclass
class PositionOptimization:
    """Results from ML-driven position optimization"""
    optimal_lower_tick: int
    optimal_upper_tick: int
    predicted_apr: float
    predicted_il: float
    capital_efficiency: float
    risk_score: float
    confidence: float
    reasoning: str

@dataclass
class LiquidityDistribution:
    """Liquidity distribution analysis across tick ranges"""
    active_liquidity: float
    total_liquidity: float
    concentration_ratio: float
    tick_ranges: List[Tuple[int, int, float]]  # (lower, upper, liquidity)
    whale_positions: List[Tuple[int, int, float]]  # Large positions

@dataclass
class FeeEarningsEstimate:
    """Fee earnings prediction for different time horizons"""
    daily_estimate: float
    weekly_estimate: float
    monthly_estimate: float
    confidence_interval: Tuple[float, float]
    assumptions: Dict[str, Any]

class AdvancedUniswapFetcher(UniswapV4Fetcher):
    """
    Enhanced Uniswap fetcher with ML-driven optimizations and advanced analytics
    """
    
    def __init__(self, web3_provider: Web3, graph_url: str, api_key: Optional[str] = None):
        super().__init__(web3_provider, graph_url, api_key)
        
        # Initialize ML models for position optimization
        try:
            import sys
            from pathlib import Path
            # Add backend path for imports
            backend_path = Path(__file__).parent.parent.parent.parent / 'backend'
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
            
            from dexbrain.models.enhanced_ml_models import UniswapLPOptimizer
            self.ml_optimizer = UniswapLPOptimizer()
            self.ml_enabled = True
            logger.info("ML optimizer initialized successfully")
        except ImportError as e:
            logger.warning(f"ML optimizer not available: {e}")
            self.ml_optimizer = None
            self.ml_enabled = False
        
        # Cache for expensive calculations
        self._liquidity_cache: Dict[str, Tuple[LiquidityDistribution, datetime]] = {}
        self._price_history_cache: Dict[str, Tuple[List[float], datetime]] = {}
        
        # Constants for calculations
        self.CACHE_TTL = timedelta(minutes=5)
        self.MIN_TICK = -887272
        self.MAX_TICK = 887272
    
    async def get_optimal_tick_ranges(
        self, 
        pool_address: str, 
        volatility_forecast: Optional[float] = None,
        capital_amount: float = 10000.0,
        risk_tolerance: str = "medium"
    ) -> PositionOptimization:
        """
        ML-driven tick range optimization for concentrated liquidity positions
        
        Args:
            pool_address: Pool contract address
            volatility_forecast: Expected volatility (if available)
            capital_amount: Amount of capital to deploy
            risk_tolerance: "low", "medium", or "high"
            
        Returns:
            PositionOptimization with recommended tick ranges and predictions
        """
        try:
            # Get comprehensive pool data
            pool_data = await self.get_comprehensive_pool_data(pool_address)
            
            if not pool_data:
                raise ValueError(f"Could not fetch data for pool {pool_address}")
            
            # Add additional context
            pool_data.update({
                'capital_amount': capital_amount,
                'risk_tolerance': risk_tolerance,
                'volatility_forecast': volatility_forecast
            })
            
            if self.ml_enabled and self.ml_optimizer:
                # Use ML model for optimization
                ml_predictions = self.ml_optimizer.predict_optimal_position(pool_data)
                
                optimization = PositionOptimization(
                    optimal_lower_tick=ml_predictions['optimal_lower_tick'],
                    optimal_upper_tick=ml_predictions['optimal_upper_tick'],
                    predicted_apr=ml_predictions['predicted_apr'],
                    predicted_il=ml_predictions['expected_impermanent_loss'],
                    capital_efficiency=self._calculate_capital_efficiency(
                        ml_predictions['optimal_lower_tick'],
                        ml_predictions['optimal_upper_tick'],
                        pool_data
                    ),
                    risk_score=ml_predictions['risk_score'],
                    confidence=ml_predictions['confidence_score'],
                    reasoning="ML-optimized based on historical patterns and current market conditions"
                )
            else:
                # Fallback to heuristic optimization
                optimization = self._heuristic_tick_optimization(pool_data, capital_amount, risk_tolerance)
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing tick ranges for {pool_address}: {e}")
            # Return conservative default
            return self._get_conservative_default(pool_address)
    
    async def predict_fee_earnings(
        self, 
        position_params: Dict[str, Any], 
        time_horizon: int = 30
    ) -> FeeEarningsEstimate:
        """
        Predict fee earnings for a liquidity position over specified time horizon
        
        Args:
            position_params: Position configuration (ticks, amount, pool)
            time_horizon: Prediction horizon in days
            
        Returns:
            FeeEarningsEstimate with predicted earnings
        """
        try:
            pool_address = position_params.get('pool_address')
            lower_tick = position_params.get('lower_tick')
            upper_tick = position_params.get('upper_tick')
            capital_amount = position_params.get('amount', 10000)
            
            # Get historical fee data
            fee_history = await self.get_historical_fee_data(pool_address, days=90)
            
            # Get current pool metrics
            pool_data = await self.get_comprehensive_pool_data(pool_address)
            
            if not pool_data or not fee_history:
                raise ValueError("Insufficient data for fee prediction")
            
            # Calculate position's share of liquidity in range
            liquidity_share = self._calculate_liquidity_share_in_range(
                pool_data, lower_tick, upper_tick, capital_amount
            )
            
            # Estimate fees based on historical patterns
            if self.ml_enabled and self.ml_optimizer:
                # Use ML predictions
                ml_data = {
                    **pool_data,
                    'position_lower_tick': lower_tick,
                    'position_upper_tick': upper_tick,
                    'capital_amount': capital_amount
                }
                
                ml_predictions = self.ml_optimizer.predict_optimal_position(ml_data)
                
                daily_fees = ml_predictions.get('fee_earnings_1d', 0)
                weekly_fees = ml_predictions.get('fee_earnings_7d', 0)
                monthly_fees = ml_predictions.get('fee_earnings_30d', 0)
                
            else:
                # Heuristic calculation
                avg_daily_volume = pool_data.get('volume_24h', 0)
                fee_tier = pool_data.get('fee_tier', 3000) / 1_000_000  # Convert to decimal
                
                # Estimate daily fees
                total_daily_fees = avg_daily_volume * fee_tier
                position_daily_fees = total_daily_fees * liquidity_share
                
                daily_fees = position_daily_fees
                weekly_fees = daily_fees * 7
                monthly_fees = daily_fees * 30
            
            # Calculate confidence interval based on volatility
            volatility = pool_data.get('price_volatility_24h', 0.1)
            confidence_range = daily_fees * volatility * 2  # ±2σ approximation
            
            return FeeEarningsEstimate(
                daily_estimate=daily_fees,
                weekly_estimate=weekly_fees,
                monthly_estimate=monthly_fees,
                confidence_interval=(
                    max(0, monthly_fees - confidence_range * 30),
                    monthly_fees + confidence_range * 30
                ),
                assumptions={
                    'average_volume': pool_data.get('volume_24h', 0),
                    'liquidity_share': liquidity_share,
                    'fee_tier': pool_data.get('fee_tier', 3000),
                    'volatility': volatility
                }
            )
            
        except Exception as e:
            logger.error(f"Error predicting fee earnings: {e}")
            return FeeEarningsEstimate(0, 0, 0, (0, 0), {})
    
    async def calculate_capital_efficiency(
        self, 
        multiple_positions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Portfolio-level capital allocation optimization
        
        Args:
            multiple_positions: List of position configurations
            
        Returns:
            Capital efficiency metrics and recommendations
        """
        try:
            total_capital = sum(pos.get('amount', 0) for pos in multiple_positions)
            total_predicted_fees = 0
            total_predicted_il = 0
            
            efficiency_scores = []
            
            for position in multiple_positions:
                # Get individual position optimization
                optimization = await self.get_optimal_tick_ranges(
                    position['pool_address'],
                    capital_amount=position.get('amount', 0)
                )
                
                # Calculate efficiency score
                efficiency = self._calculate_position_efficiency(optimization, position)
                efficiency_scores.append(efficiency)
                
                total_predicted_fees += optimization.predicted_apr * position.get('amount', 0)
                total_predicted_il += abs(optimization.predicted_il) * position.get('amount', 0)
            
            portfolio_apr = total_predicted_fees / total_capital if total_capital > 0 else 0
            portfolio_il = total_predicted_il / total_capital if total_capital > 0 else 0
            
            return {
                'portfolio_apr': portfolio_apr,
                'portfolio_il_risk': portfolio_il,
                'capital_efficiency': np.mean(efficiency_scores),
                'diversification_score': self._calculate_diversification_score(multiple_positions),
                'risk_adjusted_return': portfolio_apr / max(portfolio_il, 0.01),
                'individual_efficiencies': efficiency_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating capital efficiency: {e}")
            return {}
    
    async def get_liquidity_distribution(self, pool_address: str) -> LiquidityDistribution:
        """
        Analyze current liquidity distribution across ticks
        
        Args:
            pool_address: Pool contract address
            
        Returns:
            LiquidityDistribution with detailed analysis
        """
        # Check cache first
        cache_key = pool_address
        if cache_key in self._liquidity_cache:
            cached_data, cached_time = self._liquidity_cache[cache_key]
            if datetime.now() - cached_time < self.CACHE_TTL:
                return cached_data
        
        try:
            # Query liquidity positions from subgraph
            query = """
            query getLiquidityDistribution($pool: String!) {
              pool(id: $pool) {
                liquidity
                tick
                positions(first: 1000, orderBy: liquidity, orderDirection: desc) {
                  id
                  liquidity
                  tickLower { tickIdx }
                  tickUpper { tickIdx }
                  owner
                }
                ticks(first: 1000) {
                  tickIdx
                  liquidityNet
                  liquidityGross
                }
              }
            }
            """
            
            variables = {"pool": pool_address.lower()}
            
            async with aiohttp.ClientSession() as session:
                response = await self._make_graph_request(session, query, variables)
                
                if not response or 'pool' not in response:
                    raise ValueError("Invalid response from subgraph")
                
                pool_data = response['pool']
                positions = pool_data.get('positions', [])
                ticks = pool_data.get('ticks', [])
                
                # Calculate liquidity distribution
                total_liquidity = float(pool_data.get('liquidity', 0))
                active_liquidity = self._calculate_active_liquidity(ticks, pool_data.get('tick', 0))
                
                # Identify tick ranges with significant liquidity
                tick_ranges = self._identify_liquidity_ranges(positions)
                
                # Identify whale positions (top 10% by liquidity)
                sorted_positions = sorted(positions, key=lambda x: float(x.get('liquidity', 0)), reverse=True)
                top_10_percent = max(1, len(sorted_positions) // 10)
                whale_positions = [
                    (
                        int(pos['tickLower']['tickIdx']),
                        int(pos['tickUpper']['tickIdx']),
                        float(pos['liquidity'])
                    )
                    for pos in sorted_positions[:top_10_percent]
                ]
                
                concentration_ratio = active_liquidity / max(total_liquidity, 1)
                
                distribution = LiquidityDistribution(
                    active_liquidity=active_liquidity,
                    total_liquidity=total_liquidity,
                    concentration_ratio=concentration_ratio,
                    tick_ranges=tick_ranges,
                    whale_positions=whale_positions
                )
                
                # Cache the result
                self._liquidity_cache[cache_key] = (distribution, datetime.now())
                
                return distribution
                
        except Exception as e:
            logger.error(f"Error analyzing liquidity distribution for {pool_address}: {e}")
            return LiquidityDistribution(0, 0, 0, [], [])
    
    async def estimate_impermanent_loss(
        self, 
        position: Dict[str, Any], 
        price_scenarios: List[float]
    ) -> Dict[str, float]:
        """
        Monte Carlo estimation of impermanent loss under different price scenarios
        
        Args:
            position: Position configuration
            price_scenarios: List of potential price changes (as ratios)
            
        Returns:
            Impermanent loss estimates and statistics
        """
        try:
            current_price = float(position.get('current_price', 1.0))
            lower_tick = position.get('lower_tick')
            upper_tick = position.get('upper_tick')
            
            # Convert ticks to prices
            lower_price = self._tick_to_price(lower_tick)
            upper_price = self._tick_to_price(upper_tick)
            
            il_estimates = []
            
            for price_ratio in price_scenarios:
                new_price = current_price * price_ratio
                
                # Calculate IL for this price scenario
                if new_price < lower_price or new_price > upper_price:
                    # Position is out of range - calculate based on single token
                    il = self._calculate_out_of_range_il(
                        current_price, new_price, lower_price, upper_price
                    )
                else:
                    # Position is in range - calculate standard IL
                    il = self._calculate_in_range_il(current_price, new_price)
                
                il_estimates.append(il)
            
            il_array = np.array(il_estimates)
            
            return {
                'mean_il': float(np.mean(il_array)),
                'median_il': float(np.median(il_array)),
                'worst_case_il': float(np.max(il_array)),
                'best_case_il': float(np.min(il_array)),
                'volatility': float(np.std(il_array)),
                'scenarios_analyzed': len(price_scenarios),
                'percentile_95': float(np.percentile(il_array, 95)),
                'percentile_5': float(np.percentile(il_array, 5))
            }
            
        except Exception as e:
            logger.error(f"Error estimating impermanent loss: {e}")
            return {}
    
    # Helper methods
    
    async def get_comprehensive_pool_data(self, pool_address: str) -> Dict[str, Any]:
        """Get comprehensive pool data for ML features"""
        try:
            # Get basic pool data
            pool_data = await self.get_pool_data(pool_address)
            
            # Enhance with additional metrics
            liquidity_dist = await self.get_liquidity_distribution(pool_address)
            price_history = await self.get_price_history(pool_address, days=7)
            
            # Calculate volatility metrics
            volatility_1h = self._calculate_volatility(price_history, hours=1)
            volatility_24h = self._calculate_volatility(price_history, hours=24)
            volatility_7d = self._calculate_volatility(price_history, hours=168)
            
            # Enhance pool data
            pool_data.update({
                'liquidity_distribution': liquidity_dist.concentration_ratio,
                'active_liquidity_ratio': liquidity_dist.active_liquidity / max(liquidity_dist.total_liquidity, 1),
                'whale_concentration': len(liquidity_dist.whale_positions) / max(len(liquidity_dist.tick_ranges), 1),
                'price_volatility_1h': volatility_1h,
                'price_volatility_24h': volatility_24h,
                'price_volatility_7d': volatility_7d,
                'volume_to_tvl_ratio': pool_data.get('volume_24h', 0) / max(pool_data.get('tvl', 1), 1),
                'position_count': len(liquidity_dist.tick_ranges),
                'tick_bitmap_density': self._calculate_tick_density(liquidity_dist)
            })
            
            return pool_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive pool data: {e}")
            return {}
    
    def _calculate_capital_efficiency(self, lower_tick: int, upper_tick: int, pool_data: Dict[str, Any]) -> float:
        """Calculate capital efficiency score for tick range"""
        try:
            # Get current tick and price
            current_tick = pool_data.get('current_tick', 0)
            
            # Calculate range efficiency
            tick_range = upper_tick - lower_tick
            distance_from_current = min(
                abs(current_tick - lower_tick),
                abs(current_tick - upper_tick)
            )
            
            # Efficiency inversely related to range size and distance from current price
            base_efficiency = 1.0 / (1.0 + tick_range / 1000.0)
            proximity_bonus = 1.0 / (1.0 + distance_from_current / 1000.0)
            
            return base_efficiency * proximity_bonus
            
        except Exception:
            return 0.5  # Default efficiency
    
    def _heuristic_tick_optimization(
        self, 
        pool_data: Dict[str, Any], 
        capital_amount: float, 
        risk_tolerance: str
    ) -> PositionOptimization:
        """Fallback heuristic optimization when ML is not available"""
        
        current_tick = pool_data.get('current_tick', 0)
        volatility = pool_data.get('price_volatility_24h', 0.1)
        
        # Adjust range based on risk tolerance
        risk_multipliers = {'low': 3.0, 'medium': 2.0, 'high': 1.0}
        multiplier = risk_multipliers.get(risk_tolerance, 2.0)
        
        # Calculate tick range based on volatility
        tick_range = int(volatility * 10000 * multiplier)  # Heuristic formula
        
        lower_tick = current_tick - tick_range
        upper_tick = current_tick + tick_range
        
        # Ensure ticks are valid
        lower_tick = max(lower_tick, self.MIN_TICK)
        upper_tick = min(upper_tick, self.MAX_TICK)
        
        return PositionOptimization(
            optimal_lower_tick=lower_tick,
            optimal_upper_tick=upper_tick,
            predicted_apr=0.15,  # Conservative estimate
            predicted_il=volatility * 0.5,  # Rough IL estimate
            capital_efficiency=self._calculate_capital_efficiency(lower_tick, upper_tick, pool_data),
            risk_score=volatility,
            confidence=0.6,  # Lower confidence for heuristic
            reasoning="Heuristic optimization based on volatility and risk tolerance"
        )
    
    def _get_conservative_default(self, pool_address: str) -> PositionOptimization:
        """Conservative default optimization when data is insufficient"""
        return PositionOptimization(
            optimal_lower_tick=-1000,
            optimal_upper_tick=1000,
            predicted_apr=0.1,
            predicted_il=0.05,
            capital_efficiency=0.5,
            risk_score=0.3,
            confidence=0.3,
            reasoning="Conservative default due to insufficient data"
        )
    
    def _tick_to_price(self, tick: int) -> float:
        """Convert tick to price"""
        return 1.0001 ** tick
    
    def _calculate_volatility(self, price_history: List[float], hours: int) -> float:
        """Calculate price volatility over specified hours"""
        if not price_history or len(price_history) < 2:
            return 0.1  # Default volatility
        
        # Take last N hours of data
        recent_prices = price_history[-hours:] if len(price_history) >= hours else price_history
        
        # Calculate returns
        returns = [
            (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            for i in range(1, len(recent_prices))
        ]
        
        return float(np.std(returns)) if returns else 0.1
    
    async def get_price_history(self, pool_address: str, days: int = 7) -> List[float]:
        """Get historical price data for volatility calculations"""
        # Simplified implementation - in production, this would query historical data
        # For now, return mock data
        base_price = 1.0
        hours = days * 24
        
        # Generate realistic price movements
        np.random.seed(42)  # For consistent results
        price_changes = np.random.normal(0, 0.02, hours)  # 2% hourly volatility
        
        prices = [base_price]
        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.1))  # Prevent negative prices
        
        return prices
    
    async def get_historical_fee_data(self, pool_address: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical fee data for earnings prediction"""
        # Simplified implementation - would query real historical data
        return [
            {
                'timestamp': datetime.now() - timedelta(days=i),
                'fees_usd': 100 + np.random.normal(0, 20),  # Mock fee data
                'volume_usd': 10000 + np.random.normal(0, 2000)
            }
            for i in range(days)
        ]