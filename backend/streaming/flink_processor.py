"""
Apache Flink Stream Processor for Real-time DeFi Analytics
High-performance stream processing for ML feature engineering and real-time predictions
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import math
import os

# Import ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

from ..dexbrain.config import Config
from .kafka_producer import DexterKafkaProducer, MLPredictionEvent

logger = logging.getLogger(__name__)

@dataclass
class StreamingFeatures:
    """Real-time feature set for ML models"""
    pool_address: str
    timestamp: int
    
    # Price features
    price: float
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    price_volatility_1h: float
    price_volatility_24h: float
    
    # Volume features
    volume_1m: float
    volume_5m: float
    volume_15m: float
    volume_1h: float
    volume_24h: float
    volume_ma_10m: float
    volume_ma_1h: float
    
    # Liquidity features
    liquidity: float
    liquidity_change_1m: float
    liquidity_change_5m: float
    liquidity_utilization: float
    tick_range_width: int
    active_tick_range: int
    
    # Market microstructure
    bid_ask_spread: float
    order_flow_imbalance: float
    trade_frequency_1m: int
    large_trade_ratio: float
    
    # Technical indicators
    rsi_14: float
    bollinger_upper: float
    bollinger_lower: float
    ema_12: float
    ema_26: float
    macd: float
    
    # DeFi specific
    total_value_locked: float
    fees_24h: float
    impermanent_loss_risk: float
    capital_efficiency: float
    position_concentration: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML processing"""
        return np.array([
            self.price, self.price_change_1m, self.price_change_5m, self.price_change_15m,
            self.price_volatility_1h, self.price_volatility_24h,
            self.volume_1m, self.volume_5m, self.volume_15m, self.volume_1h, self.volume_24h,
            self.volume_ma_10m, self.volume_ma_1h,
            self.liquidity, self.liquidity_change_1m, self.liquidity_change_5m,
            self.liquidity_utilization, self.tick_range_width, self.active_tick_range,
            self.bid_ask_spread, self.order_flow_imbalance, self.trade_frequency_1m, self.large_trade_ratio,
            self.rsi_14, self.bollinger_upper, self.bollinger_lower, self.ema_12, self.ema_26, self.macd,
            self.total_value_locked, self.fees_24h, self.impermanent_loss_risk,
            self.capital_efficiency, self.position_concentration
        ])

class RealTimeFeatureEngine:
    """Real-time feature engineering engine using Flink-style stream processing"""
    
    def __init__(self, window_size: int = 1440):  # 24 hours of minute data
        self.window_size = window_size
        
        # Time series windows for each pool
        self.price_windows = {}
        self.volume_windows = {}
        self.liquidity_windows = {}
        self.trade_windows = {}
        
        # Feature scalers
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Performance tracking
        self.features_generated = 0
        self.processing_times = []
        
        logger.info("Real-time feature engine initialized")
    
    def add_pool_event(self, pool_address: str, event_data: Dict[str, Any]):
        """Add new pool event to streaming windows"""
        timestamp = event_data.get('timestamp', int(time.time()))
        
        # Initialize windows if new pool
        if pool_address not in self.price_windows:
            self._initialize_pool_windows(pool_address)
        
        # Add data to appropriate windows
        if event_data.get('event_type') == 'swap':
            self._add_swap_data(pool_address, timestamp, event_data)
        elif event_data.get('event_type') in ['mint', 'burn']:
            self._add_liquidity_data(pool_address, timestamp, event_data)
        elif event_data.get('event_type') == 'collect':
            self._add_fee_data(pool_address, timestamp, event_data)
    
    def _initialize_pool_windows(self, pool_address: str):
        """Initialize time series windows for new pool"""
        self.price_windows[pool_address] = deque(maxlen=self.window_size)
        self.volume_windows[pool_address] = deque(maxlen=self.window_size)
        self.liquidity_windows[pool_address] = deque(maxlen=self.window_size)
        self.trade_windows[pool_address] = deque(maxlen=self.window_size)
    
    def _add_swap_data(self, pool_address: str, timestamp: int, data: Dict[str, Any]):
        """Add swap event data to windows"""
        price_point = {
            'timestamp': timestamp,
            'price': data.get('price', 0),
            'volume': data.get('volume_usd', 0),
            'tick': data.get('tick', 0)
        }
        
        trade_point = {
            'timestamp': timestamp,
            'amount_usd': data.get('volume_usd', 0),
            'is_large_trade': data.get('volume_usd', 0) > 10000  # $10k threshold
        }
        
        self.price_windows[pool_address].append(price_point)
        self.trade_windows[pool_address].append(trade_point)
    
    def _add_liquidity_data(self, pool_address: str, timestamp: int, data: Dict[str, Any]):
        """Add liquidity event data to windows"""
        liquidity_point = {
            'timestamp': timestamp,
            'liquidity': data.get('liquidity', 0),
            'liquidity_delta': data.get('liquidity_delta', 0),
            'tick_lower': data.get('tick_lower', 0),
            'tick_upper': data.get('tick_upper', 0)
        }
        
        self.liquidity_windows[pool_address].append(liquidity_point)
    
    def _add_fee_data(self, pool_address: str, timestamp: int, data: Dict[str, Any]):
        """Add fee collection data to windows"""
        # Fee data can be added to volume windows as revenue
        volume_point = {
            'timestamp': timestamp,
            'fee_amount': data.get('fees_usd', 0),
            'collected_amount0': data.get('amount0', 0),
            'collected_amount1': data.get('amount1', 0)
        }
        
        self.volume_windows[pool_address].append(volume_point)
    
    def generate_features(self, pool_address: str) -> Optional[StreamingFeatures]:
        """Generate real-time features for a pool"""
        start_time = time.time()
        
        try:
            if pool_address not in self.price_windows:
                return None
            
            current_time = int(time.time())
            
            # Get current data points
            price_data = list(self.price_windows[pool_address])
            volume_data = list(self.volume_windows[pool_address])
            liquidity_data = list(self.liquidity_windows[pool_address])
            trade_data = list(self.trade_windows[pool_address])
            
            if not price_data:
                return None
            
            # Calculate features
            features = StreamingFeatures(
                pool_address=pool_address,
                timestamp=current_time,
                
                # Price features
                price=self._get_latest_price(price_data),
                price_change_1m=self._calculate_price_change(price_data, 60),
                price_change_5m=self._calculate_price_change(price_data, 300),
                price_change_15m=self._calculate_price_change(price_data, 900),
                price_volatility_1h=self._calculate_volatility(price_data, 3600),
                price_volatility_24h=self._calculate_volatility(price_data, 86400),
                
                # Volume features
                volume_1m=self._calculate_volume(volume_data, 60),
                volume_5m=self._calculate_volume(volume_data, 300),
                volume_15m=self._calculate_volume(volume_data, 900),
                volume_1h=self._calculate_volume(volume_data, 3600),
                volume_24h=self._calculate_volume(volume_data, 86400),
                volume_ma_10m=self._calculate_volume_ma(volume_data, 600, 10),
                volume_ma_1h=self._calculate_volume_ma(volume_data, 3600, 60),
                
                # Liquidity features
                liquidity=self._get_latest_liquidity(liquidity_data),
                liquidity_change_1m=self._calculate_liquidity_change(liquidity_data, 60),
                liquidity_change_5m=self._calculate_liquidity_change(liquidity_data, 300),
                liquidity_utilization=self._calculate_liquidity_utilization(price_data, liquidity_data),
                tick_range_width=self._calculate_tick_range_width(liquidity_data),
                active_tick_range=self._calculate_active_tick_range(price_data, liquidity_data),
                
                # Market microstructure
                bid_ask_spread=self._estimate_bid_ask_spread(price_data, volume_data),
                order_flow_imbalance=self._calculate_order_flow_imbalance(trade_data),
                trade_frequency_1m=self._calculate_trade_frequency(trade_data, 60),
                large_trade_ratio=self._calculate_large_trade_ratio(trade_data, 3600),
                
                # Technical indicators
                rsi_14=self._calculate_rsi(price_data, 14),
                bollinger_upper=self._calculate_bollinger_bands(price_data, 20)[1],
                bollinger_lower=self._calculate_bollinger_bands(price_data, 20)[0],
                ema_12=self._calculate_ema(price_data, 12),
                ema_26=self._calculate_ema(price_data, 26),
                macd=self._calculate_macd(price_data),
                
                # DeFi specific
                total_value_locked=self._calculate_tvl(price_data, liquidity_data),
                fees_24h=self._calculate_fees_24h(volume_data),
                impermanent_loss_risk=self._calculate_il_risk(price_data),
                capital_efficiency=self._calculate_capital_efficiency(price_data, liquidity_data),
                position_concentration=self._calculate_position_concentration(liquidity_data)
            )
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.features_generated += 1
            
            return features
            
        except Exception as e:
            logger.error(f"Feature generation error for {pool_address}: {e}")
            return None
    
    # Feature calculation methods
    def _get_latest_price(self, price_data: List[Dict]) -> float:
        """Get most recent price"""
        return price_data[-1]['price'] if price_data else 0.0
    
    def _calculate_price_change(self, price_data: List[Dict], window_seconds: int) -> float:
        """Calculate price change over time window"""
        if len(price_data) < 2:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        current_price = price_data[-1]['price']
        
        # Find price at start of window
        for point in reversed(price_data):
            if point['timestamp'] <= cutoff_time:
                old_price = point['price']
                if old_price > 0:
                    return (current_price - old_price) / old_price
                break
        
        return 0.0
    
    def _calculate_volatility(self, price_data: List[Dict], window_seconds: int) -> float:
        """Calculate price volatility over time window"""
        if len(price_data) < 10:  # Need minimum data points
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Get prices in window
        prices = [p['price'] for p in price_data if p['timestamp'] >= cutoff_time]
        
        if len(prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.0
        
        return np.std(returns) * math.sqrt(len(returns))  # Annualized volatility
    
    def _calculate_volume(self, volume_data: List[Dict], window_seconds: int) -> float:
        """Calculate volume over time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        volume = sum(v.get('volume', 0) for v in volume_data 
                    if v.get('timestamp', 0) >= cutoff_time)
        
        return volume
    
    def _calculate_volume_ma(self, volume_data: List[Dict], window_seconds: int, periods: int) -> float:
        """Calculate volume moving average"""
        if len(volume_data) < periods:
            return 0.0
        
        recent_volumes = []
        current_time = time.time()
        
        for i in range(periods):
            start_time = current_time - (i + 1) * window_seconds
            end_time = current_time - i * window_seconds
            
            period_volume = sum(v.get('volume', 0) for v in volume_data 
                              if start_time <= v.get('timestamp', 0) < end_time)
            recent_volumes.append(period_volume)
        
        return np.mean(recent_volumes) if recent_volumes else 0.0
    
    def _get_latest_liquidity(self, liquidity_data: List[Dict]) -> float:
        """Get most recent liquidity"""
        return liquidity_data[-1]['liquidity'] if liquidity_data else 0.0
    
    def _calculate_liquidity_change(self, liquidity_data: List[Dict], window_seconds: int) -> float:
        """Calculate liquidity change over time window"""
        if len(liquidity_data) < 2:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        current_liquidity = liquidity_data[-1]['liquidity']
        
        for point in reversed(liquidity_data):
            if point['timestamp'] <= cutoff_time:
                old_liquidity = point['liquidity']
                if old_liquidity > 0:
                    return (current_liquidity - old_liquidity) / old_liquidity
                break
        
        return 0.0
    
    def _calculate_liquidity_utilization(self, price_data: List[Dict], liquidity_data: List[Dict]) -> float:
        """Calculate how much of available liquidity is being utilized"""
        if not price_data or not liquidity_data:
            return 0.0
        
        current_tick = price_data[-1].get('tick', 0)
        
        # Find active liquidity around current tick
        active_liquidity = 0
        total_liquidity = 0
        
        for liq_point in liquidity_data[-10:]:  # Recent liquidity positions
            tick_lower = liq_point.get('tick_lower', 0)
            tick_upper = liq_point.get('tick_upper', 0)
            liquidity = liq_point.get('liquidity', 0)
            
            total_liquidity += liquidity
            
            if tick_lower <= current_tick <= tick_upper:
                active_liquidity += liquidity
        
        return active_liquidity / max(total_liquidity, 1)
    
    def _calculate_tick_range_width(self, liquidity_data: List[Dict]) -> int:
        """Calculate average tick range width"""
        if not liquidity_data:
            return 0
        
        ranges = []
        for point in liquidity_data[-10:]:
            tick_lower = point.get('tick_lower', 0)
            tick_upper = point.get('tick_upper', 0)
            ranges.append(tick_upper - tick_lower)
        
        return int(np.mean(ranges)) if ranges else 0
    
    def _calculate_active_tick_range(self, price_data: List[Dict], liquidity_data: List[Dict]) -> int:
        """Calculate tick range with active positions"""
        if not price_data or not liquidity_data:
            return 0
        
        current_tick = price_data[-1].get('tick', 0)
        
        active_ranges = []
        for point in liquidity_data[-10:]:
            tick_lower = point.get('tick_lower', 0)
            tick_upper = point.get('tick_upper', 0)
            
            if tick_lower <= current_tick <= tick_upper:
                active_ranges.append(tick_upper - tick_lower)
        
        return int(np.mean(active_ranges)) if active_ranges else 0
    
    def _estimate_bid_ask_spread(self, price_data: List[Dict], volume_data: List[Dict]) -> float:
        """Estimate bid-ask spread from price and volume data"""
        if len(price_data) < 2:
            return 0.0
        
        # Simple estimation based on price volatility and volume
        recent_prices = [p['price'] for p in price_data[-10:]]
        price_range = max(recent_prices) - min(recent_prices)
        avg_price = np.mean(recent_prices)
        
        if avg_price > 0:
            return price_range / avg_price
        
        return 0.0
    
    def _calculate_order_flow_imbalance(self, trade_data: List[Dict]) -> float:
        """Calculate order flow imbalance"""
        if len(trade_data) < 2:
            return 0.0
        
        # Simplified calculation based on trade sizes
        recent_trades = trade_data[-20:]
        
        large_trades = sum(1 for t in recent_trades if t.get('is_large_trade', False))
        total_trades = len(recent_trades)
        
        return large_trades / max(total_trades, 1)
    
    def _calculate_trade_frequency(self, trade_data: List[Dict], window_seconds: int) -> int:
        """Calculate trade frequency over time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        trades = sum(1 for t in trade_data if t.get('timestamp', 0) >= cutoff_time)
        return trades
    
    def _calculate_large_trade_ratio(self, trade_data: List[Dict], window_seconds: int) -> float:
        """Calculate ratio of large trades"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_trades = [t for t in trade_data if t.get('timestamp', 0) >= cutoff_time]
        
        if not recent_trades:
            return 0.0
        
        large_trades = sum(1 for t in recent_trades if t.get('is_large_trade', False))
        return large_trades / len(recent_trades)
    
    def _calculate_rsi(self, price_data: List[Dict], periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(price_data) < periods + 1:
            return 50.0  # Neutral RSI
        
        prices = [p['price'] for p in price_data[-periods-1:]]
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, price_data: List[Dict], periods: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands (lower, upper)"""
        if len(price_data) < periods:
            current_price = self._get_latest_price(price_data)
            return current_price * 0.98, current_price * 1.02  # 2% bands
        
        prices = [p['price'] for p in price_data[-periods:]]
        ma = np.mean(prices)
        std = np.std(prices)
        
        lower_band = ma - (2 * std)
        upper_band = ma + (2 * std)
        
        return lower_band, upper_band
    
    def _calculate_ema(self, price_data: List[Dict], periods: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(price_data) < periods:
            return self._get_latest_price(price_data)
        
        prices = [p['price'] for p in price_data[-periods:]]
        
        # Calculate EMA
        multiplier = 2 / (periods + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, price_data: List[Dict]) -> float:
        """Calculate MACD (12-day EMA - 26-day EMA)"""
        ema_12 = self._calculate_ema(price_data, 12)
        ema_26 = self._calculate_ema(price_data, 26)
        
        return ema_12 - ema_26
    
    def _calculate_tvl(self, price_data: List[Dict], liquidity_data: List[Dict]) -> float:
        """Calculate Total Value Locked"""
        if not price_data or not liquidity_data:
            return 0.0
        
        current_price = price_data[-1]['price']
        current_liquidity = liquidity_data[-1]['liquidity'] if liquidity_data else 0
        
        # Simplified TVL calculation
        return current_price * current_liquidity
    
    def _calculate_fees_24h(self, volume_data: List[Dict]) -> float:
        """Calculate fees collected in last 24 hours"""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 hours
        
        fees = sum(v.get('fee_amount', 0) for v in volume_data 
                  if v.get('timestamp', 0) >= cutoff_time)
        
        return fees
    
    def _calculate_il_risk(self, price_data: List[Dict]) -> float:
        """Calculate impermanent loss risk based on price volatility"""
        volatility_24h = self._calculate_volatility(price_data, 86400)
        
        # Higher volatility = higher IL risk
        # Scale from 0 to 1
        il_risk = min(volatility_24h / 0.5, 1.0)  # 50% volatility = max risk
        
        return il_risk
    
    def _calculate_capital_efficiency(self, price_data: List[Dict], liquidity_data: List[Dict]) -> float:
        """Calculate capital efficiency of positions"""
        utilization = self._calculate_liquidity_utilization(price_data, liquidity_data)
        
        # Capital efficiency is inverse of range width and proportional to utilization
        return utilization * 100  # Scale to 0-100
    
    def _calculate_position_concentration(self, liquidity_data: List[Dict]) -> float:
        """Calculate how concentrated liquidity positions are"""
        if not liquidity_data:
            return 0.0
        
        # Calculate concentration based on tick range widths
        recent_ranges = []
        for point in liquidity_data[-10:]:
            tick_lower = point.get('tick_lower', 0)
            tick_upper = point.get('tick_upper', 0)
            recent_ranges.append(tick_upper - tick_lower)
        
        if not recent_ranges:
            return 0.0
        
        avg_range = np.mean(recent_ranges)
        
        # Smaller ranges = higher concentration
        # Scale inversely (larger number = more concentrated)
        concentration = 10000 / max(avg_range, 100)  # Prevent division by zero
        
        return min(concentration, 100)  # Cap at 100
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get feature engine performance metrics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'features_generated': self.features_generated,
            'average_processing_time': avg_processing_time,
            'pools_tracked': len(self.price_windows),
            'memory_usage_mb': sum(len(w) for w in self.price_windows.values()) * 0.001  # Rough estimate
        }

class FlinkStreamProcessor:
    """
    Apache Flink-style stream processor for real-time DeFi analytics
    Processes incoming Kafka streams and generates ML features and predictions
    """
    
    def __init__(self):
        self.feature_engine = RealTimeFeatureEngine()
        self.kafka_producer = None
        
        # Processing configuration
        self.processing_interval = 60  # Process every minute
        self.batch_size = 100
        
        # State management
        self.running = False
        self.last_processing_time = {}
        
        # Performance tracking
        self.events_processed = 0
        self.predictions_generated = 0
        
        logger.info("Flink stream processor initialized")
    
    async def start(self):
        """Start the stream processor"""
        self.kafka_producer = DexterKafkaProducer()
        await self.kafka_producer.start()
        
        self.running = True
        logger.info("Flink stream processor started")
    
    async def stop(self):
        """Stop the stream processor"""
        self.running = False
        
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        logger.info("Flink stream processor stopped")
    
    async def process_stream_event(self, event_data: Dict[str, Any]):
        """Process incoming stream event"""
        try:
            pool_address = event_data.get('pool_address')
            if not pool_address:
                return
            
            # Add event to feature engine
            self.feature_engine.add_pool_event(pool_address, event_data)
            self.events_processed += 1
            
            # Check if it's time to generate features and predictions
            current_time = time.time()
            last_processed = self.last_processing_time.get(pool_address, 0)
            
            if current_time - last_processed >= self.processing_interval:
                await self._generate_and_send_prediction(pool_address)
                self.last_processing_time[pool_address] = current_time
                
        except Exception as e:
            logger.error(f"Error processing stream event: {e}")
    
    async def _generate_and_send_prediction(self, pool_address: str):
        """Generate features and ML prediction for a pool"""
        try:
            # Generate real-time features
            features = self.feature_engine.generate_features(pool_address)
            
            if not features:
                return
            
            # Generate ML prediction (simplified for demo)
            prediction = self._generate_ml_prediction(features)
            
            if prediction:
                # Send prediction to Kafka
                await self.kafka_producer.send_ml_prediction(prediction)
                self.predictions_generated += 1
                
                logger.debug(f"Generated prediction for {pool_address}: {prediction.action_recommendation}")
                
        except Exception as e:
            logger.error(f"Error generating prediction for {pool_address}: {e}")
    
    def _generate_ml_prediction(self, features: StreamingFeatures) -> Optional[MLPredictionEvent]:
        """Generate ML prediction from features (simplified)"""
        try:
            # Convert features to array
            feature_array = features.to_array()
            
            # Simple rule-based prediction (replace with actual ML model)
            confidence = 0.75
            action = "hold"
            expected_return = 0.05
            risk_score = 0.3
            
            # Simple logic based on features
            if features.price_volatility_1h > 0.2:  # High volatility
                if features.volume_1h > features.volume_ma_1h * 1.5:  # High volume
                    action = "narrow_range"
                    confidence = 0.8
                    risk_score = 0.4
                else:
                    action = "widen_range"
                    confidence = 0.7
                    risk_score = 0.5
            elif features.capital_efficiency < 30:  # Low efficiency
                action = "rebalance"
                confidence = 0.85
                risk_score = 0.2
            elif features.rsi_14 > 70:  # Overbought
                action = "compound"
                confidence = 0.9
                risk_score = 0.1
            
            prediction = MLPredictionEvent(
                pool_address=features.pool_address,
                timestamp=features.timestamp,
                model_name="flink_stream_processor",
                prediction_type="real_time_strategy",
                action_recommendation=action,
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                features_used=[
                    "price_volatility_1h", "volume_1h", "capital_efficiency", "rsi_14"
                ],
                model_version="v1.0.0"
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating ML prediction: {e}")
            return None
    
    def get_processor_metrics(self) -> Dict[str, Any]:
        """Get stream processor performance metrics"""
        feature_metrics = self.feature_engine.get_performance_metrics()
        
        return {
            'events_processed': self.events_processed,
            'predictions_generated': self.predictions_generated,
            'prediction_rate': self.predictions_generated / max(self.events_processed, 1),
            'pools_active': len(self.last_processing_time),
            'is_running': self.running,
            'feature_engine_metrics': feature_metrics
        }

# Factory function
async def create_flink_processor() -> FlinkStreamProcessor:
    """Create and start Flink stream processor"""
    processor = FlinkStreamProcessor()
    await processor.start()
    return processor

# Example usage
async def test_flink_processor():
    """Test the Flink stream processor"""
    processor = await create_flink_processor()
    
    try:
        # Simulate pool events
        test_events = [
            {
                'pool_address': '0x1234567890123456789012345678901234567890',
                'timestamp': int(time.time()),
                'event_type': 'swap',
                'price': 3000.0,
                'volume_usd': 50000.0,
                'tick': 200000,
                'liquidity': 1000000
            },
            {
                'pool_address': '0x1234567890123456789012345678901234567890',
                'timestamp': int(time.time()) + 60,
                'event_type': 'mint',
                'liquidity': 1100000,
                'liquidity_delta': 100000,
                'tick_lower': 199000,
                'tick_upper': 201000
            }
        ]
        
        # Process events
        for event in test_events:
            await processor.process_stream_event(event)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Print metrics
        metrics = processor.get_processor_metrics()
        print(f"Processor metrics: {metrics}")
        
    finally:
        await processor.stop()

if __name__ == "__main__":
    asyncio.run(test_flink_processor())