"""
DLMM Strategy Backtesting Framework
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from decimal import Decimal
import pandas as pd

@dataclass
class Position:
    lower_price: Decimal
    upper_price: Decimal
    fee_tier: Decimal
    liquidity: Decimal
    entry_price: Decimal
    timestamp: pd.Timestamp

@dataclass
class BacktestResult:
    total_return: Decimal
    fees_earned: Decimal
    impermanent_loss: Decimal
    sharpe_ratio: float
    max_drawdown: Decimal
    positions: List[Position]

class DLMMBacktester:
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data
        self.positions: List[Position] = []
        
    def simulate_position(
        self,
        position: Position,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> Dict:
        """Simulate single position performance"""
        period_data = self.data[
            (self.data.index >= start_time) & 
            (self.data.index <= end_time)
        ]
        
        fees_earned = self._calculate_fees(position, period_data)
        il = self._calculate_impermanent_loss(position, period_data)
        total_return = fees_earned - il
        
        return {
            "fees_earned": fees_earned,
            "impermanent_loss": il,
            "total_return": total_return,
            "price_range_exits": self._count_range_exits(position, period_data)
        }

    def backtest_strategy(
        self,
        strategy_generator,
        initial_capital: Decimal,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        rebalance_interval: pd.Timedelta = pd.Timedelta(days=7)
    ) -> BacktestResult:
        """Run full strategy backtest"""
        current_time = start_time
        results = []
        
        while current_time < end_time:
            # Get strategy recommendation
            strategy = strategy_generator.generate_strategy(
                self._get_pool_features(current_time)
            )
            
            # Create and simulate position
            position = Position(
                lower_price=strategy["range"][0],
                upper_price=strategy["range"][1],
                fee_tier=strategy["fee"],
                liquidity=self._calculate_optimal_liquidity(
                    initial_capital,
                    strategy["range"]
                ),
                entry_price=self.data.loc[current_time, "price"],
                timestamp=current_time
            )
            
            result = self.simulate_position(
                position,
                current_time,
                min(current_time + rebalance_interval, end_time)
            )
            
            results.append(result)
            self.positions.append(position)
            current_time += rebalance_interval
            
        return self._calculate_backtest_metrics(results)
    
    def _calculate_fees(
        self,
        position: Position,
        period_data: pd.DataFrame
    ) -> Decimal:
        """Calculate fees earned by position"""
        volume = period_data["volume"].sum()
        in_range_ratio = self._calculate_in_range_ratio(position, period_data)
        return (
            Decimal(str(volume)) *
            position.fee_tier *
            Decimal(str(in_range_ratio))
        )
    
    def _calculate_impermanent_loss(
        self,
        position: Position,
        period_data: pd.DataFrame
    ) -> Decimal:
        """Calculate impermanent loss"""
        start_price = Decimal(str(period_data.iloc[0]["price"]))
        end_price = Decimal(str(period_data.iloc[-1]["price"]))
        price_ratio = end_price / start_price
        
        if position.lower_price <= end_price <= position.upper_price:
            il = 2 * (price_ratio ** Decimal('0.5')) - price_ratio - 1
            return abs(il * position.liquidity)
        return Decimal('0')
    
    def _calculate_in_range_ratio(
        self,
        position: Position,
        period_data: pd.DataFrame
    ) -> float:
        """Calculate what fraction of time price was in range"""
        in_range = (
            (period_data["price"] >= float(position.lower_price)) &
            (period_data["price"] <= float(position.upper_price))
        )
        return in_range.mean()
    
    def _count_range_exits(
        self,
        position: Position,
        period_data: pd.DataFrame
    ) -> int:
        """Count how many times price exited the range"""
        in_range = (
            (period_data["price"] >= float(position.lower_price)) &
            (period_data["price"] <= float(position.upper_price))
        )
        return (~in_range).astype(int).diff().fillna(0).abs().sum() // 2
    
    def _calculate_optimal_liquidity(
        self,
        capital: Decimal,
        price_range: tuple
    ) -> Decimal:
        """Calculate optimal liquidity distribution"""
        range_width = Decimal(str(price_range[1] - price_range[0]))
        return capital / range_width
    
    def _get_pool_features(
        self,
        timestamp: pd.Timestamp
    ) -> Dict:
        """Get pool features for strategy generation"""
        window = self.data[:timestamp].tail(24)  # 24 hour window
        return {
            "volatility": float(window["price"].std()),
            "volume_24h": float(window["volume"].sum()),
            "tvl": float(window["tvl"].iloc[-1]),
            "price_history": window["price"].tolist(),
            "fee_history": window["fees"].tolist()
        }
        
    def _calculate_backtest_metrics(
        self,
        results: List[Dict]
    ) -> BacktestResult:
        """Calculate aggregate backtest metrics"""
        returns = [r["total_return"] for r in results]
        
        return BacktestResult(
            total_return=sum(returns),
            fees_earned=sum(r["fees_earned"] for r in results),
            impermanent_loss=sum(r["impermanent_loss"] for r in results),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(returns),
            positions=self.positions
        )
    
    def _calculate_sharpe_ratio(self, returns: List[Decimal]) -> float:
        """Calculate Sharpe ratio of returns"""
        returns_array = np.array([float(r) for r in returns])
        return float(
            np.mean(returns_array) / np.std(returns_array)
            if np.std(returns_array) != 0
            else 0
        )
    
    def _calculate_max_drawdown(self, returns: List[Decimal]) -> Decimal:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum([float(r) for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return Decimal(str(np.max(drawdowns)))
