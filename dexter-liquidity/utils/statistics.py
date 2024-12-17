import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import logging
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
STRATEGY_EXECUTIONS = Counter('strategy_executions_total', 'Number of strategy executions', ['agent_type'])
EXECUTION_DURATION = Histogram('execution_duration_seconds', 'Time spent executing strategies')
ACTIVE_POSITIONS = Gauge('active_positions', 'Number of active positions', ['agent_type'])
TOTAL_VALUE_LOCKED = Gauge('total_value_locked', 'Total value locked in positions', ['agent_type'])

@dataclass
class ExecutionStats:
    timestamp: datetime
    duration: float
    success: bool
    agent_type: str
    trigger: str
    gas_used: float
    value_locked: float

class StatisticsCollector:
    def __init__(self):
        self.execution_history: List[ExecutionStats] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'execution_times': [],
            'success_rates': [],
            'gas_costs': []
        }
        
    async def record_execution(
        self,
        agent_type: str,
        trigger: str,
        start_time: float,
        success: bool,
        gas_used: float,
        value_locked: float
    ):
        """Record execution statistics"""
        duration = time.time() - start_time
        
        stats = ExecutionStats(
            timestamp=datetime.now(),
            duration=duration,
            success=success,
            agent_type=agent_type,
            trigger=trigger,
            gas_used=gas_used,
            value_locked=value_locked
        )
        
        self.execution_history.append(stats)
        
        # Update Prometheus metrics
        STRATEGY_EXECUTIONS.labels(agent_type=agent_type).inc()
        EXECUTION_DURATION.observe(duration)
        ACTIVE_POSITIONS.labels(agent_type=agent_type).set(value_locked)
        TOTAL_VALUE_LOCKED.labels(agent_type=agent_type).set(value_locked)
        
        # Update performance metrics
        self.performance_metrics['execution_times'].append(duration)
        self.performance_metrics['success_rates'].append(1 if success else 0)
        self.performance_metrics['gas_costs'].append(gas_used)
        
        logger.info(
            f"Recorded execution - Agent: {agent_type}, Duration: {duration:.2f}s, "
            f"Success: {success}, Gas: {gas_used:.2f}"
        )
        
    def get_summary_statistics(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for recent executions"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_stats = [
            stat for stat in self.execution_history
            if stat.timestamp >= cutoff_time
        ]
        
        if not recent_stats:
            return {}
            
        return {
            'execution_count': len(recent_stats),
            'success_rate': np.mean([stat.success for stat in recent_stats]),
            'avg_duration': np.mean([stat.duration for stat in recent_stats]),
            'avg_gas_cost': np.mean([stat.gas_used for stat in recent_stats]),
            'total_value_locked': sum(stat.value_locked for stat in recent_stats),
            'by_agent': self._get_agent_breakdown(recent_stats),
            'by_trigger': self._get_trigger_breakdown(recent_stats)
        }
        
    def _get_agent_breakdown(self, stats: List[ExecutionStats]) -> Dict[str, Dict[str, float]]:
        """Get statistics breakdown by agent type"""
        breakdown = {}
        for agent_type in set(stat.agent_type for stat in stats):
            agent_stats = [stat for stat in stats if stat.agent_type == agent_type]
            breakdown[agent_type] = {
                'execution_count': len(agent_stats),
                'success_rate': np.mean([stat.success for stat in agent_stats]),
                'avg_duration': np.mean([stat.duration for stat in agent_stats]),
                'avg_gas_cost': np.mean([stat.gas_used for stat in agent_stats]),
                'value_locked': sum(stat.value_locked for stat in agent_stats)
            }
        return breakdown
        
    def _get_trigger_breakdown(self, stats: List[ExecutionStats]) -> Dict[str, Dict[str, float]]:
        """Get statistics breakdown by trigger type"""
        breakdown = {}
        for trigger in set(stat.trigger for stat in stats):
            trigger_stats = [stat for stat in stats if stat.trigger == trigger]
            breakdown[trigger] = {
                'execution_count': len(trigger_stats),
                'success_rate': np.mean([stat.success for stat in trigger_stats]),
                'avg_duration': np.mean([stat.duration for stat in trigger_stats])
            }
        return breakdown
