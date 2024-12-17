from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

class ExecutionType(Enum):
    EVENT_BASED = "event"
    PERIODIC = "periodic"
    CONTRACT_AUTOMATION = "contract"

@dataclass
class ExecutionConfig:
    type: ExecutionType
    interval: Optional[int] = None  # seconds for periodic
    event_names: Optional[List[str]] = None
    contract_address: Optional[str] = None
    min_interval: Optional[int] = None    # minimum time between executions
    max_retries: int = 3                  # maximum retry attempts
    retry_delay: int = 5                  # seconds between retries

# Configuration for different execution types
EXECUTION_CONFIG: Dict[str, ExecutionConfig] = {
    "price_updates": ExecutionConfig(
        type=ExecutionType.EVENT_BASED,
        event_names=["Swap"],
        min_interval=1,  # Minimum 1 second between updates
        max_retries=3
    ),
    
    "position_rebalance": ExecutionConfig(
        type=ExecutionType.PERIODIC,
        interval=3600,  # Every hour
        min_interval=1800,  # Minimum 30 minutes between rebalances
        max_retries=5,
        retry_delay=10
    ),
    
    "emergency_checks": ExecutionConfig(
        type=ExecutionType.PERIODIC,
        interval=300,  # Every 5 minutes
        min_interval=60,  # Minimum 1 minute between checks
        max_retries=3,
        retry_delay=2
    ),
    
    "liquidity_monitoring": ExecutionConfig(
        type=ExecutionType.EVENT_BASED,
        event_names=["Mint", "Burn"],
        min_interval=5,
        max_retries=3
    ),
    
    "gas_price_monitoring": ExecutionConfig(
        type=ExecutionType.PERIODIC,
        interval=60,  # Every minute
        min_interval=30,
        max_retries=2
    ),
    
    "daily_metrics": ExecutionConfig(
        type=ExecutionType.PERIODIC,
        interval=86400,  # Every 24 hours
        min_interval=43200,  # Minimum 12 hours between runs
        max_retries=5,
        retry_delay=300  # 5 minutes between retries
    )
}

# Execution priorities (lower number = higher priority)
EXECUTION_PRIORITIES = {
    "emergency_checks": 1,
    "gas_price_monitoring": 2,
    "price_updates": 3,
    "liquidity_monitoring": 4,
    "position_rebalance": 5,
    "daily_metrics": 6
}

# Maximum concurrent executions by type
MAX_CONCURRENT_EXECUTIONS = {
    ExecutionType.EVENT_BASED: 5,
    ExecutionType.PERIODIC: 3,
    ExecutionType.CONTRACT_AUTOMATION: 2
}

# Resource limits
RESOURCE_LIMITS = {
    "max_memory_percent": 85,
    "max_cpu_percent": 90,
    "max_disk_percent": 95
}

# Error thresholds
ERROR_THRESHOLDS = {
    "max_consecutive_failures": 5,
    "error_cooldown_period": 300,  # 5 minutes
    "max_daily_errors": 100
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_execution_interval": 0.1,  # seconds
    "max_execution_time": 30,       # seconds
    "warning_execution_time": 10    # seconds
}
