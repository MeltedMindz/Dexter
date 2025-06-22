"""
Structured Logging Configuration for Dexter Protocol
Defines log categories and formats for vault operations and DexBrain intelligence
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class LogCategory(Enum):
    """Log categories for different system components"""
    VAULT_STRATEGY = "vault_strategy"
    VAULT_OPTIMIZATION = "vault_optimization"
    COMPOUND_SUCCESS = "compound_success"
    COMPOUND_FAILED = "compound_failed"
    COMPOUND_OPPORTUNITIES = "compound_opportunities"
    VAULT_INTELLIGENCE = "vault_intelligence"
    INTELLIGENCE_FEED = "intelligence_feed"
    AI_PREDICTION = "ai_prediction"
    RANGE_OPTIMIZATION = "range_optimization"
    PERFORMANCE_TRACKING = "performance_tracking"
    GAMMA_OPTIMIZATION = "gamma_optimization"
    MULTI_RANGE = "multi_range"
    FEE_COLLECTION = "fee_collection"
    GAS_OPTIMIZATION = "gas_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    NETWORK = "network"
    SYSTEM = "system"
    ERROR = "error"

class StructuredLogger:
    """Structured logger for vault operations and DexBrain intelligence"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers if they don't exist
        if not self.logger.handlers:
            # File handler for vault operations
            vault_handler = logging.FileHandler('/opt/dexter-ai/vault-operations.log')
            vault_handler.setLevel(logging.INFO)
            
            # Console handler for development
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # JSON formatter for structured logging
            formatter = StructuredFormatter()
            vault_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(vault_handler)
            self.logger.addHandler(console_handler)
    
    def log_vault_strategy(
        self,
        strategy_type: str,
        confidence: float,
        expected_apr: float,
        ranges_count: int,
        vault_address: str = None,
        execution_time: float = None,
        **kwargs
    ):
        """Log vault strategy prediction"""
        metadata = {
            'strategy_type': strategy_type,
            'confidence': confidence,
            'expected_apr': expected_apr,
            'ranges_count': ranges_count,
            'vault_address': vault_address,
            'execution_time': execution_time,
            **kwargs
        }
        
        message = (f"Strategy prediction complete | "
                  f"Recommended: {strategy_type} | "
                  f"Confidence: {confidence:.2%} | "
                  f"Expected APR: {expected_apr:.2%} | "
                  f"Ranges: {ranges_count}")
        
        if execution_time:
            message += f" | Execution Time: {execution_time:.3f}s"
        
        self._log_structured(LogCategory.VAULT_STRATEGY, message, metadata)
    
    def log_compound_success(
        self,
        token_id: int,
        tx_hash: str,
        gas_used: int,
        gas_cost_usd: float,
        fees_compounded_usd: float,
        net_profit: float,
        strategy: str,
        execution_time: float = None,
        **kwargs
    ):
        """Log successful compound operation"""
        metadata = {
            'token_id': token_id,
            'tx_hash': tx_hash,
            'gas_used': gas_used,
            'gas_cost_usd': gas_cost_usd,
            'fees_compounded_usd': fees_compounded_usd,
            'net_profit': net_profit,
            'strategy': strategy,
            'execution_time': execution_time,
            **kwargs
        }
        
        message = (f"COMPOUND_SUCCESS | "
                  f"Token ID: {token_id} | "
                  f"TX Hash: {tx_hash} | "
                  f"Gas Used: {gas_used:,} | "
                  f"Gas Cost: ${gas_cost_usd:.2f} | "
                  f"Fees Compounded: ${fees_compounded_usd:.2f} | "
                  f"Net Profit: ${net_profit:.2f}")
        
        if execution_time:
            message += f" | Execution Time: {execution_time:.3f}s"
        
        self._log_structured(LogCategory.COMPOUND_SUCCESS, message, metadata)
    
    def log_compound_opportunities(
        self,
        opportunities_count: int,
        total_profit_potential: float,
        top_priority_score: float,
        **kwargs
    ):
        """Log compound opportunities found"""
        metadata = {
            'opportunities_count': opportunities_count,
            'total_profit_potential': total_profit_potential,
            'top_priority_score': top_priority_score,
            **kwargs
        }
        
        message = (f"Found {opportunities_count} compound opportunities | "
                  f"Total Profit Potential: ${total_profit_potential:,.2f} | "
                  f"Top Priority Score: {top_priority_score:.2f}")
        
        self._log_structured(LogCategory.COMPOUND_OPPORTUNITIES, message, metadata)
    
    def log_vault_intelligence(
        self,
        vault_address: str,
        strategy: str,
        confidence: float,
        compound_opportunities: int,
        execution_time: float = None,
        **kwargs
    ):
        """Log vault intelligence generation"""
        metadata = {
            'vault_address': vault_address,
            'strategy': strategy,
            'confidence': confidence,
            'compound_opportunities': compound_opportunities,
            'execution_time': execution_time,
            **kwargs
        }
        
        message = (f"Generated vault intelligence for {vault_address} | "
                  f"Strategy: {strategy} | "
                  f"Confidence: {confidence:.2%} | "
                  f"Compound Opportunities: {compound_opportunities}")
        
        if execution_time:
            message += f" | Execution Time: {execution_time:.3f}s"
        
        self._log_structured(LogCategory.VAULT_INTELLIGENCE, message, metadata)
    
    def log_gamma_optimization(
        self,
        base_range: tuple,
        base_allocation: float,
        limit_range: tuple,
        limit_allocation: float,
        current_tick: int,
        volatility: float = None,
        execution_time: float = None,
        **kwargs
    ):
        """Log Gamma-style dual position optimization"""
        metadata = {
            'base_range': base_range,
            'base_allocation': base_allocation,
            'limit_range': limit_range,
            'limit_allocation': limit_allocation,
            'current_tick': current_tick,
            'volatility': volatility,
            'execution_time': execution_time,
            **kwargs
        }
        
        message = (f"Dual position optimization complete | "
                  f"Base Range: [{base_range[0]}, {base_range[1]}] ({base_allocation:.2%}) | "
                  f"Limit Range: [{limit_range[0]}, {limit_range[1]}] ({limit_allocation:.2%})")
        
        if volatility:
            message += f" | Volatility: {volatility:.4f}"
        
        if execution_time:
            message += f" | Execution Time: {execution_time:.3f}s"
        
        self._log_structured(LogCategory.GAMMA_OPTIMIZATION, message, metadata)
    
    def log_intelligence_feed(
        self,
        agent_id: str,
        insights_count: int,
        predictions_count: int,
        quality_score: float,
        **kwargs
    ):
        """Log intelligence feed served to agent"""
        metadata = {
            'agent_id': agent_id,
            'insights_count': insights_count,
            'predictions_count': predictions_count,
            'quality_score': quality_score,
            **kwargs
        }
        
        message = (f"Intelligence served to agent {agent_id} | "
                  f"Insights: {insights_count} | "
                  f"Predictions: {predictions_count} | "
                  f"Quality Score: {quality_score:.2%}")
        
        self._log_structured(LogCategory.INTELLIGENCE_FEED, message, metadata)
    
    def log_performance_tracking(
        self,
        metric_type: str,
        value: float,
        vault_address: str = None,
        time_period: str = None,
        **kwargs
    ):
        """Log performance tracking metrics"""
        metadata = {
            'metric_type': metric_type,
            'value': value,
            'vault_address': vault_address,
            'time_period': time_period,
            **kwargs
        }
        
        message = f"Performance metric | {metric_type}: {value}"
        if vault_address:
            message += f" | Vault: {vault_address}"
        if time_period:
            message += f" | Period: {time_period}"
        
        self._log_structured(LogCategory.PERFORMANCE_TRACKING, message, metadata)
    
    def log_error(self, error_message: str, error_type: str = None, **kwargs):
        """Log error with context"""
        metadata = {
            'error_type': error_type,
            **kwargs
        }
        
        self._log_structured(LogCategory.ERROR, error_message, metadata, level=logging.ERROR)
    
    def _log_structured(
        self, 
        category: LogCategory, 
        message: str, 
        metadata: Dict[str, Any], 
        level: int = logging.INFO
    ):
        """Internal method to log structured data"""
        extra = {
            'category': category.value,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.log(level, message, extra=extra)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'type': getattr(record, 'category', 'log')
        }
        
        # Add metadata if present
        if hasattr(record, 'metadata'):
            log_entry['metadata'] = record.metadata
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)

# Global logger instances for different components
vault_logger = StructuredLogger('VaultMLEngine')
compound_logger = StructuredLogger('CompoundService')
dexbrain_logger = StructuredLogger('DexBrain')
gamma_logger = StructuredLogger('GammaStyleOptimizer')
performance_logger = StructuredLogger('PerformanceTracker')

# Convenience functions for easy import
def log_vault_strategy(*args, **kwargs):
    vault_logger.log_vault_strategy(*args, **kwargs)

def log_compound_success(*args, **kwargs):
    compound_logger.log_compound_success(*args, **kwargs)

def log_compound_opportunities(*args, **kwargs):
    compound_logger.log_compound_opportunities(*args, **kwargs)

def log_vault_intelligence(*args, **kwargs):
    dexbrain_logger.log_vault_intelligence(*args, **kwargs)

def log_gamma_optimization(*args, **kwargs):
    gamma_logger.log_gamma_optimization(*args, **kwargs)

def log_intelligence_feed(*args, **kwargs):
    dexbrain_logger.log_intelligence_feed(*args, **kwargs)

def log_performance_tracking(*args, **kwargs):
    performance_logger.log_performance_tracking(*args, **kwargs)

def log_error(component: str, *args, **kwargs):
    """Log error from any component"""
    logger = StructuredLogger(component)
    logger.log_error(*args, **kwargs)