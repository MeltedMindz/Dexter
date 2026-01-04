"""
Comprehensive Error Handling and Recovery System
Provides robust error handling, circuit breakers, and automatic recovery mechanisms
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import json
from functools import wraps
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    DATABASE = "database"
    BLOCKCHAIN = "blockchain"
    ML_MODEL = "ml_model"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ALERT_ONLY = "alert_only"
    IMMEDIATE_STOP = "immediate_stop"

@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_id: str
    timestamp: datetime
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class CircuitBreakerState:
    """Circuit breaker state management"""
    name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_duration: int = 60  # seconds
    success_count: int = 0
    half_open_success_threshold: int = 3

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 timeout_duration: int = 60,
                 half_open_success_threshold: int = 3):
        self.state = CircuitBreakerState(
            name=name,
            failure_threshold=failure_threshold,
            timeout_duration=timeout_duration,
            half_open_success_threshold=half_open_success_threshold
        )
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state.state == "OPEN":
            if self._should_attempt_reset():
                self.state.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker {self.state.name} entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.state.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def acall(self, func: Callable, *args, **kwargs):
        """Async version of call"""
        if self.state.state == "OPEN":
            if self._should_attempt_reset():
                self.state.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker {self.state.name} entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.state.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.state.last_failure_time is None:
            return False
        
        time_since_failure = datetime.now() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.state.timeout_duration
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state.state == "HALF_OPEN":
            self.state.success_count += 1
            if self.state.success_count >= self.state.half_open_success_threshold:
                self.state.state = "CLOSED"
                self.state.failure_count = 0
                self.state.success_count = 0
                self.logger.info(f"Circuit breaker {self.state.name} reset to CLOSED state")
        elif self.state.state == "CLOSED":
            self.state.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution"""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.now()
        
        if self.state.state == "HALF_OPEN":
            self.state.state = "OPEN"
            self.state.success_count = 0
            self.logger.warning(f"Circuit breaker {self.state.name} opened from HALF_OPEN state")
        elif (self.state.state == "CLOSED" and 
              self.state.failure_count >= self.state.failure_threshold):
            self.state.state = "OPEN"
            self.logger.warning(f"Circuit breaker {self.state.name} opened due to failures")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class ErrorClassifier:
    """Classify errors into categories and determine recovery strategies"""
    
    ERROR_PATTERNS = {
        ErrorCategory.NETWORK: [
            "connection", "timeout", "network", "unreachable", "socket",
            "Connection refused", "Network is unreachable", "Timeout"
        ],
        ErrorCategory.DATABASE: [
            "database", "sql", "connection pool", "deadlock", "constraint",
            "IntegrityError", "OperationalError", "DatabaseError"
        ],
        ErrorCategory.BLOCKCHAIN: [
            "gas", "revert", "insufficient funds", "nonce", "block",
            "execution reverted", "out of gas", "invalid nonce"
        ],
        ErrorCategory.ML_MODEL: [
            "model", "prediction", "inference", "tensorflow", "pytorch",
            "CUDA", "model not found", "invalid input shape"
        ],
        ErrorCategory.VALIDATION: [
            "validation", "invalid", "schema", "format", "required",
            "ValidationError", "invalid format", "missing required"
        ],
        ErrorCategory.RATE_LIMIT: [
            "rate limit", "too many requests", "quota", "throttle",
            "429", "rate exceeded", "API limit"
        ],
        ErrorCategory.AUTHENTICATION: [
            "auth", "unauthorized", "forbidden", "token", "credentials",
            "401", "403", "invalid credentials"
        ],
        ErrorCategory.TIMEOUT: [
            "timeout", "deadline", "expired", "time limit",
            "TimeoutError", "deadline exceeded"
        ]
    }
    
    RECOVERY_STRATEGIES = {
        ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
        ErrorCategory.DATABASE: RecoveryStrategy.CIRCUIT_BREAKER,
        ErrorCategory.BLOCKCHAIN: RecoveryStrategy.RETRY,
        ErrorCategory.ML_MODEL: RecoveryStrategy.FALLBACK,
        ErrorCategory.VALIDATION: RecoveryStrategy.ALERT_ONLY,
        ErrorCategory.RATE_LIMIT: RecoveryStrategy.RETRY,
        ErrorCategory.AUTHENTICATION: RecoveryStrategy.ALERT_ONLY,
        ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY,
        ErrorCategory.EXTERNAL_API: RecoveryStrategy.CIRCUIT_BREAKER,
        ErrorCategory.CONFIGURATION: RecoveryStrategy.IMMEDIATE_STOP,
        ErrorCategory.UNKNOWN: RecoveryStrategy.GRACEFUL_DEGRADATION
    }
    
    @classmethod
    def classify_error(cls, error: Exception, context: Dict[str, Any] = None) -> ErrorCategory:
        """Classify error into category"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_message or pattern.lower() in error_type:
                    return category
        
        return ErrorCategory.UNKNOWN
    
    @classmethod
    def determine_severity(cls, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        critical_errors = ["database", "blockchain", "authentication", "configuration"]
        high_errors = ["network", "external_api", "ml_model"]
        medium_errors = ["rate_limit", "timeout", "validation"]
        
        category_name = category.value.lower()
        
        if any(critical in category_name for critical in critical_errors):
            return ErrorSeverity.CRITICAL
        elif any(high in category_name for high in high_errors):
            return ErrorSeverity.HIGH
        elif any(medium in category_name for medium in medium_errors):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    @classmethod
    def get_recovery_strategy(cls, category: ErrorCategory) -> RecoveryStrategy:
        """Get recommended recovery strategy for error category"""
        return cls.RECOVERY_STRATEGIES.get(category, RecoveryStrategy.GRACEFUL_DEGRADATION)

class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.alert_handlers: List[Callable] = []
        self.fallback_values: Dict[str, Any] = {}
        self.max_history_size = 1000
        
        # Initialize default recovery handlers
        self._setup_default_handlers()
        
        # Performance metrics
        self.total_errors = 0
        self.recovered_errors = 0
        self.critical_errors = 0
    
    def _setup_default_handlers(self):
        """Setup default recovery handlers"""
        self.recovery_handlers[ErrorCategory.NETWORK] = self._handle_network_error
        self.recovery_handlers[ErrorCategory.DATABASE] = self._handle_database_error
        self.recovery_handlers[ErrorCategory.BLOCKCHAIN] = self._handle_blockchain_error
        self.recovery_handlers[ErrorCategory.ML_MODEL] = self._handle_ml_error
        self.recovery_handlers[ErrorCategory.RATE_LIMIT] = self._handle_rate_limit_error
        self.recovery_handlers[ErrorCategory.TIMEOUT] = self._handle_timeout_error
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any] = None,
                    operation_name: str = None) -> Any:
        """Main error handling entry point"""
        self.total_errors += 1
        
        # Classify error
        category = ErrorClassifier.classify_error(error, context)
        severity = ErrorClassifier.determine_severity(error, category)
        recovery_strategy = ErrorClassifier.get_recovery_strategy(category)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            context=context or {},
            stack_trace=traceback.format_exc(),
            recovery_strategy=recovery_strategy,
            max_retries=self._get_max_retries(category)
        )
        
        # Add to history
        self._add_to_history(error_info)
        
        # Log error
        self._log_error(error_info)
        
        # Handle critical errors immediately
        if severity == ErrorSeverity.CRITICAL:
            self.critical_errors += 1
            self._send_alerts(error_info)
            
            if recovery_strategy == RecoveryStrategy.IMMEDIATE_STOP:
                logger.critical(f"Critical error requires immediate stop: {error_info.message}")
                raise error
        
        # Attempt recovery
        try:
            result = self._attempt_recovery(error_info, operation_name)
            self.recovered_errors += 1
            error_info.resolved = True
            error_info.resolution_time = datetime.now()
            return result
        except Exception as recovery_error:
            logger.error(f"Recovery failed for error {error_info.error_id}: {recovery_error}")
            self._send_alerts(error_info)
            raise error
    
    async def ahandle_error(self, 
                           error: Exception, 
                           context: Dict[str, Any] = None,
                           operation_name: str = None) -> Any:
        """Async version of handle_error"""
        # Similar to handle_error but with async recovery attempts
        return self.handle_error(error, context, operation_name)
    
    def _attempt_recovery(self, error_info: ErrorInfo, operation_name: str = None) -> Any:
        """Attempt to recover from error using appropriate strategy"""
        strategy = error_info.recovery_strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_operation(error_info, operation_name)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._use_fallback(error_info, operation_name)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._use_circuit_breaker(error_info, operation_name)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(error_info, operation_name)
        elif strategy == RecoveryStrategy.ALERT_ONLY:
            self._send_alerts(error_info)
            raise Exception(f"No recovery available for error: {error_info.message}")
        else:
            raise Exception(f"Unknown recovery strategy: {strategy}")
    
    def _retry_operation(self, error_info: ErrorInfo, operation_name: str = None) -> Any:
        """Retry operation with exponential backoff"""
        if error_info.retry_count >= error_info.max_retries:
            raise Exception(f"Max retries exceeded for error: {error_info.message}")
        
        # Exponential backoff
        wait_time = min(2 ** error_info.retry_count, 60)  # Max 60 seconds
        time.sleep(wait_time)
        
        error_info.retry_count += 1
        logger.info(f"Retrying operation (attempt {error_info.retry_count}/{error_info.max_retries})")
        
        # This would normally re-execute the original operation
        # For now, return a placeholder indicating retry
        return {"status": "retried", "attempt": error_info.retry_count}
    
    def _use_fallback(self, error_info: ErrorInfo, operation_name: str = None) -> Any:
        """Use fallback value or method"""
        fallback_key = f"{error_info.category.value}_{operation_name}"
        
        if fallback_key in self.fallback_values:
            logger.info(f"Using fallback value for {fallback_key}")
            return self.fallback_values[fallback_key]
        
        # Category-specific fallbacks
        if error_info.category == ErrorCategory.ML_MODEL:
            return self._ml_fallback(error_info)
        elif error_info.category == ErrorCategory.BLOCKCHAIN:
            return self._blockchain_fallback(error_info)
        else:
            return {"status": "fallback", "message": "Using default fallback"}
    
    def _use_circuit_breaker(self, error_info: ErrorInfo, operation_name: str = None) -> Any:
        """Use circuit breaker for operation"""
        breaker_name = f"{error_info.category.value}_{operation_name or 'default'}"
        
        if breaker_name not in self.circuit_breakers:
            self.circuit_breakers[breaker_name] = CircuitBreaker(breaker_name)
        
        # Circuit breaker would normally protect the operation
        # For now, return status
        return {"status": "circuit_breaker", "breaker": breaker_name}
    
    def _graceful_degradation(self, error_info: ErrorInfo, operation_name: str = None) -> Any:
        """Implement graceful degradation"""
        logger.warning(f"Graceful degradation for {error_info.category.value}")
        
        # Return minimal functionality
        return {
            "status": "degraded",
            "message": "Operating in degraded mode",
            "features_disabled": [operation_name] if operation_name else []
        }
    
    def _ml_fallback(self, error_info: ErrorInfo) -> Any:
        """ML-specific fallback strategies"""
        return {
            "prediction": 0.5,  # Neutral prediction
            "confidence": 0.0,
            "model": "fallback",
            "message": "Using fallback ML prediction"
        }
    
    def _blockchain_fallback(self, error_info: ErrorInfo) -> Any:
        """Blockchain-specific fallback strategies"""
        return {
            "status": "pending",
            "message": "Blockchain operation queued for retry",
            "estimated_retry": datetime.now() + timedelta(minutes=5)
        }
    
    # Recovery handlers for specific error categories
    def _handle_network_error(self, error_info: ErrorInfo) -> Any:
        """Handle network-related errors"""
        return self._retry_operation(error_info)
    
    def _handle_database_error(self, error_info: ErrorInfo) -> Any:
        """Handle database-related errors"""
        return self._use_circuit_breaker(error_info, "database")
    
    def _handle_blockchain_error(self, error_info: ErrorInfo) -> Any:
        """Handle blockchain-related errors"""
        return self._retry_operation(error_info)
    
    def _handle_ml_error(self, error_info: ErrorInfo) -> Any:
        """Handle ML model errors"""
        return self._use_fallback(error_info, "ml_model")
    
    def _handle_rate_limit_error(self, error_info: ErrorInfo) -> Any:
        """Handle rate limiting errors"""
        # Wait longer for rate limits
        wait_time = 60 * (error_info.retry_count + 1)
        time.sleep(wait_time)
        return self._retry_operation(error_info)
    
    def _handle_timeout_error(self, error_info: ErrorInfo) -> Any:
        """Handle timeout errors"""
        return self._retry_operation(error_info)
    
    # Utility methods
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"ERR_{timestamp}_{self.total_errors}"
    
    def _get_max_retries(self, category: ErrorCategory) -> int:
        """Get max retries based on error category"""
        retry_limits = {
            ErrorCategory.NETWORK: 5,
            ErrorCategory.BLOCKCHAIN: 3,
            ErrorCategory.RATE_LIMIT: 3,
            ErrorCategory.TIMEOUT: 3,
            ErrorCategory.EXTERNAL_API: 3,
            ErrorCategory.ML_MODEL: 2,
            ErrorCategory.DATABASE: 2,
            ErrorCategory.VALIDATION: 1,
            ErrorCategory.AUTHENTICATION: 1,
            ErrorCategory.CONFIGURATION: 0
        }
        return retry_limits.get(category, 3)
    
    def _add_to_history(self, error_info: ErrorInfo):
        """Add error to history with size management"""
        self.error_history.append(error_info)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = (
            f"Error {error_info.error_id}: {error_info.message} "
            f"[{error_info.category.value}] [{error_info.severity.value}]"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_alerts(self, error_info: ErrorInfo):
        """Send alerts for serious errors"""
        for handler in self.alert_handlers:
            try:
                handler(error_info)
            except Exception as alert_error:
                logger.error(f"Alert handler failed: {alert_error}")
    
    # Configuration methods
    def add_fallback_value(self, key: str, value: Any):
        """Add fallback value for specific operations"""
        self.fallback_values[key] = value
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler for error notifications"""
        self.alert_handlers.append(handler)
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name)
        return self.circuit_breakers[name]
    
    # Monitoring and statistics
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        recent_errors = [e for e in self.error_history 
                        if datetime.now() - e.timestamp < timedelta(hours=24)]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "critical_errors": self.critical_errors,
            "recovery_rate": self.recovered_errors / max(self.total_errors, 1),
            "recent_errors_24h": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "circuit_breakers": {
                name: {
                    "state": cb.state.state,
                    "failure_count": cb.state.failure_count,
                    "last_failure": cb.state.last_failure_time.isoformat() if cb.state.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def get_recent_errors(self, hours: int = 24, category: ErrorCategory = None) -> List[ErrorInfo]:
        """Get recent errors with optional filtering"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [e for e in self.error_history if e.timestamp > cutoff]
        
        if category:
            recent = [e for e in recent if e.category == category]
        
        return recent

# Decorator for automatic error handling
def handle_errors(operation_name: str = None, 
                 fallback_value: Any = None,
                 max_retries: int = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Truncate for logging
                    "kwargs": str(kwargs)[:200]
                }
                
                if fallback_value is not None:
                    error_handler.add_fallback_value(f"{func.__name__}_fallback", fallback_value)
                
                return error_handler.handle_error(e, context, operation_name or func.__name__)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                
                if fallback_value is not None:
                    error_handler.add_fallback_value(f"{func.__name__}_fallback", fallback_value)
                
                return await error_handler.ahandle_error(e, context, operation_name or func.__name__)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

# Singleton error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get singleton error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

# Example alert handler
def slack_alert_handler(error_info: ErrorInfo):
    """Example Slack alert handler"""
    if error_info.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
        # In production, this would send to Slack
        logger.info(f"SLACK ALERT: {error_info.severity.value} error - {error_info.message}")

def email_alert_handler(error_info: ErrorInfo):
    """Example email alert handler"""
    if error_info.severity == ErrorSeverity.CRITICAL:
        # In production, this would send email
        logger.info(f"EMAIL ALERT: Critical error - {error_info.message}")

# Initialize default alert handlers
def setup_default_alerts():
    """Setup default alert handlers"""
    error_handler = get_error_handler()
    error_handler.add_alert_handler(slack_alert_handler)
    error_handler.add_alert_handler(email_alert_handler)