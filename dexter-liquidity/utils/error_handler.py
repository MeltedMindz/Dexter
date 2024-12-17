import asyncio
import functools
import logging
import traceback
from typing import Type, Dict, Any, Callable, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DexterError(Exception):
    """Base exception for Dexter system"""
    pass

class NetworkError(DexterError):
    """Network-related errors"""
    pass

class DataError(DexterError):
    """Data validation or processing errors"""
    pass

class ExecutionError(DexterError):
    """Strategy execution errors"""
    pass

class ErrorHandler:
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
    def with_retries(
        self,
        retries: int = 3,
        delay: float = 1.0,
        exponential_backoff: bool = True,
        exceptions: tuple = (Exception,)
    ):
        """Decorator for automatic retry handling"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Reset error count on success
                        self.error_counts[func.__name__] = 0
                        return result
                        
                    except exceptions as e:
                        last_exception = e
                        self._record_error(func.__name__, e)
                        
                        if attempt < retries:
                            wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                            logger.warning(
                                f"Retry {attempt + 1}/{retries} for {func.__name__} "
                                f"after {wait_time}s: {str(e)}"
                            )
                            await asyncio.sleep(wait_time)
                            
                            # Try recovery strategy if available
                            await self._try_recovery(e)
                        else:
                            logger.error(
                                f"All retries failed for {func.__name__}: {str(e)}\n"
                                f"{traceback.format_exc()}"
                            )
                            
                raise last_exception
                
            return wrapper
        return decorator
    
    def with_fallback(self, fallback_func: Callable):
        """Decorator to provide fallback behavior"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Function {func.__name__} failed, using fallback: {str(e)}"
                    )
                    return await fallback_func(*args, **kwargs)
            return wrapper
        return decorator
    
    def register_recovery_strategy(
        self,
        exception_type: Type[Exception],
        strategy: Callable
    ):
        """Register a recovery strategy for an exception type"""
        self.recovery_strategies[exception_type] = strategy
        
    async def _try_recovery(self, error: Exception):
        """Attempt to recover from an error"""
        for exc_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exc_type):
                try:
                    await strategy(error)
                except Exception as e:
                    logger.error(f"Recovery strategy failed: {str(e)}")
                    
    def _record_error(self, function_name: str, error: Exception):
        """Record error occurrence"""
        self.error_counts[function_name] = self.error_counts.get(function_name, 0) + 1
        self.last_errors[function_name] = datetime.now()
        
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'error_counts': self.error_counts.copy(),
            'last_errors': {
                name: timestamp.isoformat()
                for name, timestamp in self.last_errors.items()
            }
        }
