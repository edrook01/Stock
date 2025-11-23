"""
Retry Handler with Exponential Backoff
Handles retries with exponential backoff and jitter for robust error handling.
"""

import asyncio
import random
import logging
from typing import Callable, Any, Optional, TypeVar, List
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from function call
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Add jitter (random 0-1 second)
                jitter = random.uniform(0, 1)
                total_delay = delay + jitter
                
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}. "
                    f"Retrying in {total_delay:.2f} seconds..."
                )
                
                await asyncio.sleep(total_delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                )
    
    # If we get here, all retries failed
    raise last_exception


def retry_with_backoff_sync(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """
    Retry a synchronous function with exponential backoff.
    
    Args:
        func: Synchronous function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from function call
        
    Raises:
        Last exception if all retries fail
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Add jitter (random 0-1 second)
                jitter = random.uniform(0, 1)
                total_delay = delay + jitter
                
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}. "
                    f"Retrying in {total_delay:.2f} seconds..."
                )
                
                time.sleep(total_delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                )
    
    # If we get here, all retries failed
    raise last_exception


def retry_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Usage:
        @retry_decorator(max_retries=3)
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_with_backoff(
                    func,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    exceptions=exceptions,
                    *args,
                    **kwargs
                )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return retry_with_backoff_sync(
                    func,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    exceptions=exceptions,
                    *args,
                    **kwargs
                )
            return sync_wrapper
    return decorator

