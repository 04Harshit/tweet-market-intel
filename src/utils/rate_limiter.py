"""
Rate limiting utility for API requests and web scraping.
"""

import time
import random
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from collections import deque
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter with exponential backoff and jitter"""
    
    def __init__(
        self,
        max_calls_per_minute: int = 30,
        max_calls_per_hour: int = 300,
        retry_attempts: int = 3,
        backoff_factor: float = 2.0,
        jitter_min: float = 0.5,
        jitter_max: float = 1.5
    ):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_hour = max_calls_per_hour
        self.retry_attempts = retry_attempts
        self.backoff_factor = backoff_factor
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        
        # Tracking calls
        self.minute_calls = deque()
        self.hour_calls = deque()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_calls = 0
        self.total_wait_time = 0
        self.retry_count = 0
    
    def wait_if_needed(self):
        """
        Wait if rate limits would be exceeded.
        Should be called before each request.
        """
        with self.lock:
            now = time.time()
            
            # Clean old calls
            self._clean_old_calls(now)
            
            # Check limits
            wait_time = self._calculate_wait_time(now)
            
            if wait_time > 0:
                logger.debug(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.total_wait_time += wait_time
            
            # Record this call
            self.minute_calls.append(now)
            self.hour_calls.append(now)
            self.total_calls += 1
    
    def _clean_old_calls(self, now: float):
        """Remove calls older than the tracking windows"""
        # Clean minute window (60 seconds)
        while self.minute_calls and now - self.minute_calls[0] > 60:
            self.minute_calls.popleft()
        
        # Clean hour window (3600 seconds)
        while self.hour_calls and now - self.hour_calls[0] > 3600:
            self.hour_calls.popleft()
    
    def _calculate_wait_time(self, now: float) -> float:
        """Calculate how long to wait before next request"""
        wait_time = 0
        
        # Check minute limit
        if len(self.minute_calls) >= self.max_calls_per_minute:
            oldest_call = self.minute_calls[0]
            time_since_oldest = now - oldest_call
            if time_since_oldest < 60:
                wait_time = max(wait_time, 60 - time_since_oldest)
        
        # Check hour limit
        if len(self.hour_calls) >= self.max_calls_per_hour:
            oldest_call = self.hour_calls[0]
            time_since_oldest = now - oldest_call
            if time_since_oldest < 3600:
                wait_time = max(wait_time, 3600 - time_since_oldest)
        
        # Add jitter
        if wait_time > 0:
            jitter = random.uniform(self.jitter_min, self.jitter_max)
            wait_time *= jitter
        
        return wait_time
    
    def execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic and rate limiting.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Wait if needed
                self.wait_if_needed()
                
                # Execute function
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.retry_count += 1
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                    
                    # Calculate backoff time
                    backoff_time = self.backoff_factor ** attempt
                    backoff_time *= random.uniform(self.jitter_min, self.jitter_max)
                    
                    logger.info(f"Backing off for {backoff_time:.2f} seconds")
                    time.sleep(backoff_time)
                
                else:
                    # Non-rate-limit error, retry immediately or raise
                    logger.error(f"Error on attempt {attempt + 1}: {e}")
                    if attempt == self.retry_attempts - 1:
                        raise
        
        # All retries failed
        raise last_exception or Exception("All retry attempts failed")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if exception is a rate limit error"""
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit',
            'too many requests',
            '429',
            'quota',
            'limit exceeded',
            'throttled'
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                'total_calls': self.total_calls,
                'current_minute_calls': len(self.minute_calls),
                'current_hour_calls': len(self.hour_calls),
                'total_wait_time': self.total_wait_time,
                'avg_wait_per_call': self.total_wait_time / self.total_calls if self.total_calls > 0 else 0,
                'retry_count': self.retry_count,
                'max_calls_per_minute': self.max_calls_per_minute,
                'max_calls_per_hour': self.max_calls_per_hour
            }
    
    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.total_calls = 0
            self.total_wait_time = 0
            self.retry_count = 0


class DistributedRateLimiter(RateLimiter):
    """
    Rate limiter for distributed systems.
    Uses shared storage (like Redis) for coordination.
    """
    
    def __init__(self, *args, redis_client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis = redis_client
        self.client_id = f"client_{random.randint(1000, 9999)}"
    
    def wait_if_needed(self):
        """Distributed version using Redis for coordination"""
        if not self.redis:
            # Fallback to local rate limiting
            return super().wait_if_needed()
        
        # Implementation using Redis sorted sets for distributed rate limiting
        # This is a simplified version
        pass


# Decorator version
def rate_limited(limiter: RateLimiter):
    """
    Decorator to add rate limiting to functions.
    
    Args:
        limiter: RateLimiter instance
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return limiter.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


def main():
    """Test the rate limiter"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rate Limiter Test')
    parser.add_argument('--calls', type=int, default=10, help='Number of calls to make')
    parser.add_argument('--interval', type=float, default=0.1, help='Interval between calls')
    
    args = parser.parse_args()
    
    # Create rate limiter with tight limits for testing
    limiter = RateLimiter(
        max_calls_per_minute=5,
        max_calls_per_hour=10,
        retry_attempts=2
    )
    
    def mock_request(i):
        print(f"Making request {i + 1}")
        return f"Response {i + 1}"
    
    print("Testing rate limiter...")
    print(f"Making {args.calls} calls with {args.interval}s interval")
    
    results = []
    for i in range(args.calls):
        try:
            result = limiter.execute_with_retry(mock_request, i)
            results.append(result)
            time.sleep(args.interval)
        except Exception as e:
            print(f"Request {i + 1} failed: {e}")
    
    stats = limiter.get_stats()
    print(f"\n📊 Rate Limiter Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Completed {len(results)} successful requests")


if __name__ == "__main__":
    main()