"""
Rate Limiter
Prevents API rate limit errors (429) by tracking request frequency.
"""

import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter with sliding window.
    Tracks calls and waits if approaching limit.
    
    Free tier limits:
    - Groq: 30 requests/min
    - HuggingFace: ~1000/day
    """
    
    def __init__(self, calls_per_minute: int = 30, name: str = "default"):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls allowed per minute
            name: Name for logging purposes
        """
        self.calls_per_minute = calls_per_minute
        self.name = name
        self.calls: list = []
    
    def wait_if_needed(self) -> float:
        """
        Check if we're approaching rate limit and wait if necessary.
        
        Returns:
            Number of seconds waited (0 if no wait needed)
        """
        now = time.time()
        
        # Remove calls older than 60 seconds
        self.calls = [t for t in self.calls if now - t < 60]
        
        waited = 0.0
        
        # If at 80% capacity, add small delay
        if len(self.calls) >= self.calls_per_minute * 0.8:
            wait_time = 1.0
            logger.info(f"RateLimiter[{self.name}]: Approaching limit, waiting {wait_time}s")
            time.sleep(wait_time)
            waited = wait_time
        
        # If at limit, wait until oldest call expires
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 0.5  # +0.5s buffer
            if sleep_time > 0:
                logger.warning(f"RateLimiter[{self.name}]: At limit, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                waited = sleep_time
        
        # Record this call
        self.calls.append(time.time())
        
        return waited
    
    def get_remaining(self) -> int:
        """Get remaining calls available in current window."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        return max(0, self.calls_per_minute - len(self.calls))
    
    def get_usage_percent(self) -> float:
        """Get current usage as percentage."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        return (len(self.calls) / self.calls_per_minute) * 100


# Global rate limiters for different services
_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(name: str, calls_per_minute: int = 30) -> RateLimiter:
    """
    Get or create a named rate limiter (singleton pattern).
    
    Args:
        name: Unique name for the limiter (e.g., "groq", "huggingface")
        calls_per_minute: Rate limit (only used on creation)
        
    Returns:
        RateLimiter instance
    """
    if name not in _limiters:
        _limiters[name] = RateLimiter(calls_per_minute, name)
    return _limiters[name]


# Pre-configured limiters for common services
def get_groq_limiter() -> RateLimiter:
    """Get rate limiter for Groq API (30 req/min)."""
    return get_rate_limiter("groq", 30)


def get_huggingface_limiter() -> RateLimiter:
    """Get rate limiter for HuggingFace API (conservative ~15 req/min)."""
    return get_rate_limiter("huggingface", 15)
