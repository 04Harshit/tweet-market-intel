"""
Proxy manager for rotating IP addresses and handling proxy failures.
"""

import logging
import random
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ProxyManager:
    """Manages proxy rotation and validation"""
    
    def __init__(
        self,
        proxy_list: List[str] = None,
        max_retries: int = 3,
        validation_url: str = "http://httpbin.org/ip",
        validation_timeout: int = 5,
        min_success_rate: float = 0.7
    ):
        self.proxy_list = proxy_list or []
        self.max_retries = max_retries
        self.validation_url = validation_url
        self.validation_timeout = validation_timeout
        self.min_success_rate = min_success_rate
        
        # Proxy statistics
        self.proxy_stats: Dict[str, Dict] = {}
        self.current_proxy_index = 0
        
        # Initialize stats
        for proxy in self.proxy_list:
            self.proxy_stats[proxy] = {
                'success_count': 0,
                'failure_count': 0,
                'total_response_time': 0,
                'last_used': None,
                'last_success': None,
                'last_failure': None,
                'consecutive_failures': 0,
                'is_active': True
            }
        
        # Validate proxies on initialization
        if self.proxy_list:
            self.validate_proxies()
    
    def get_proxy(self, strategy: str = "round_robin") -> Optional[str]:
        """
        Get a proxy based on strategy.
        
        Args:
            strategy: 'round_robin', 'random', 'best_performing', 'least_used'
            
        Returns:
            Proxy string or None if no proxies available
        """
        active_proxies = self.get_active_proxies()
        
        if not active_proxies:
            logger.warning("No active proxies available")
            return None
        
        if strategy == "round_robin":
            proxy = self._get_round_robin(active_proxies)
        elif strategy == "random":
            proxy = self._get_random(active_proxies)
        elif strategy == "best_performing":
            proxy = self._get_best_performing(active_proxies)
        elif strategy == "least_used":
            proxy = self._get_least_used(active_proxies)
        else:
            proxy = self._get_round_robin(active_proxies)
        
        # Update last used time
        if proxy in self.proxy_stats:
            self.proxy_stats[proxy]['last_used'] = datetime.now()
        
        logger.debug(f"Selected proxy: {proxy}")
        return proxy
    
    def _get_round_robin(self, proxies: List[str]) -> str:
        """Round-robin selection"""
        proxy = proxies[self.current_proxy_index % len(proxies)]
        self.current_proxy_index += 1
        return proxy
    
    def _get_random(self, proxies: List[str]) -> str:
        """Random selection"""
        return random.choice(proxies)
    
    def _get_best_performing(self, proxies: List[str]) -> str:
        """Select proxy with highest success rate"""
        best_proxy = None
        best_score = -1
        
        for proxy in proxies:
            stats = self.proxy_stats[proxy]
            total = stats['success_count'] + stats['failure_count']
            
            if total == 0:
                score = 0.5  # Default score for unused proxies
            else:
                success_rate = stats['success_count'] / total
                # Penalize recent failures
                failure_penalty = stats['consecutive_failures'] * 0.1
                score = success_rate - failure_penalty
            
            if score > best_score:
                best_score = score
                best_proxy = proxy
        
        return best_proxy
    
    def _get_least_used(self, proxies: List[str]) -> str:
        """Select least recently used proxy"""
        least_used = None
        oldest_time = datetime.now()
        
        for proxy in proxies:
            last_used = self.proxy_stats[proxy]['last_used']
            if last_used is None or last_used < oldest_time:
                oldest_time = last_used
                least_used = proxy
        
        return least_used or proxies[0]
    
    def get_active_proxies(self) -> List[str]:
        """Get list of currently active proxies"""
        return [
            proxy for proxy in self.proxy_list
            if self.proxy_stats.get(proxy, {}).get('is_active', False)
        ]
    
    def validate_proxies(self, max_workers: int = 10):
        """
        Validate all proxies concurrently.
        
        Args:
            max_workers: Maximum concurrent validation threads
        """
        if not self.proxy_list:
            logger.warning("No proxies to validate")
            return
        
        logger.info(f"Validating {len(self.proxy_list)} proxies...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_proxy = {
                executor.submit(self._validate_single_proxy, proxy): proxy
                for proxy in self.proxy_list
            }
            
            for future in as_completed(future_to_proxy):
                proxy = future_to_proxy[future]
                try:
                    is_valid, response_time = future.result(timeout=self.validation_timeout + 2)
                    self._update_proxy_stats(proxy, is_valid, response_time)
                except Exception as e:
                    logger.error(f"Error validating proxy {proxy}: {e}")
                    self._update_proxy_stats(proxy, False, 0)
        
        active_count = len(self.get_active_proxies())
        logger.info(f"Proxy validation complete: {active_count}/{len(self.proxy_list)} active")
    
    def _validate_single_proxy(self, proxy: str) -> Tuple[bool, float]:
        """
        Validate a single proxy.
        
        Args:
            proxy: Proxy string
            
        Returns:
            Tuple of (is_valid, response_time)
        """
        proxies = {
            'http': proxy,
            'https': proxy
        }
        
        start_time = time.time()
        
        try:
            response = requests.get(
                self.validation_url,
                proxies=proxies,
                timeout=self.validation_timeout,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Verify the response shows the proxy IP
                response_data = response.json()
                if 'origin' in response_data:
                    logger.debug(f"Proxy {proxy} validated in {response_time:.2f}s")
                    return True, response_time
            
            return False, response_time
            
        except requests.RequestException as e:
            response_time = time.time() - start_time
            logger.debug(f"Proxy {proxy} failed: {e}")
            return False, response_time
    
    def _update_proxy_stats(self, proxy: str, is_valid: bool, response_time: float):
        """Update statistics for a proxy"""
        if proxy not in self.proxy_stats:
            self.proxy_stats[proxy] = {
                'success_count': 0,
                'failure_count': 0,
                'total_response_time': 0,
                'last_used': None,
                'last_success': None,
                'last_failure': None,
                'consecutive_failures': 0,
                'is_active': True
            }
        
        stats = self.proxy_stats[proxy]
        
        if is_valid:
            stats['success_count'] += 1
            stats['total_response_time'] += response_time
            stats['last_success'] = datetime.now()
            stats['consecutive_failures'] = 0
            
            # Calculate success rate
            total = stats['success_count'] + stats['failure_count']
            success_rate = stats['success_count'] / total if total > 0 else 0
            
            # Deactivate if success rate is too low
            if success_rate < self.min_success_rate:
                stats['is_active'] = False
                logger.warning(f"Proxy {proxy} deactivated (success rate: {success_rate:.2f})")
            else:
                stats['is_active'] = True
        else:
            stats['failure_count'] += 1
            stats['last_failure'] = datetime.now()
            stats['consecutive_failures'] += 1
            
            # Deactivate after too many consecutive failures
            if stats['consecutive_failures'] >= self.max_retries:
                stats['is_active'] = False
                logger.warning(f"Proxy {proxy} deactivated ({stats['consecutive_failures']} consecutive failures)")
    
    def report_success(self, proxy: str, response_time: float = 0):
        """Report successful use of a proxy"""
        self._update_proxy_stats(proxy, True, response_time)
    
    def report_failure(self, proxy: str):
        """Report failed use of a proxy"""
        self._update_proxy_stats(proxy, False, 0)
    
    def get_proxy_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics for all proxies"""
        return self.proxy_stats.copy()
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics"""
        active_proxies = self.get_active_proxies()
        total_proxies = len(self.proxy_list)
        
        if total_proxies == 0:
            return {
                'total_proxies': 0,
                'active_proxies': 0,
                'active_percentage': 0,
                'avg_success_rate': 0,
                'avg_response_time': 0
            }
        
        # Calculate average success rate and response time
        total_success = 0
        total_failures = 0
        total_response_time = 0
        total_requests = 0
        
        for proxy, stats in self.proxy_stats.items():
            total_success += stats['success_count']
            total_failures += stats['failure_count']
            total_response_time += stats['total_response_time']
            total_requests += stats['success_count'] + stats['failure_count']
        
        avg_success_rate = total_success / total_requests if total_requests > 0 else 0
        avg_response_time = total_response_time / total_success if total_success > 0 else 0
        
        return {
            'total_proxies': total_proxies,
            'active_proxies': len(active_proxies),
            'active_percentage': len(active_proxies) / total_proxies * 100,
            'avg_success_rate': avg_success_rate,
            'avg_response_time': avg_response_time,
            'total_successful_requests': total_success,
            'total_failed_requests': total_failures
        }
    
    def add_proxy(self, proxy: str, validate: bool = True):
        """Add a new proxy to the list"""
        if proxy not in self.proxy_list:
            self.proxy_list.append(proxy)
            self.proxy_stats[proxy] = {
                'success_count': 0,
                'failure_count': 0,
                'total_response_time': 0,
                'last_used': None,
                'last_success': None,
                'last_failure': None,
                'consecutive_failures': 0,
                'is_active': True
            }
            
            if validate:
                is_valid, response_time = self._validate_single_proxy(proxy)
                self._update_proxy_stats(proxy, is_valid, response_time)
            
            logger.info(f"Added proxy: {proxy}")
    
    def remove_proxy(self, proxy: str):
        """Remove a proxy from the list"""
        if proxy in self.proxy_list:
            self.proxy_list.remove(proxy)
            if proxy in self.proxy_stats:
                del self.proxy_stats[proxy]
            logger.info(f"Removed proxy: {proxy}")


def main():
    """Test the proxy manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Proxy Manager Test')
    parser.add_argument('--proxies', type=str, help='Comma-separated list of proxies')
    parser.add_argument('--validate', action='store_true', help='Validate proxies')
    
    args = parser.parse_args()
    
    # Example proxies (replace with real ones)
    if args.proxies:
        proxy_list = [p.strip() for p in args.proxies.split(',')]
    else:
        proxy_list = [
            'http://proxy1.example.com:8080',
            'http://proxy2.example.com:8080',
            'http://proxy3.example.com:8080'
        ]
    
    print(f"Testing Proxy Manager with {len(proxy_list)} proxies")
    
    proxy_manager = ProxyManager(proxy_list)
    
    if args.validate:
        proxy_manager.validate_proxies()
    
    # Test proxy selection strategies
    strategies = ['round_robin', 'random', 'best_performing', 'least_used']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        for i in range(5):
            proxy = proxy_manager.get_proxy(strategy)
            print(f"  {i + 1}. {proxy}")
    
    # Get statistics
    summary = proxy_manager.get_summary_stats()
    print(f"\n📊 Proxy Manager Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()