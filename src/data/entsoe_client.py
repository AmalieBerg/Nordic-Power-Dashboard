"""
Production-ready ENTSO-E API Client
Handles data fetching with retry logic, error handling, and logging
"""

import logging
import time
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError, InvalidPSRTypeError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntsoeAPIClient:
    """
    Production-grade wrapper for ENTSO-E API
    
    Features:
    - Automatic retry logic with exponential backoff
    - Comprehensive error handling
    - Rate limit management
    - Logging for monitoring
    """
    
    # Norwegian bidding zones
    NORWEGIAN_ZONES = {
        'NO1': 'NO_1',  # Oslo
        'NO2': 'NO_2',  # Bergen/Kristiansand
        'NO3': 'NO_3',  # Trondheim
        'NO4': 'NO_4',  # Tromsø
        'NO5': 'NO_5',  # Southwest
    }
    
    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30
    ):
        """
        Initialize ENTSO-E API client
        
        Args:
            api_key: ENTSO-E API token
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Initialize entsoe-py client
        self.client = EntsoePandasClient(api_key=api_key)
        
        logger.info("EntsoeAPIClient initialized")
    
    def fetch_day_ahead_prices(
        self,
        zone: str,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> Optional[pd.Series]:
        """
        Fetch day-ahead prices with retry logic
        
        Args:
            zone: Bidding zone code (e.g., 'NO_2')
            start: Start timestamp (must have timezone)
            end: End timestamp (must have timezone)
            
        Returns:
            pandas Series with prices (EUR/MWh) or None if failed
        """
        # Validate timezone
        if start.tz is None or end.tz is None:
            raise ValueError("start and end must have timezone information")
        
        # Validate zone
        if zone not in self.NORWEGIAN_ZONES.values():
            valid_zones = ', '.join(self.NORWEGIAN_ZONES.values())
            raise ValueError(f"Invalid zone: {zone}. Valid: {valid_zones}")
        
        logger.info(f"Fetching prices for {zone} from {start.date()} to {end.date()}")
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                prices = self.client.query_day_ahead_prices(
                    zone,
                    start=start,
                    end=end
                )
                
                logger.info(f"Successfully fetched {len(prices)} price points for {zone}")
                return prices
                
            except NoMatchingDataError:
                logger.warning(f"No data available for {zone} in specified period")
                return None
                
            except InvalidPSRTypeError as e:
                logger.error(f"Invalid PSR type: {e}")
                return None
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {type(e).__name__}: {e}"
                )
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All retry attempts exhausted for {zone}")
                    return None
        
        return None
    
    def fetch_multiple_zones(
        self,
        zones: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        delay_between_calls: float = 0.2
    ) -> dict[str, Optional[pd.Series]]:
        """
        Fetch prices for multiple zones with rate limiting
        
        Args:
            zones: List of zone codes
            start: Start timestamp
            end: End timestamp
            delay_between_calls: Delay between API calls (seconds)
            
        Returns:
            Dictionary mapping zone -> prices Series
        """
        results = {}
        
        for i, zone in enumerate(zones):
            logger.info(f"Fetching zone {i+1}/{len(zones)}: {zone}")
            
            prices = self.fetch_day_ahead_prices(zone, start, end)
            results[zone] = prices
            
            # Rate limiting: avoid hitting 400 requests/minute
            if i < len(zones) - 1:  # Don't delay after last request
                time.sleep(delay_between_calls)
        
        successful = sum(1 for p in results.values() if p is not None)
        logger.info(f"Fetched {successful}/{len(zones)} zones successfully")
        
        return results
    
    def fetch_latest_prices(
        self,
        zones: List[str],
        days: int = 7
    ) -> dict[str, Optional[pd.Series]]:
        """
        Convenience method to fetch recent prices
        
        Args:
            zones: List of zone codes
            days: Number of days to fetch
            
        Returns:
            Dictionary mapping zone -> prices Series
        """
        end = pd.Timestamp.now(tz='Europe/Oslo').floor('D')
        start = end - pd.Timedelta(days=days)
        
        return self.fetch_multiple_zones(zones, start, end)
    
    def fetch_historical_prices(
        self,
        zone: str,
        years: int = 2
    ) -> Optional[pd.Series]:
        """
        Fetch multi-year historical data
        
        Args:
            zone: Bidding zone code
            years: Number of years to fetch
            
        Returns:
            pandas Series with historical prices
        """
        end = pd.Timestamp.now(tz='Europe/Oslo').floor('D')
        start = end - pd.Timedelta(days=years*365)
        
        logger.info(f"Fetching {years} years of historical data for {zone}")
        
        # entsoe-py automatically handles multi-year requests
        # by splitting into multiple API calls
        return self.fetch_day_ahead_prices(zone, start, end)
    
    @staticmethod
    def get_norwegian_zones() -> List[str]:
        """Get list of all Norwegian bidding zones"""
        return list(EntsoeAPIClient.NORWEGIAN_ZONES.values())


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('ENTSOE_API_TOKEN')
    
    if not api_key:
        print("Error: ENTSOE_API_TOKEN not found in environment")
        exit(1)
    
    # Create client
    client = EntsoeAPIClient(api_key=api_key)
    
    # Test: Fetch last 7 days for Bergen
    print("\n" + "="*60)
    print("TEST: Fetching last 7 days for Bergen (NO_2)")
    print("="*60 + "\n")
    
    end = pd.Timestamp.now(tz='Europe/Oslo')
    start = end - pd.Timedelta(days=7)
    
    prices = client.fetch_day_ahead_prices('NO_2', start, end)
    
    if prices is not None:
        print(f" SUCCESS!")
        print(f"   Data points: {len(prices)}")
        print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"   Price range: {prices.min():.2f} - {prices.max():.2f} EUR/MWh")
        print(f"   Mean: {prices.mean():.2f} EUR/MWh")
        print(f"\n   First 5 hours:")
        print(prices.head())
    else:
        print(" FAILED to fetch data")
    
    # Test: Fetch multiple zones
    print("\n" + "="*60)
    print("TEST: Fetching multiple Norwegian zones")
    print("="*60 + "\n")
    
    zones = ['NO_1', 'NO_2', 'NO_5']
    results = client.fetch_latest_prices(zones, days=1)
    
    for zone, prices in results.items():
        if prices is not None:
            print(f" {zone}: {len(prices)} prices, mean={prices.mean():.2f} EUR/MWh")
        else:
            print(f" {zone}: Failed")
