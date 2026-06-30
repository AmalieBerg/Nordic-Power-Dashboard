"""
Data Fetcher - Orchestration Layer for Nordic Power Dashboard

This module orchestrates the data pipeline by:
1. Checking what data exists in database
2. Determining what's missing or needs updating
3. Fetching new/missing data from ENTSO-E API
4. Handling backfills for historical data
5. Managing updates for existing zones

Author: Amalie Berg
Created: December 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .entsoe_client import EntsoeAPIClient
from .database import PriceDatabase
from ..utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Orchestrates data fetching and storage for Nordic power prices.
    
    This class provides high-level methods to:
    - Backfill historical data for all zones
    - Update existing zones with latest data
    - Handle missing data gaps
    - Manage the complete data pipeline
    
    Examples:
        # Initialize
        fetcher = DataFetcher()
        
        # Backfill all Norwegian zones with 2 years of data
        fetcher.backfill_all_zones(years=2)
        
        # Update with latest data
        fetcher.update_all_zones()
        
        # Get pipeline status
        status = fetcher.get_pipeline_status()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize DataFetcher with API client and database.
        
        Args:
            api_key: ENTSO-E API token (uses Config.ENTSOE_API_TOKEN if None)
            db_path: Path to SQLite database (uses Config.DATABASE_PATH if None)
        """
        self.api_key = api_key or Config.ENTSOE_API_TOKEN
        self.client = EntsoeAPIClient(api_key=self.api_key)
        self.db = PriceDatabase(db_path=db_path)
        
        logger.info("DataFetcher initialized")
        logger.info(f"Tracking zones: {Config.TRACKED_ZONES}")
    
    def backfill_zone(
        self,
        zone: str,
        years: int = 2,
        force: bool = False
    ) -> Dict[str, any]:
        """
        Backfill historical data for a single zone.
        
        Args:
            zone: Bidding zone code (e.g., 'NO_2')
            years: Number of years of historical data to fetch
            force: If True, refetch even if data exists
            
        Returns:
            Dictionary with backfill results:
                - zone: Zone code
                - records_added: Number of new records
                - date_range: Tuple of (start_date, end_date)
                - skipped: True if data existed and force=False
                
        Example:
            result = fetcher.backfill_zone('NO_2', years=2)
            print(f"Added {result['records_added']} records")
        """
        logger.info(f"Starting backfill for {zone} ({years} years)")
        
        # Check if data already exists
        if not force:
            latest = self.db.get_latest_timestamp(zone)
            if latest is not None:
                logger.info(f"{zone} already has data up to {latest}, skipping backfill")
                return {
                    'zone': zone,
                    'records_added': 0,
                    'date_range': None,
                    'skipped': True
                }
        
        # Calculate date range
        end = pd.Timestamp.now(tz='Europe/Oslo')
        start = end - pd.Timedelta(days=years * 365)
        
        logger.info(f"Fetching {zone} from {start.date()} to {end.date()}")
        
        try:
            # Fetch data (client handles multi-year requests automatically)
            prices = self.client.fetch_day_ahead_prices(zone, start, end)
            
            if prices is None or len(prices) == 0:
                logger.warning(f"No data returned for {zone}")
                return {
                    'zone': zone,
                    'records_added': 0,
                    'date_range': None,
                    'skipped': False
                }
            
            # Store in database
            records_added = self.db.insert_prices(zone, prices, replace_existing=force)
            
            logger.info(f" Backfill complete for {zone}: {records_added} records")
            
            return {
                'zone': zone,
                'records_added': records_added,
                'date_range': (start, end),
                'skipped': False
            }
            
        except Exception as e:
            logger.error(f" Backfill failed for {zone}: {e}")
            return {
                'zone': zone,
                'records_added': 0,
                'date_range': None,
                'skipped': False,
                'error': str(e)
            }
    
    def backfill_all_zones(
        self,
        zones: Optional[List[str]] = None,
        years: int = 2,
        force: bool = False
    ) -> Dict[str, Dict]:
        """
        Backfill historical data for multiple zones.
        
        Args:
            zones: List of zone codes (uses Config.TRACKED_ZONES if None)
            years: Number of years of historical data
            force: If True, refetch even if data exists
            
        Returns:
            Dictionary mapping zone codes to backfill results
            
        Example:
            results = fetcher.backfill_all_zones(years=2)
            total_records = sum(r['records_added'] for r in results.values())
            print(f"Total records added: {total_records}")
        """
        zones = zones or Config.TRACKED_ZONES
        logger.info(f"Starting backfill for {len(zones)} zones")
        
        results = {}
        for zone in zones:
            result = self.backfill_zone(zone, years=years, force=force)
            results[zone] = result
        
        # Summary
        total_records = sum(r['records_added'] for r in results.values())
        successful = sum(1 for r in results.values() if r['records_added'] > 0)
        
        logger.info("=" * 60)
        logger.info(f"BACKFILL SUMMARY")
        logger.info(f"  Zones processed: {len(zones)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Total records: {total_records}")
        logger.info("=" * 60)
        
        return results
    
    def update_zone(
        self,
        zone: str,
        days: int = 7
    ) -> Dict[str, any]:
        """
        Update a zone with latest data.
        
        Fetches recent data and merges with existing records.
        Useful for daily/periodic updates.
        
        Args:
            zone: Bidding zone code
            days: Number of recent days to fetch (default 7)
            
        Returns:
            Dictionary with update results
            
        Example:
            result = fetcher.update_zone('NO_2', days=7)
        """
        logger.info(f"Updating {zone} with last {days} days")
        
        try:
            # Calculate date range
            end = pd.Timestamp.now(tz='Europe/Oslo')
            start = end - pd.Timedelta(days=days)
            
            # Fetch latest data
            prices = self.client.fetch_day_ahead_prices(zone, start, end)
            
            if prices is None or len(prices) == 0:
                logger.warning(f"No data returned for {zone}")
                return {
                    'zone': zone,
                    'records_added': 0,
                    'date_range': (start, end)
                }
            
            # Store (will update existing records if they exist)
            records_added = self.db.insert_prices(zone, prices, replace_existing=True)
            
            logger.info(f" Updated {zone}: {records_added} records")
            
            return {
                'zone': zone,
                'records_added': records_added,
                'date_range': (start, end)
            }
            
        except Exception as e:
            logger.error(f" Update failed for {zone}: {e}")
            return {
                'zone': zone,
                'records_added': 0,
                'date_range': None,
                'error': str(e)
            }
    
    def update_all_zones(
        self,
        zones: Optional[List[str]] = None,
        days: int = 7
    ) -> Dict[str, Dict]:
        """
        Update all zones with latest data.
        
        Args:
            zones: List of zone codes (uses Config.TRACKED_ZONES if None)
            days: Number of recent days to fetch
            
        Returns:
            Dictionary mapping zone codes to update results
            
        Example:
            # Daily update - run this every day
            results = fetcher.update_all_zones(days=7)
        """
        zones = zones or Config.TRACKED_ZONES
        logger.info(f"Updating {len(zones)} zones with last {days} days")
        
        results = {}
        for zone in zones:
            result = self.update_zone(zone, days=days)
            results[zone] = result
        
        # Summary
        total_records = sum(r['records_added'] for r in results.values())
        successful = sum(1 for r in results.values() if r['records_added'] > 0)
        
        logger.info("=" * 60)
        logger.info(f"UPDATE SUMMARY")
        logger.info(f"  Zones updated: {len(zones)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Total records: {total_records}")
        logger.info("=" * 60)
        
        return results
    
    def fill_gaps(
        self,
        zone: str,
        max_gap_days: int = 7
    ) -> Dict[str, any]:
        """
        Find and fill gaps in historical data for a zone.
        
        Args:
            zone: Bidding zone code
            max_gap_days: Maximum gap size to fill (prevents huge fetches)
            
        Returns:
            Dictionary with gap-filling results
        """
        logger.info(f"Checking for gaps in {zone} data")
        
        try:
            # Get all existing data
            all_prices = self.db.get_prices(zone)
            
            if all_prices is None or len(all_prices) == 0:
                logger.info(f"No existing data for {zone}, run backfill instead")
                return {
                    'zone': zone,
                    'gaps_found': 0,
                    'gaps_filled': 0
                }
            
            # Find gaps (missing hours)
            timestamps = pd.DatetimeIndex(all_prices.index)
            full_range = pd.date_range(
                start=timestamps.min(),
                end=timestamps.max(),
                freq='H',
                tz='Europe/Oslo'
            )
            
            missing = full_range.difference(timestamps)
            
            if len(missing) == 0:
                logger.info(f"No gaps found in {zone} data")
                return {
                    'zone': zone,
                    'gaps_found': 0,
                    'gaps_filled': 0
                }
            
            logger.info(f"Found {len(missing)} missing hours in {zone}")
            
            # Fill gaps (in chunks to avoid huge requests)
            total_filled = 0
            gaps_filled = 0
            
            # Group consecutive missing timestamps
            if len(missing) > 0:
                missing_series = pd.Series(missing)
                missing_series = missing_series.sort_values()
                gaps = []
                current_gap_start = missing_series.iloc[0]
                current_gap_end = missing_series.iloc[0]
                
                for ts in missing_series.iloc[1:]:
                    if (ts - current_gap_end).total_seconds() / 3600 <= 1:
                        current_gap_end = ts
                    else:
                        gaps.append((current_gap_start, current_gap_end))
                        current_gap_start = ts
                        current_gap_end = ts
                
                gaps.append((current_gap_start, current_gap_end))
                
                logger.info(f"Found {len(gaps)} gap periods")
                
                # Fill each gap
                for gap_start, gap_end in gaps:
                    gap_days = (gap_end - gap_start).days + 1
                    
                    if gap_days > max_gap_days:
                        logger.warning(f"Gap too large ({gap_days} days), skipping")
                        continue
                    
                    logger.info(f"Filling gap: {gap_start.date()} to {gap_end.date()}")
                    
                    prices = self.client.fetch_day_ahead_prices(
                        zone,
                        gap_start - pd.Timedelta(hours=1),  # Add buffer
                        gap_end + pd.Timedelta(hours=1)
                    )
                    
                    if prices is not None and len(prices) > 0:
                        records = self.db.insert_prices(zone, prices)
                        total_filled += records
                        gaps_filled += 1
            
            logger.info(f" Filled {gaps_filled} gaps with {total_filled} records")
            
            return {
                'zone': zone,
                'gaps_found': len(gaps) if len(missing) > 0 else 0,
                'gaps_filled': gaps_filled,
                'records_added': total_filled
            }
            
        except Exception as e:
            logger.error(f" Gap filling failed for {zone}: {e}")
            return {
                'zone': zone,
                'gaps_found': 0,
                'gaps_filled': 0,
                'error': str(e)
            }
    
    def get_pipeline_status(self) -> Dict[str, any]:
        """
        Get comprehensive status of the data pipeline.
        
        Returns:
            Dictionary with pipeline status including:
                - database_stats: Overall database statistics
                - zone_status: Per-zone data coverage
                - data_quality: Completeness metrics
                
        Example:
            status = fetcher.get_pipeline_status()
            print(f"Total records: {status['database_stats']['total_records']}")
            
            for zone, info in status['zone_status'].items():
                print(f"{zone}: {info['days_of_data']} days")
        """
        logger.info("Generating pipeline status report")
        
        # Get database stats
        db_stats = self.db.get_database_stats()
        
        # Get per-zone status
        zones = self.db.get_all_zones()
        zone_status = {}
        
        for zone in zones:
            latest = self.db.get_latest_timestamp(zone)
            all_prices = self.db.get_prices(zone)
            
            if all_prices is not None and len(all_prices) > 0:
                first = all_prices.index.min()
                last = all_prices.index.max()
                days = (last - first).days + 1
                hours = len(all_prices)
                expected_hours = days * 24
                completeness = (hours / expected_hours) * 100 if expected_hours > 0 else 0
                
                zone_status[zone] = {
                    'records': len(all_prices),
                    'first_date': first.strftime('%Y-%m-%d'),
                    'latest_date': last.strftime('%Y-%m-%d'),
                    'days_of_data': days,
                    'completeness_pct': round(completeness, 2),
                    'hours_behind': (pd.Timestamp.now(tz='Europe/Oslo') - last).total_seconds() / 3600
                }
        
        return {
            'database_stats': db_stats,
            'zone_status': zone_status,
            'tracked_zones': Config.TRACKED_ZONES,
            'timestamp': pd.Timestamp.now(tz='Europe/Oslo').isoformat()
        }
    
    def print_status(self):
        """
        Print a formatted status report.
        
        Example:
            fetcher.print_status()
        """
        status = self.get_pipeline_status()
        
        print("\n" + "=" * 70)
        print("NORDIC POWER DASHBOARD - DATA PIPELINE STATUS")
        print("=" * 70)
        
        # Database stats
        db_stats = status['database_stats']
        print(f"\n DATABASE STATISTICS:")
        print(f"  Total records: {db_stats['total_records']:,}")
        print(f"  Database size: {db_stats['file_size_mb']:.2f} MB")
        
        # Per-zone status
        print(f"\n  ZONE COVERAGE:")
        zone_status = status['zone_status']
        
        for zone in Config.TRACKED_ZONES:
            if zone in zone_status:
                info = zone_status[zone]
                print(f"\n  {zone}:")
                print(f"    Records: {info['records']:,}")
                print(f"    Date range: {info['first_date']} to {info['latest_date']}")
                print(f"    Coverage: {info['days_of_data']} days")
                print(f"    Completeness: {info['completeness_pct']:.1f}%")
                print(f"    Hours behind: {info['hours_behind']:.1f}")
            else:
                print(f"\n  {zone}:  NO DATA")
        
        print("\n" + "=" * 70 + "\n")


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    """
    Test the DataFetcher with a complete workflow.
    
    Run this script to:
    1. Initialize fetcher
    2. Backfill 30 days for Bergen (NO_2)
    3. Update with latest 7 days
    4. Check for gaps
    5. Display status
    """
    
    print("=" * 70)
    print("DATA FETCHER TEST")
    print("=" * 70)
    
    # Initialize
    print("\n1⃣  Initializing DataFetcher...")
    fetcher = DataFetcher()
    
    # Test with Bergen zone
    test_zone = 'NO_2'
    
    # Backfill 30 days
    print(f"\n2⃣  Backfilling {test_zone} with 30 days of data...")
    result = fetcher.backfill_zone(test_zone, years=30/365)
    
    if result['records_added'] > 0:
        print(f"    Added {result['records_added']} records")
    elif result['skipped']:
        print(f"   ⏭  Data already exists, skipped")
    else:
        print(f"     No data added")
    
    # Update with latest
    print(f"\n3⃣  Updating {test_zone} with latest 7 days...")
    result = fetcher.update_zone(test_zone, days=7)
    print(f"    Updated with {result['records_added']} records")
    
    # Check for gaps
    print(f"\n4⃣  Checking for data gaps in {test_zone}...")
    result = fetcher.fill_gaps(test_zone)
    if result['gaps_found'] > 0:
        print(f"   Found {result['gaps_found']} gaps, filled {result['gaps_filled']}")
    else:
        print(f"    No gaps found")
    
    # Show status
    print(f"\n5⃣  Pipeline Status:")
    fetcher.print_status()
    
    print("\n" + "=" * 70)
    print(" ALL DATA FETCHER TESTS COMPLETED!")
    print("=" * 70)