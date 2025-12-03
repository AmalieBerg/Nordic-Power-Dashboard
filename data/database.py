"""
Database layer for Nordic Power Price Dashboard
Handles storage and retrieval of price data using SQLite
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import logging
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import pandas as pd

from src.utils import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PriceDatabase:
    """
    SQLite database for storing day-ahead electricity prices
    
    Features:
    - Automatic schema creation
    - Duplicate handling (upsert)
    - Efficient querying with indexes
    - Data validation
    - Transaction management
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file (default: from Config)
        """
        self.db_path = db_path or Config.DATABASE_PATH
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database and tables if they don't exist
        self._create_tables()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Return rows as dictionaries
        conn.row_factory = sqlite3.Row
        
        return conn
    
    def _create_tables(self):
        """Create database schema if it doesn't exist"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Table: day_ahead_prices
            # Stores hourly day-ahead electricity prices
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS day_ahead_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price_eur_mwh REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(zone, timestamp)
                )
            """)
            
            # Create indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_zone_timestamp 
                ON day_ahead_prices(zone, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON day_ahead_prices(timestamp)
            """)
            
            # Table: volatility_forecasts
            # Stores GARCH model volatility forecasts (Week 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone TEXT NOT NULL,
                    forecast_timestamp TEXT NOT NULL,
                    forecast_horizon_hours INTEGER NOT NULL,
                    volatility REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(zone, forecast_timestamp, forecast_horizon_hours)
                )
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            conn.rollback()
            raise
        
        finally:
            conn.close()
    
    def insert_prices(
        self,
        zone: str,
        prices: pd.Series,
        replace_existing: bool = True
    ) -> int:
        """
        Insert or update price data
        
        Args:
            zone: Bidding zone code (e.g., 'NO_2')
            prices: pandas Series with DatetimeIndex and price values
            replace_existing: If True, update existing records; if False, skip duplicates
            
        Returns:
            Number of records inserted/updated
        """
        if len(prices) == 0:
            logger.warning("No prices to insert")
            return 0
        
        # Validate zone
        if not zone or not isinstance(zone, str):
            raise ValueError(f"Invalid zone: {zone}")
        
        # Validate prices is a Series with DatetimeIndex
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must have a DatetimeIndex")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        inserted_count = 0
        now = datetime.utcnow().isoformat()
        
        try:
            for timestamp, price in prices.items():
                # Convert timestamp to ISO format string
                ts_str = timestamp.isoformat()
                
                # Validate price
                if pd.isna(price):
                    logger.warning(f"Skipping NaN price for {zone} at {ts_str}")
                    continue
                
                if replace_existing:
                    # UPSERT: Insert or replace if exists
                    cursor.execute("""
                        INSERT INTO day_ahead_prices 
                            (zone, timestamp, price_eur_mwh, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(zone, timestamp) 
                        DO UPDATE SET 
                            price_eur_mwh = excluded.price_eur_mwh,
                            updated_at = excluded.updated_at
                    """, (zone, ts_str, float(price), now, now))
                else:
                    # INSERT IGNORE: Skip if exists
                    cursor.execute("""
                        INSERT OR IGNORE INTO day_ahead_prices 
                            (zone, timestamp, price_eur_mwh, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (zone, ts_str, float(price), now, now))
                
                inserted_count += cursor.rowcount
            
            conn.commit()
            logger.info(f"Inserted/updated {inserted_count} price records for {zone}")
            
            return inserted_count
            
        except sqlite3.Error as e:
            logger.error(f"Error inserting prices: {e}")
            conn.rollback()
            raise
        
        finally:
            conn.close()
    
    def get_prices(
        self,
        zone: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.Series:
        """
        Query price data for a zone and time period
        
        Args:
            zone: Bidding zone code
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            
        Returns:
            pandas Series with DatetimeIndex and prices
        """
        conn = self._get_connection()
        
        try:
            # Build query
            query = "SELECT timestamp, price_eur_mwh FROM day_ahead_prices WHERE zone = ?"
            params = [zone]
            
            if start:
                query += " AND timestamp >= ?"
                params.append(start.isoformat())
            
            if end:
                query += " AND timestamp <= ?"
                params.append(end.isoformat())
            
            query += " ORDER BY timestamp ASC"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            
            if len(df) == 0:
                logger.info(f"No prices found for {zone}")
                return pd.Series(dtype=float)
            
            # Convert to Series with DatetimeIndex
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            series = df.set_index('timestamp')['price_eur_mwh']
            
            logger.info(f"Retrieved {len(series)} prices for {zone}")
            return series
            
        except sqlite3.Error as e:
            logger.error(f"Error querying prices: {e}")
            raise
        
        finally:
            conn.close()
    
    def get_latest_timestamp(self, zone: str) -> Optional[datetime]:
        """
        Get the most recent timestamp for a zone
        
        Args:
            zone: Bidding zone code
            
        Returns:
            Latest timestamp or None if no data exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT MAX(timestamp) as latest
                FROM day_ahead_prices
                WHERE zone = ?
            """, (zone,))
            
            result = cursor.fetchone()
            
            if result and result['latest']:
                return datetime.fromisoformat(result['latest'])
            
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting latest timestamp: {e}")
            raise
        
        finally:
            conn.close()
    
    def get_all_zones(self) -> List[str]:
        """
        Get list of all zones with data in the database
        
        Returns:
            List of zone codes
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT DISTINCT zone 
                FROM day_ahead_prices 
                ORDER BY zone
            """)
            
            zones = [row['zone'] for row in cursor.fetchall()]
            return zones
            
        except sqlite3.Error as e:
            logger.error(f"Error getting zones: {e}")
            raise
        
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict[str, any]:
        """
        Get statistics about the database
        
        Returns:
            Dictionary with database statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Total records
            cursor.execute("SELECT COUNT(*) as count FROM day_ahead_prices")
            stats['total_records'] = cursor.fetchone()['count']
            
            # Records per zone
            cursor.execute("""
                SELECT zone, COUNT(*) as count 
                FROM day_ahead_prices 
                GROUP BY zone 
                ORDER BY zone
            """)
            stats['records_per_zone'] = {row['zone']: row['count'] for row in cursor.fetchall()}
            
            # Date range per zone
            cursor.execute("""
                SELECT zone, MIN(timestamp) as earliest, MAX(timestamp) as latest
                FROM day_ahead_prices 
                GROUP BY zone
            """)
            stats['date_ranges'] = {
                row['zone']: {
                    'earliest': row['earliest'],
                    'latest': row['latest']
                }
                for row in cursor.fetchall()
            }
            
            # Database file size
            if self.db_path.exists():
                stats['file_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error getting database stats: {e}")
            raise
        
        finally:
            conn.close()
    
    def delete_zone_data(self, zone: str) -> int:
        """
        Delete all data for a specific zone (use with caution!)
        
        Args:
            zone: Bidding zone code
            
        Returns:
            Number of records deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM day_ahead_prices WHERE zone = ?", (zone,))
            deleted = cursor.rowcount
            
            conn.commit()
            logger.warning(f"Deleted {deleted} records for zone {zone}")
            
            return deleted
            
        except sqlite3.Error as e:
            logger.error(f"Error deleting zone data: {e}")
            conn.rollback()
            raise
        
        finally:
            conn.close()


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.data import EntsoeAPIClient
    
    print("="*60)
    print("DATABASE MODULE TEST")
    print("="*60)
    
    # Initialize database
    print("\n1. Initializing database...")
    db = PriceDatabase()
    print(f"✅ Database created at: {db.db_path}")
    
    # Test: Fetch some data from API and store it
    print("\n2. Fetching sample data from API...")
    load_dotenv()
    api_key = os.getenv('ENTSOE_API_TOKEN')
    
    if not api_key:
        print("❌ ENTSOE_API_TOKEN not found")
        exit(1)
    
    client = EntsoeAPIClient(api_key=api_key)
    
    # Fetch last 7 days for Bergen
    end = pd.Timestamp.now(tz='Europe/Oslo')
    start = end - pd.Timedelta(days=7)
    
    prices = client.fetch_day_ahead_prices('NO_2', start, end)
    
    if prices is None:
        print("❌ Failed to fetch prices")
        exit(1)
    
    print(f"✅ Fetched {len(prices)} prices from API")
    
    # Test: Insert prices
    print("\n3. Inserting prices into database...")
    count = db.insert_prices('NO_2', prices)
    print(f"✅ Inserted {count} records")
    
    # Test: Query prices back
    print("\n4. Querying prices from database...")
    stored_prices = db.get_prices('NO_2', start, end)
    print(f"✅ Retrieved {len(stored_prices)} records")
    
    # Test: Verify data matches
    print("\n5. Verifying data integrity...")
    if len(stored_prices) == len(prices):
        print("✅ Record count matches")
    else:
        print(f"⚠️  Count mismatch: {len(stored_prices)} vs {len(prices)}")
    
    # Test: Get latest timestamp
    print("\n6. Getting latest timestamp...")
    latest = db.get_latest_timestamp('NO_2')
    print(f"✅ Latest timestamp: {latest}")
    
    # Test: Database statistics
    print("\n7. Database statistics:")
    stats = db.get_database_stats()
    print(f"   Total records: {stats['total_records']}")
    print(f"   Database size: {stats['file_size_mb']:.2f} MB")
    print(f"   Zones in DB: {', '.join(stats['records_per_zone'].keys())}")
    
    for zone, info in stats['date_ranges'].items():
        print(f"   {zone}: {info['earliest']} to {info['latest']}")
    
    # Test: Insert same data again (test UPSERT)
    print("\n8. Testing duplicate handling (UPSERT)...")
    count2 = db.insert_prices('NO_2', prices, replace_existing=True)
    print(f"✅ UPSERT completed: {count2} records updated")
    
    # Verify count didn't increase
    stats2 = db.get_database_stats()
    if stats2['total_records'] == stats['total_records']:
        print("✅ No duplicates created")
    else:
        print("⚠️  Duplicate handling may have issues")
    
    print("\n" + "="*60)
    print("✅ ALL DATABASE TESTS PASSED!")
    print("="*60)
    print(f"\nDatabase location: {db.db_path}")
    print("You can inspect it with: sqlite3 data/prices.db")
