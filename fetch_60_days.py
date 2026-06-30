#!/usr/bin/env python3
"""
Fetch 60 Days of Data for NO_2 (Bergen)
========================================
Standalone script that fetches historical data directly from ENTSO-E.

Usage:
    python fetch_60_days.py

Author: Amalie Berg
"""

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("="*70)
    print("FETCH 60 DAYS - NO_2 (BERGEN)")
    print("="*70)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_file = PROJECT_ROOT / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    api_key = os.getenv('ENTSOE_API_TOKEN')
    if not api_key:
        print("\n ERROR: ENTSOE_API_TOKEN not found!")
        print("Create a .env file with: ENTSOE_API_TOKEN=your-token")
        return False
    
    print(f"\n API token: {api_key[:4]}...{api_key[-4:]}")
    
    # Import modules
    import pandas as pd
    from entsoe import EntsoePandasClient
    import sqlite3
    
    # Initialize client
    client = EntsoePandasClient(api_key=api_key)
    print(" API client ready")
    
    # Fetch 60 days
    print("\n⏳ Fetching 60 days of NO_2 prices (1-3 minutes)...")
    end = pd.Timestamp.now(tz='Europe/Oslo')
    start = end - pd.Timedelta(days=60)
    
    prices = client.query_day_ahead_prices('NO_2', start=start, end=end)
    print(f" Fetched {len(prices)} price points!")
    
    # Store in database
    db_path = PROJECT_ROOT / 'data' / 'prices.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
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
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_zone_timestamp 
        ON day_ahead_prices(zone, timestamp)
    """)
    
    # Insert data
    now = datetime.now().isoformat()
    records = []
    for ts, price_val in prices.items():
        if pd.notna(price_val):
            records.append((
                'NO_2',
                ts.isoformat(),
                float(price_val),
                now,
                now
            ))
    
    cursor.executemany("""
        INSERT INTO day_ahead_prices (zone, timestamp, price_eur_mwh, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(zone, timestamp) DO UPDATE SET
            price_eur_mwh = excluded.price_eur_mwh,
            updated_at = excluded.updated_at
    """, records)
    
    conn.commit()
    print(f" Stored {len(records)} records")
    
    # Verify
    cursor.execute("""
        SELECT COUNT(*), MIN(timestamp), MAX(timestamp) 
        FROM day_ahead_prices WHERE zone = 'NO_2'
    """)
    count, min_ts, max_ts = cursor.fetchone()
    
    min_dt = pd.to_datetime(min_ts)
    max_dt = pd.to_datetime(max_ts)
    days = (max_dt - min_dt).days + 1
    
    print(f"\n Database now contains:")
    print(f"   Records: {count}")
    print(f"   Days: {days}")
    print(f"   From: {min_ts[:19]}")
    print(f"   To: {max_ts[:19]}")
    
    conn.close()
    
    print("\n" + "="*70)
    print(" SUCCESS! Now run: streamlit run app.py")
    print("="*70)
    return True


if __name__ == '__main__':
    main()
