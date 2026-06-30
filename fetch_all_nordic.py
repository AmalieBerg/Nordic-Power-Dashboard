#!/usr/bin/env python3
"""
Fetch All Nordic-Baltic Zones - 60 Days
========================================
Fetches historical data for all 15 Nordic-Baltic price zones:
- Norway (5 zones)
- Sweden (4 zones)
- Denmark (2 zones)
- Finland (1 zone)
- Estonia (1 zone)
- Latvia (1 zone)
- Lithuania (1 zone)

Usage:
    python fetch_all_nordic.py


Author: Amalie Berg
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# All Nordic-Baltic price zones
ZONES = {
    # Norway
    'NO_1': 'Norway - Oslo (Southeast)',
    'NO_2': 'Norway - Kristiansand (South)',
    'NO_3': 'Norway - Trondheim (Central)',
    'NO_4': 'Norway - Tromsø (North)',
    'NO_5': 'Norway - Bergen (West)',
    # Sweden
    'SE_1': 'Sweden - Luleå (North)',
    'SE_2': 'Sweden - Sundsvall (North-Central)',
    'SE_3': 'Sweden - Stockholm (Central)',
    'SE_4': 'Sweden - Malmö (South)',
    # Denmark
    'DK_1': 'Denmark - West (Jutland)',
    'DK_2': 'Denmark - East (Zealand)',
    # Finland
    'FI': 'Finland',
    # Baltic States
    'EE': 'Estonia',
    'LV': 'Latvia',
    'LT': 'Lithuania',
}


def main():
    print("="*70)
    print("FETCH ALL NORDIC-BALTIC ZONES - 60 DAYS")
    print("="*70)
    print(f"\nZones to fetch: {len(ZONES)}")
    print("   Norway: NO_1, NO_2, NO_3, NO_4, NO_5")
    print("   Sweden: SE_1, SE_2, SE_3, SE_4")
    print("   Denmark: DK_1, DK_2")
    print("   Finland: FI")
    print("   Estonia: EE")
    print("   Latvia: LV")
    print("   Lithuania: LT")
    
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
    
    # Time range
    end = pd.Timestamp.now(tz='Europe/Oslo')
    start = end - pd.Timedelta(days=60)
    print(f"\n Date range: {start.date()} to {end.date()}")
    print(f"⏱  Estimated time: {len(ZONES) * 1.5:.0f}-{len(ZONES) * 2.5:.0f} minutes")
    
    # Database setup
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
    
    print("\n" + "-"*70)
    
    # Fetch each zone
    success_count = 0
    for i, (zone, name) in enumerate(ZONES.items(), 1):
        print(f"\n[{i}/{len(ZONES)}] ⏳ Fetching {zone} ({name})...")
        
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
            
            if prices is None or len(prices) == 0:
                print(f"     No data returned for {zone}")
                continue
            
            print(f"    Got {len(prices)} price points")
            
            # Build records
            now = datetime.now().isoformat()
            records = []
            for ts, price_val in prices.items():
                if pd.notna(price_val):
                    records.append((
                        zone,
                        ts.isoformat(),
                        float(price_val),
                        now,
                        now
                    ))
            
            # Insert
            cursor.executemany("""
                INSERT INTO day_ahead_prices (zone, timestamp, price_eur_mwh, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(zone, timestamp) DO UPDATE SET
                    price_eur_mwh = excluded.price_eur_mwh,
                    updated_at = excluded.updated_at
            """, records)
            
            conn.commit()
            print(f"    Stored {len(records)} records")
            success_count += 1
            
            # Small delay to avoid rate limiting
            time.sleep(3)
            
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    cursor.execute("""
        SELECT zone, COUNT(*), MIN(timestamp), MAX(timestamp)
        FROM day_ahead_prices
        GROUP BY zone
        ORDER BY zone
    """)
    
    print("\n   Zone          | Records | Days | Date Range")
    print("   " + "-"*55)
    
    for zone, count, min_ts, max_ts in cursor.fetchall():
        min_dt = pd.to_datetime(min_ts)
        max_dt = pd.to_datetime(max_ts)
        days = (max_dt - min_dt).days + 1
        print(f"   {zone:13} | {count:7} | {days:4} | {min_ts[:10]} to {max_ts[:10]}")
    
    conn.close()
    
    print("\n" + "="*70)
    print(f" DONE! Successfully fetched {success_count}/{len(ZONES)} zones")
    print("   Run: streamlit run app.py")
    print("="*70)
    return True


if __name__ == '__main__':
    main()
