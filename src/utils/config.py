"""
Configuration management for Nordic Power Dashboard
Handles environment variables and application settings
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Database
    DATABASE_PATH = DATA_DIR / "prices.db"
    
    # ENTSO-E API
    ENTSOE_API_TOKEN = os.getenv('ENTSOE_API_TOKEN')
    ENTSOE_RATE_LIMIT_DELAY = 0.2  # seconds between requests
    ENTSOE_MAX_RETRIES = 3
    ENTSOE_RETRY_DELAY = 1.0  # initial retry delay in seconds
    
    # Data fetching
    HISTORICAL_YEARS = 2  # How many years of historical data to store
    UPDATE_FREQUENCY_HOURS = 6  # How often to fetch new data
    
    # Norwegian bidding zones to track
    TRACKED_ZONES = ['NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5']
    
    # Forecasting
    GARCH_FORECAST_HORIZON = 24  # hours ahead to forecast
    GARCH_P = 1  # GARCH(p,q) - p parameter
    GARCH_Q = 1  # GARCH(p,q) - q parameter
    
    # Dashboard
    STREAMLIT_PORT = 8501
    DASHBOARD_TITLE = "Nordic Power Price Dashboard"
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check API token
        if not cls.ENTSOE_API_TOKEN:
            errors.append("ENTSOE_API_TOKEN not set in environment")
        
        # Create directories if they don't exist
        for directory in [cls.DATA_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration (excluding secrets)"""
        print("Configuration:")
        print(f"  Project Root: {cls.PROJECT_ROOT}")
        print(f"  Data Dir: {cls.DATA_DIR}")
        print(f"  Database: {cls.DATABASE_PATH}")
        print(f"  API Token: {' Set' if cls.ENTSOE_API_TOKEN else ' Not set'}")
        print(f"  Tracked Zones: {', '.join(cls.TRACKED_ZONES)}")
        print(f"  Historical Years: {cls.HISTORICAL_YEARS}")
        print(f"  Update Frequency: {cls.UPDATE_FREQUENCY_HOURS} hours")


if __name__ == "__main__":
    # Validate configuration
    is_valid, errors = Config.validate()
    
    if is_valid:
        print(" Configuration is valid\n")
        Config.print_config()
    else:
        print(" Configuration errors:")
        for error in errors:
            print(f"   - {error}")
