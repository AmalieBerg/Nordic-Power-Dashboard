"""
Production GARCH Forecasting Pipeline
=====================================

End-to-end pipeline for Nordic power price volatility forecasting:
1. Fetch historical prices from database
2. Estimate GARCH model
3. Generate 24-hour volatility forecasts
4. Backtest historical performance
5. Generate performance reports

Author: Amalie Berg
Date: December 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Import project modules
from src.data.database import PriceDatabase
from src.utils.config import Config

# Import GARCH modules (will be in same directory when moved to src/models/)
try:
    from garch_forecaster import GARCHForecaster
    from backtest import VolatilityBacktest
except ImportError:
    # If running as module
    from src.models.garch_forecaster import GARCHForecaster
    from src.models.backtest import VolatilityBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForecastPipeline:
    """
    Production pipeline for GARCH volatility forecasting.

    Workflow:
    1. Load historical prices from database
    2. Prepare data for GARCH estimation
    3. Generate forecasts
    4. Evaluate performance
    5. Generate reports

    Attributes:
        db: PriceDatabase instance
        forecaster: GARCHForecaster instance
        backtest: VolatilityBacktest instance
        zone: Bidding zone (e.g., 'NO_2' for Bergen)
    """

    def __init__(
        self,
        zone: str = 'NO_2',
        lookback_window: int = 168  # 7 days
    ):
        """
        Initialize forecast pipeline.

        Args:
            zone: Bidding zone for forecasting
            lookback_window: Hours of data for GARCH estimation
        """
        self.zone = zone
        self.lookback_window = lookback_window

        # Initialize components
        config = Config()
        self.db = PriceDatabase(config.DATABASE_PATH)
        self.forecaster = GARCHForecaster(lookback_window=lookback_window)
        self.backtest = VolatilityBacktest()

        logger.info(
            f"Initialized ForecastPipeline for {zone} "
            f"with {lookback_window}h lookback"
        )

    def load_prices(
        self,
        zone: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Load historical prices from database.

        Args:
            zone: Bidding zone (uses self.zone if None)
            start_date: Start of price history (filters after loading)
            end_date: End of price history (filters after loading)

        Returns:
            DataFrame with hourly prices
        """
        zone = zone or self.zone

        logger.info(f"Loading prices for {zone}...")

        # Get all prices for zone (Week 1-2 database doesn't support date filtering)
        prices = self.db.get_prices(zone=zone)

        if prices is None or len(prices) == 0:
            raise ValueError(f"No prices found for {zone}")

        # Filter by date if specified
        if start_date is not None:
            prices = prices[prices.index >= start_date]
        if end_date is not None:
            prices = prices[prices.index <= end_date]

        if len(prices) == 0:
            raise ValueError(
                f"No prices found for {zone} between {start_date} and {end_date}"
            )

        logger.info(
            f"Loaded {len(prices)} prices for {zone} "
            f"from {prices.index[0]} to {prices.index[-1]}"
        )

        return prices

    def run_daily_forecast(
        self,
        zone: Optional[str] = None,
        horizon: int = 24
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Generate daily volatility forecast.

        This is the main production function - run this daily to get
        next 24 hours volatility forecast.

        Args:
            zone: Bidding zone (uses self.zone if None)
            horizon: Forecast horizon in hours

        Returns:
            Tuple of (forecast_df, diagnostics_dict)
        """
        zone = zone or self.zone

        logger.info(f"Running daily forecast for {zone}...")

        # Load latest prices
        prices = self.load_prices(zone)

        # Check we have enough data
        if len(prices) < self.lookback_window + 24:
            raise ValueError(
                f"Insufficient data: need {self.lookback_window + 24}h, "
                f"have {len(prices)}h"
            )

        # Generate forecast
        # Handle both Series and DataFrame from database
        if isinstance(prices, pd.Series):
            price_series = prices
        elif isinstance(prices, pd.DataFrame):
            # Try common column names
            if 'price' in prices.columns:
                price_series = prices['price']
            elif 'Price' in prices.columns:
                price_series = prices['Price']
            elif len(prices.columns) == 1:
                # Single column DataFrame - use it
                price_series = prices.iloc[:, 0]
            else:
                raise ValueError(
                    f"Cannot identify price column. Available columns: {prices.columns.tolist()}"
                )
        else:
            raise TypeError(f"Expected Series or DataFrame, got {type(prices)}")

        forecast = self.forecaster.forecast_volatility(
            price_series,
            horizon=horizon,
            refit=True
        )

        # Get model diagnostics
        diagnostics = self.forecaster.get_model_diagnostics()

        # Add metadata
        forecast['zone'] = zone
        forecast['forecast_time'] = prices.index[-1]

        logger.info(
            f"Daily forecast complete: "
            f"{len(forecast)} hours ahead, "
            f"mean_vol={forecast['forecast_vol'].mean():.4f}"
        )

        return forecast, diagnostics

    def backtest_historical(
        self,
        zone: Optional[str] = None,
        test_days: int = 30,
        step_hours: int = 24
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Run full backtest on historical data.

        This evaluates how the GARCH model would have performed
        over the past N days.

        Args:
            zone: Bidding zone (uses self.zone if None)
            test_days: Number of days to backtest
            step_hours: Hours between forecasts

        Returns:
            Tuple of (results_df, metrics_dict)
        """
        zone = zone or self.zone

        logger.info(
            f"Running backtest for {zone}: "
            f"{test_days} days, step={step_hours}h"
        )

        # Load all available prices
        prices = self.load_prices(zone)

        # Define backtest period
        end_date = prices.index[-1]
        start_date = end_date - pd.Timedelta(days=test_days)

        # Need enough history before start_date for GARCH estimation
        required_start = start_date - pd.Timedelta(hours=self.lookback_window)

        if prices.index[0] > required_start:
            logger.warning(
                f"Insufficient history for full backtest. "
                f"Adjusting start date from {start_date} to "
                f"{prices.index[0] + pd.Timedelta(hours=self.lookback_window)}"
            )
            start_date = prices.index[0] + pd.Timedelta(hours=self.lookback_window)

        # Run rolling forecast
        # Handle both Series and DataFrame from database
        if isinstance(prices, pd.Series):
            price_series = prices
        elif isinstance(prices, pd.DataFrame):
            # Try common column names
            if 'price' in prices.columns:
                price_series = prices['price']
            elif 'Price' in prices.columns:
                price_series = prices['Price']
            elif len(prices.columns) == 1:
                price_series = prices.iloc[:, 0]
            else:
                raise ValueError(
                    f"Cannot identify price column. Available columns: {prices.columns.tolist()}"
                )
        else:
            raise TypeError(f"Expected Series or DataFrame, got {type(prices)}")

        rolling_results = self.forecaster.rolling_forecast(
            price_series,
            start_date=start_date,
            end_date=end_date,
            horizon=24,
            step=step_hours
        )

        # Evaluate performance
        forecast_df = rolling_results.set_index('target_time')
        metrics = self.backtest.evaluate_forecasts(
            forecast_df[['forecast_vol', 'lower_ci', 'upper_ci']],
            forecast_df['actual_vol']
        )

        logger.info(
            f"Backtest complete: {len(rolling_results)} forecasts, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"Direction={metrics['direction_accuracy']:.1f}%"
        )

        return rolling_results, metrics

    def generate_report(
        self,
        backtest_results: pd.DataFrame,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Generate comprehensive performance report.

        Args:
            backtest_results: DataFrame from backtest_historical()
            metrics: Dictionary from evaluate_forecasts()
            save_path: Path to save report (optional)
        """
        logger.info("Generating performance report...")

        # Print metrics
        self.backtest.print_metrics(metrics)

        # Create visualizations
        results_df = backtest_results.set_index('target_time')
        fig = self.backtest.plot_results(
            results_df,
            title=f"GARCH Volatility Forecast - {self.zone}"
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Report saved to {save_path}")

        return fig

    def get_latest_forecast(
        self,
        zone: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get latest forecast in JSON-serializable format.

        Useful for API endpoints or dashboard integration.

        Args:
            zone: Bidding zone (uses self.zone if None)

        Returns:
            Dictionary with forecast data
        """
        zone = zone or self.zone

        forecast, diagnostics = self.run_daily_forecast(zone)

        # Convert to JSON-serializable format
        output = {
            'zone': zone,
            'forecast_time': forecast['forecast_time'].iloc[0].isoformat(),
            'horizon_hours': len(forecast),
            'mean_volatility': float(forecast['forecast_vol'].mean()),
            'max_volatility': float(forecast['forecast_vol'].max()),
            'min_volatility': float(forecast['forecast_vol'].min()),
            'forecasts': [
                {
                    'timestamp': ts.isoformat(),
                    'volatility': float(vol),
                    'volatility_pct': float(pct),
                    'lower_ci': float(low),
                    'upper_ci': float(high)
                }
                for ts, vol, pct, low, high in zip(
                    forecast.index,
                    forecast['forecast_vol'],
                    forecast['volatility_pct'],
                    forecast['lower_ci'],
                    forecast['upper_ci']
                )
            ],
            'model_diagnostics': {
                'omega': float(diagnostics['omega']),
                'alpha': float(diagnostics['alpha']),
                'gamma': float(diagnostics.get('gamma', 0.0)),
                'beta': float(diagnostics['beta']),
                'persistence': float(diagnostics['persistence']),
                'is_igarch': bool(diagnostics.get('is_igarch', diagnostics['persistence'] >= 0.995))
            }
        }

        return output


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Run complete GARCH forecasting pipeline.
    """
    print("\n" + "="*70)
    print("GARCH VOLATILITY FORECASTING PIPELINE")
    print("Nordic Power Prices - Production System")
    print("="*70)

    # Configuration
    ZONE = 'NO_2'  # Bergen
    BACKTEST_DAYS = 7  # Test on last 7 days

    try:
        # Initialize pipeline
        print(f"\nInitializing pipeline for {ZONE}...")
        pipeline = ForecastPipeline(zone=ZONE, lookback_window=168)

        # Step 1: Generate today's forecast
        print("\n" + "-"*70)
        print("STEP 1: Daily Volatility Forecast")
        print("-"*70)

        forecast, diagnostics = pipeline.run_daily_forecast()

        print(f"\n24-hour forecast generated")
        print(f"\nGJR-GARCH(1,1) Parameters:")
        print(f"  omega:        {diagnostics['omega']:.6f}")
        print(f"  alpha:        {diagnostics['alpha']:.6f}")
        print(f"  gamma:        {diagnostics.get('gamma', 0.0):.6f}")
        print(f"  beta:         {diagnostics['beta']:.6f}")
        print(f"  Persistence:  {diagnostics['persistence']:.6f}")
        if diagnostics.get('is_igarch', diagnostics['persistence'] >= 0.995):
            print(f"  Note:         IGARCH detected (persistence ~ 1)")

        print(f"\nForecast Summary:")
        print(f"  Mean volatility:  {forecast['forecast_vol'].mean():.4f}")
        print(f"  Min volatility:   {forecast['forecast_vol'].min():.4f}")
        print(f"  Max volatility:   {forecast['forecast_vol'].max():.4f}")
        print(f"  As % of price:    {forecast['volatility_pct'].mean():.2f}%")

        # Step 2: Run backtest
        print("\n" + "-"*70)
        print(f"STEP 2: Historical Backtest ({BACKTEST_DAYS} days)")
        print("-"*70)

        results, metrics = pipeline.backtest_historical(
            test_days=BACKTEST_DAYS,
            step_hours=24
        )

        print(f"\nBacktest complete: {len(results)} forecasts")

        # Step 3: Generate report
        print("\n" + "-"*70)
        print("STEP 3: Performance Report")
        print("-"*70)

        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save plot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = results_dir / f"garch_forecast_{ZONE}_{timestamp}.png"

        pipeline.generate_report(results, metrics, save_path=str(plot_path))

        print(f"\nPlot saved to: {plot_path}")

        # Summary
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*70)

        print(f"\nKey Performance Metrics:")
        print(f"  RMSE:                {metrics['rmse']:.4f}")
        print(f"  MAE:                 {metrics['mae']:.4f}")
        print(f"  Direction Accuracy:  {metrics['direction_accuracy']:.1f}%")
        print(f"  R^2 (MZ regression): {metrics['mz_r2']:.4f}")

        # Save metrics to text file
        metrics_path = results_dir / f"metrics_{ZONE}_{timestamp}.txt"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("GARCH VOLATILITY FORECAST - PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Zone: {ZONE}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Forecast Horizon: 24 hours\n\n")
            f.write("KEY METRICS:\n")
            f.write(f"  RMSE:                {metrics['rmse']:.4f}\n")
            f.write(f"  MAE:                 {metrics['mae']:.4f}\n")
            f.write(f"  Direction Accuracy:  {metrics['direction_accuracy']:.1f}%\n")
            f.write(f"  R^2 (MZ regression): {metrics['mz_r2']:.4f}\n")
            f.write(f"  MAPE:                {metrics['mape']:.2f}%\n")
            f.write(f"  Bias:                {metrics['bias']:.4f}\n")
            f.write(f"  Coverage:            {metrics['coverage']:.1f}%\n\n")
            f.write("GJR-GARCH PARAMETERS:\n")
            f.write(f"  omega:        {diagnostics['omega']:.6f}\n")
            f.write(f"  alpha:        {diagnostics['alpha']:.6f}\n")
            f.write(f"  gamma:        {diagnostics.get('gamma', 0.0):.6f}\n")
            f.write(f"  beta:         {diagnostics['beta']:.6f}\n")
            f.write(f"  Persistence:  {diagnostics['persistence']:.6f}\n")
            if diagnostics.get('is_igarch', diagnostics['persistence'] >= 0.995):
                f.write(f"  Note:         IGARCH detected (persistence ~ 1)\n")

        print(f"Metrics saved to: {metrics_path}")

        if metrics['direction_accuracy'] > 60:
            print(f"\nEXCELLENT: Forecasts capture volatility dynamics well")
        elif metrics['direction_accuracy'] > 50:
            print(f"\nGOOD: Forecasts better than random")
        else:
            print(f"\nNEEDS IMPROVEMENT: Consider model refinements")

        return pipeline, forecast, results, metrics

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    pipeline, forecast, results, metrics = main()

    print("\nGARCH forecasting system ready for production.")
    print("\nUsage examples:")
    print("  # Get latest forecast:")
    print("  forecast_data = pipeline.get_latest_forecast('NO_2')")
    print("\n  # Run backtest:")
    print("  results, metrics = pipeline.backtest_historical(test_days=30)")
    print("\n  # Generate report:")
    print("  pipeline.generate_report(results, metrics, 'report.png')")
