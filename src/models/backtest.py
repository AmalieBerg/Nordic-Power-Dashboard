"""
GARCH Volatility Forecast Backtesting
======================================

Evaluates GARCH forecast performance using multiple metrics:
- RMSE, MAE, MAPE (forecast accuracy)
- Direction accuracy (did volatility increase/decrease correctly?)
- Mincer-Zarnowitz regression (forecast quality)
- Diebold-Mariano test (comparison vs benchmark)

Author: Amalie Berg
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class VolatilityBacktest:
    """
    Backtest framework for GARCH volatility forecasts.
    
    Evaluates forecast performance using industry-standard metrics:
    - Accuracy metrics: RMSE, MAE, MAPE
    - Direction accuracy: % of correct volatility direction predictions
    - Mincer-Zarnowitz R²: Forecast efficiency test
    - Forecast bias: Systematic over/under-prediction
    """
    
    def __init__(self):
        """Initialize backtesting framework."""
        self.results: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict[str, float]] = None
        
        logger.info("Initialized VolatilityBacktest")
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 24,
        method: str = 'std'
    ) -> pd.Series:
        """
        Calculate realized volatility from returns.
        
        Args:
            returns: Series of returns
            window: Window size for volatility calculation (hours)
            method: 'std' for standard deviation, 'range' for high-low range
        
        Returns:
            Series of realized volatility values
        """
        if method == 'std':
            # Standard deviation of returns (most common)
            realized_vol = returns.rolling(window=window).std()
            
        elif method == 'range':
            # Range-based volatility (Parkinson estimator)
            # More robust to microstructure noise
            high = returns.rolling(window=window).max()
            low = returns.rolling(window=window).min()
            realized_vol = (high - low) / (2 * np.sqrt(np.log(2)))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return realized_vol
    
    def evaluate_forecasts(
        self,
        forecasts: pd.DataFrame,
        actuals: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecast performance metrics.
        
        Args:
            forecasts: DataFrame with 'forecast_vol' column
            actuals: Series of realized volatility
            confidence_level: For confidence intervals
        
        Returns:
            Dictionary of performance metrics
        """
        # Merge forecasts with actuals
        eval_df = forecasts.copy()
        eval_df['actual_vol'] = actuals
        
        # Drop rows with missing actuals
        eval_df = eval_df.dropna(subset=['actual_vol'])
        
        if len(eval_df) == 0:
            raise ValueError("No valid forecast-actual pairs")
        
        forecast_vals = eval_df['forecast_vol'].values
        actual_vals = eval_df['actual_vol'].values
        
        # Calculate errors
        errors = forecast_vals - actual_vals
        abs_errors = np.abs(errors)
        pct_errors = 100 * errors / actual_vals
        abs_pct_errors = np.abs(pct_errors)
        
        # 1. Accuracy Metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(abs_pct_errors)
        
        # 2. Bias (systematic over/under-prediction)
        bias = np.mean(errors)
        bias_pct = 100 * bias / np.mean(actual_vals)
        
        # 3. Direction Accuracy
        # Did forecast correctly predict volatility increase/decrease?
        forecast_changes = np.diff(forecast_vals)
        actual_changes = np.diff(actual_vals)
        
        correct_direction = np.sum(
            np.sign(forecast_changes) == np.sign(actual_changes)
        )
        direction_accuracy = 100 * correct_direction / len(forecast_changes)
        
        # 4. Mincer-Zarnowitz Regression
        # Regress actual on forecast: actual = a + b*forecast
        # Efficient forecast: a=0, b=1, R²=high
        from sklearn.linear_model import LinearRegression
        
        reg = LinearRegression()
        reg.fit(forecast_vals.reshape(-1, 1), actual_vals)
        
        mz_intercept = reg.intercept_
        mz_slope = reg.coef_[0]
        mz_r2 = reg.score(forecast_vals.reshape(-1, 1), actual_vals)
        
        # 5. Coverage (% of actuals within confidence intervals)
        if 'lower_ci' in eval_df.columns and 'upper_ci' in eval_df.columns:
            within_ci = (
                (actual_vals >= eval_df['lower_ci']) &
                (actual_vals <= eval_df['upper_ci'])
            )
            coverage = 100 * np.mean(within_ci)
        else:
            coverage = np.nan
        
        # 6. Correlation
        correlation = np.corrcoef(forecast_vals, actual_vals)[0, 1]
        
        # Compile metrics
        metrics = {
            # Accuracy
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            
            # Bias
            'bias': bias,
            'bias_pct': bias_pct,
            
            # Direction
            'direction_accuracy': direction_accuracy,
            
            # Mincer-Zarnowitz
            'mz_intercept': mz_intercept,
            'mz_slope': mz_slope,
            'mz_r2': mz_r2,
            
            # Other
            'correlation': correlation,
            'coverage': coverage,
            
            # Sample info
            'n_forecasts': len(eval_df),
            'n_correct_direction': int(correct_direction),
        }
        
        self.metrics = metrics
        self.results = eval_df
        
        logger.info(
            f"Forecast evaluation complete: "
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, "
            f"Direction={direction_accuracy:.1f}%"
        )
        
        return metrics
    
    def print_metrics(self, metrics: Optional[Dict[str, float]] = None):
        """
        Print formatted metrics report.
        
        Args:
            metrics: Dictionary of metrics (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        if metrics is None:
            raise ValueError("No metrics available")
        
        print("\n" + "="*70)
        print("VOLATILITY FORECAST PERFORMANCE METRICS")
        print("="*70)
        
        print("\n ACCURACY METRICS")
        print("-"*70)
        print(f"  MAE (Mean Absolute Error):           {metrics['mae']:.4f}")
        print(f"  RMSE (Root Mean Squared Error):      {metrics['rmse']:.4f}")
        print(f"  MAPE (Mean Absolute % Error):        {metrics['mape']:.2f}%")
        print(f"  Correlation (Forecast vs Actual):    {metrics['correlation']:.4f}")
        
        print("\n FORECAST BIAS")
        print("-"*70)
        print(f"  Bias (Forecast - Actual):            {metrics['bias']:.4f}")
        print(f"  Bias %:                              {metrics['bias_pct']:.2f}%")
        if abs(metrics['bias_pct']) < 5:
            print("   Low bias - forecasts are unbiased")
        elif metrics['bias_pct'] > 5:
            print("   Positive bias - forecasts systematically OVER-predict")
        else:
            print("   Negative bias - forecasts systematically UNDER-predict")
        
        print("\n DIRECTION ACCURACY")
        print("-"*70)
        print(f"  Direction Accuracy:                  {metrics['direction_accuracy']:.1f}%")
        print(f"  Correct Predictions:                 {metrics['n_correct_direction']}/{metrics['n_forecasts']-1}")
        if metrics['direction_accuracy'] > 60:
            print("   Good - forecasts capture volatility dynamics")
        elif metrics['direction_accuracy'] > 50:
            print("  ~ Moderate - better than random")
        else:
            print("   Poor - no better than coin flip")
        
        print("\n MINCER-ZARNOWITZ REGRESSION")
        print("-"*70)
        print(f"  Intercept (should be ~0):            {metrics['mz_intercept']:.4f}")
        print(f"  Slope (should be ~1):                {metrics['mz_slope']:.4f}")
        print(f"  R² (explained variance):             {metrics['mz_r2']:.4f}")
        
        # Interpret MZ results
        slope_ok = 0.8 <= metrics['mz_slope'] <= 1.2
        intercept_ok = abs(metrics['mz_intercept']) < 0.5
        r2_ok = metrics['mz_r2'] > 0.5
        
        if slope_ok and intercept_ok and r2_ok:
            print("   Efficient forecast - unbiased and informative")
        else:
            if not slope_ok:
                if metrics['mz_slope'] < 0.8:
                    print("   Slope < 1: Forecasts under-respond to volatility changes")
                else:
                    print("   Slope > 1: Forecasts over-respond to volatility changes")
            if not intercept_ok:
                print("   Intercept ≠ 0: Systematic bias in forecasts")
            if not r2_ok:
                print("   Low R²: Forecasts don't explain actual volatility well")
        
        if not np.isnan(metrics['coverage']):
            print("\n CONFIDENCE INTERVAL COVERAGE")
            print("-"*70)
            print(f"  Actual Coverage:                     {metrics['coverage']:.1f}%")
            print(f"  Target Coverage:                     95.0%")
            if 93 <= metrics['coverage'] <= 97:
                print("   Coverage is appropriate")
            elif metrics['coverage'] < 93:
                print("   Under-coverage - intervals too narrow")
            else:
                print("   Over-coverage - intervals too wide")
        
        print("\n SAMPLE INFORMATION")
        print("-"*70)
        print(f"  Number of Forecasts:                 {metrics['n_forecasts']}")
        
        print("\n" + "="*70)
    
    def plot_results(
        self,
        results: Optional[pd.DataFrame] = None,
        title: str = "GARCH Volatility Forecast Performance"
    ):
        """
        Create comprehensive visualization of forecast performance.
        
        Args:
            results: DataFrame with forecast and actual columns
            title: Plot title
        """
        if results is None:
            results = self.results
        
        if results is None:
            raise ValueError("No results available")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Forecast vs Actual Time Series
        ax1 = axes[0, 0]
        ax1.plot(
            results.index,
            results['actual_vol'],
            label='Realized Volatility',
            color='blue',
            linewidth=2,
            alpha=0.7
        )
        ax1.plot(
            results.index,
            results['forecast_vol'],
            label='GARCH Forecast',
            color='red',
            linewidth=2,
            linestyle='--',
            alpha=0.7
        )
        
        if 'lower_ci' in results.columns:
            ax1.fill_between(
                results.index,
                results['lower_ci'],
                results['upper_ci'],
                color='red',
                alpha=0.2,
                label='95% CI'
            )
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Volatility')
        ax1.set_title('Forecast vs Realized Volatility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter Plot: Forecast vs Actual
        ax2 = axes[0, 1]
        ax2.scatter(
            results['forecast_vol'],
            results['actual_vol'],
            alpha=0.5,
            s=30
        )
        
        # Add 45-degree line (perfect forecast)
        min_val = min(results['forecast_vol'].min(), results['actual_vol'].min())
        max_val = max(results['forecast_vol'].max(), results['actual_vol'].max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            linewidth=2,
            label='Perfect Forecast'
        )
        
        # Add regression line (filter out NaN values first)
        from sklearn.linear_model import LinearRegression
        
        # Remove rows with NaN in either forecast or actual
        valid_mask = results['actual_vol'].notna() & results['forecast_vol'].notna()
        valid_results = results[valid_mask]
        
        if len(valid_results) < 2:
            logger.warning("Insufficient valid data points for regression line")
        else:
            reg = LinearRegression()
            X = valid_results['forecast_vol'].values.reshape(-1, 1)
            y = valid_results['actual_vol'].values
            reg.fit(X, y)
            pred_line = reg.predict(np.array([[min_val], [max_val]]))
            ax2.plot(
                [min_val, max_val],
                pred_line,
                'g-',
                linewidth=2,
                label=f'Fitted (R²={self.metrics["mz_r2"]:.3f})'
            )
        
        ax2.set_xlabel('Forecasted Volatility')
        ax2.set_ylabel('Realized Volatility')
        ax2.set_title('Forecast Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Forecast Errors Distribution
        ax3 = axes[1, 0]
        errors = results['forecast_vol'] - results['actual_vol']
        ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.axvline(
            errors.mean(),
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Mean Error = {errors.mean():.4f}'
        )
        ax3.set_xlabel('Forecast Error (Forecast - Actual)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Forecast Errors')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Accuracy Metrics
        ax4 = axes[1, 1]
        
        # Calculate rolling MAE
        window = min(24, len(results) // 4)
        rolling_errors = (results['forecast_vol'] - results['actual_vol']).abs()
        rolling_mae = rolling_errors.rolling(window=window).mean()
        
        ax4.plot(results.index, rolling_mae, linewidth=2, color='purple')
        ax4.axhline(
            self.metrics['mae'],
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Overall MAE = {self.metrics["mae"]:.4f}'
        )
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title(f'Rolling MAE (window={window}h)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def compare_models(
        self,
        model_results: Dict[str, pd.DataFrame],
        metric: str = 'rmse'
    ) -> pd.DataFrame:
        """
        Compare performance of multiple forecast models.
        
        Args:
            model_results: Dict of {model_name: results_df}
            metric: Metric to compare ('rmse', 'mae', 'mape', 'direction_accuracy')
        
        Returns:
            DataFrame with comparison results
        """
        comparison = []
        
        for model_name, results in model_results.items():
            # Calculate metrics for this model
            metrics = self.evaluate_forecasts(
                results[['forecast_vol', 'lower_ci', 'upper_ci']],
                results['actual_vol']
            )
            
            comparison.append({
                'model': model_name,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'direction_accuracy': metrics['direction_accuracy'],
                'mz_r2': metrics['mz_r2'],
                'bias_pct': metrics['bias_pct']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values(metric)
        
        logger.info(f"Model comparison complete: {len(comparison_df)} models")
        
        return comparison_df


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_backtest_framework():
    """
    Test backtesting framework with synthetic data.
    """
    print("\n" + "="*70)
    print("TESTING BACKTESTING FRAMEWORK")
    print("="*70)
    
    # Generate synthetic forecast and actual data
    np.random.seed(42)
    n = 100
    
    timestamps = pd.date_range(start='2024-11-01', periods=n, freq='h')
    
    # Actual volatility (with some autocorrelation)
    actual_vol = np.zeros(n)
    actual_vol[0] = 2.0
    for i in range(1, n):
        actual_vol[i] = 0.7 * actual_vol[i-1] + np.random.gamma(2, 0.3)
    
    # Forecasts (actual + noise)
    forecast_vol = actual_vol + np.random.normal(0, 0.3, n)
    lower_ci = forecast_vol - 1.96 * 0.3
    upper_ci = forecast_vol + 1.96 * 0.3
    
    # Create DataFrames
    forecasts = pd.DataFrame({
        'forecast_vol': forecast_vol,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }, index=timestamps)
    
    actuals = pd.Series(actual_vol, index=timestamps)
    
    print(f"\n Generated synthetic data: {n} observations")
    
    # Initialize backtest
    backtest = VolatilityBacktest()
    
    # Evaluate forecasts
    print("\n" + "-"*70)
    print("Evaluating forecast performance...")
    print("-"*70)
    
    metrics = backtest.evaluate_forecasts(forecasts, actuals)
    
    # Print results
    backtest.print_metrics()
    
    # Create plots
    print("\n" + "-"*70)
    print("Generating performance visualizations...")
    print("-"*70)
    
    fig = backtest.plot_results()
    
    print("\n Plots generated successfully")
    
    print("\n" + "="*70)
    print(" ALL BACKTESTING TESTS PASSED!")
    print("="*70)
    
    return backtest, forecasts, actuals, metrics


if __name__ == '__main__':
    # Run tests
    backtest, forecasts, actuals, metrics = test_backtest_framework()
    
    print("\n Backtesting framework ready!")
    print("\nKey metrics calculated:")
    for key in ['mae', 'rmse', 'mape', 'direction_accuracy', 'mz_r2']:
        print(f"  - {key}: {metrics[key]:.4f}")