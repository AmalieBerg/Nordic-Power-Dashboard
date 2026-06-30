"""
GARCH Volatility Forecasting for Nordic-Baltic Power Prices
============================================================

GJR-GARCH(1,1) with Student-t residuals for hourly electricity price volatility.

Why GJR-GARCH with Student-t (not plain GARCH with Normal)?
- Electricity prices show asymmetric volatility response to shocks (leverage effects).
  GJR adds a gamma term that captures this.
- Hourly power returns have very fat tails (extreme spikes, near-zero prices).
  Student-t handles fat tails far better than Normal.
- Standard in the energy econometrics literature (e.g. Erdogdu 2011, Higgs 2009).

Persistence interpretation:
- α + β + γ/2 < 1  stationary, mean-reverting volatility (forecasts vary)
- α + β + γ/2 ≈ 1  IGARCH / unit root in volatility (forecasts are flat)
  This is not a bug. It's the data telling us shocks have permanent effects,
  which is common in markets with structural breaks (Continental/Baltic zones
  post-2022 energy crisis). For these zones, GARCH is the wrong tool;
  regime-switching or longer estimation windows would be appropriate.

Author: Amalie Berg
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
import warnings
from arch import arch_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class GARCHForecaster:
    """
    GJR-GARCH(1,1) volatility forecaster with Student-t residuals.

    Conditional variance:
        σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1}

    where I(·) is the indicator function for negative shocks.
    """

    def __init__(
        self,
        lookback_window: int = 168,
        mean_model: str = 'Constant',
        p: int = 1,
        o: int = 1,
        q: int = 1,
        return_clip_sigma: float = 4.0,
    ):
        """
        Args:
            lookback_window: Hours of historical data for estimation (default 168 = 7 days).
            mean_model: 'Constant' (recommended), 'Zero', or 'AR'.
            p: ARCH lag order.
            o: Asymmetric (GJR) lag order. o=1 enables leverage effects.
            q: GARCH lag order.
            return_clip_sigma: Clip returns at ±N standard deviations before fitting.
                Handles electricity price spikes that destabilise MLE. 4.0 is standard.
        """
        self.lookback_window = lookback_window
        self.mean_model = mean_model
        self.p = p
        self.o = o
        self.q = q
        self.return_clip_sigma = return_clip_sigma

        self.fitted_model: Optional[Any] = None
        self.last_estimation_time: Optional[pd.Timestamp] = None

        model_name = "GJR-GARCH" if o > 0 else "GARCH"
        logger.info(
            f"Initialized {model_name}({p},{o},{q}) forecaster: "
            f"window={lookback_window}h, mean={mean_model}, dist=Student-t"
        )

    def prepare_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute log returns (in %), clip extreme outliers.

        Clipping is essential for electricity: a price drop from €100 to €0.02
        produces a log return of ~-850%, which dominates MLE and forces
        persistence to the unit-root boundary. We clip at ±N·σ before fitting.
        """
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to compute returns")

        # Replace zeros/negatives to avoid log(0)
        safe_prices = prices.replace(0, np.nan).where(prices > 0, np.nan)
        returns = 100 * np.log(safe_prices / safe_prices.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        # Clip extreme outliers (electricity spikes)
        if self.return_clip_sigma is not None and len(returns) > 10:
            std = returns.std()
            mean = returns.mean()
            lower = mean - self.return_clip_sigma * std
            upper = mean + self.return_clip_sigma * std
            n_clipped = ((returns < lower) | (returns > upper)).sum()
            if n_clipped > 0:
                logger.info(
                    f"Clipped {n_clipped} returns beyond ±{self.return_clip_sigma}σ "
                    f"({100*n_clipped/len(returns):.1f}% of observations)"
                )
            returns = returns.clip(lower=lower, upper=upper)

        return returns

    def estimate_garch(self, returns: pd.Series, show_summary: bool = False) -> Any:
        """
        Fit GJR-GARCH(1,1) with Student-t residuals via maximum likelihood.

        No parameter constraints are imposed. If the data implies persistence ≈ 1,
        we report it honestly rather than scaling parameters to force stationarity.
        """
        clean = returns.dropna()
        if len(clean) < self.lookback_window:
            raise ValueError(
                f"Insufficient data: need {self.lookback_window} obs, got {len(clean)}"
            )

        model = arch_model(
            clean,
            mean=self.mean_model,
            vol='Garch',
            p=self.p,
            o=self.o,
            q=self.q,
            dist='t',
            rescale=False,
        )

        fitted = model.fit(disp='off', show_warning=False)
        self.fitted_model = fitted
        self.last_estimation_time = returns.index[-1]

        if show_summary:
            print(fitted.summary())

        diag = self._diagnostics_from_fit(fitted)
        persistence_note = " [IGARCH]" if diag['persistence'] >= 0.995 else ""
        logger.info(
            f"Fit complete: ω={diag['omega']:.4f}, α={diag['alpha']:.4f}, "
            f"γ={diag['gamma']:.4f}, β={diag['beta']:.4f}, "
            f"persistence={diag['persistence']:.4f}{persistence_note}"
        )
        return fitted

    def forecast_volatility(
        self,
        prices: pd.Series,
        horizon: int = 24,
        confidence_level: float = 0.95,
        refit: bool = True,
    ) -> pd.DataFrame:
        """
        Generate h-step-ahead volatility forecasts.

        Returns DataFrame with columns:
            forecast_vol  - point forecast of std deviation (in % return space)
            lower_ci, upper_ci - approximate confidence bands
            volatility_pct - forecast as % of current price level
        """
        returns = self.prepare_returns(prices)
        est_returns = returns.iloc[-self.lookback_window:]

        if refit or self.fitted_model is None:
            fitted = self.estimate_garch(est_returns)
        else:
            fitted = self.fitted_model

        forecasts = fitted.forecast(horizon=horizon, reindex=False)
        variance = forecasts.variance.values[-1, :]
        volatility = np.sqrt(variance)

        z = 1.96 if confidence_level == 0.95 else 2.576
        lower_ci = volatility * (1 - z * 0.1)
        upper_ci = volatility * (1 + z * 0.1)

        last_ts = prices.index[-1]
        forecast_index = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h',
        )

        current_price = prices.iloc[-1]
        result = pd.DataFrame({
            'timestamp': forecast_index,
            'forecast_vol': volatility,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'volatility_pct': (volatility / max(abs(current_price), 1e-6)) * 100,
        }).set_index('timestamp')

        logger.info(
            f"Forecast: horizon={horizon}h, mean={volatility.mean():.4f}, "
            f"range=[{volatility.min():.4f}, {volatility.max():.4f}], "
            f"variation={volatility.max() - volatility.min():.4f}"
        )
        return result

    def rolling_forecast(
        self,
        prices: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        horizon: int = 24,
        step: int = 24,
    ) -> pd.DataFrame:
        """
        Walk-forward out-of-sample forecasts for backtesting.
        At each step, refit on history up to that point and forecast forward.
        """
        logger.info(
            f"Rolling forecast: {start_date} to {end_date}, "
            f"horizon={horizon}h, step={step}h"
        )

        all_forecasts = []
        current = start_date
        n_done = 0

        while current <= end_date:
            try:
                hist = prices[prices.index <= current]
                if len(hist) < self.lookback_window + horizon:
                    current += pd.Timedelta(hours=step)
                    continue

                fc = self.forecast_volatility(hist, horizon=horizon, refit=True)
                fc['forecast_time'] = current
                fc['target_time'] = fc.index
                all_forecasts.append(fc)
                n_done += 1

                if n_done % 10 == 0:
                    logger.info(f"  ...{n_done} forecasts generated")
            except Exception as e:
                logger.warning(f"Forecast failed at {current}: {e}")

            current += pd.Timedelta(hours=step)

        if not all_forecasts:
            raise ValueError("No forecasts generated — check data sufficiency")

        results = pd.concat(all_forecasts, ignore_index=True)

        # Realized volatility for evaluation (24h rolling std of returns)
        returns = self.prepare_returns(prices)

        def realized(row):
            t = row['target_time']
            window = returns[
                (returns.index >= t - pd.Timedelta(hours=23)) &
                (returns.index <= t)
            ]
            return window.std() if len(window) >= 12 else np.nan

        results['actual_vol'] = results.apply(realized, axis=1)
        logger.info(f"Rolling forecast complete: {len(results)} forecasts")
        return results

    def _diagnostics_from_fit(self, fitted) -> Dict[str, float]:
        """Extract GARCH parameters from a fitted model."""
        p = fitted.params
        alpha = float(p.get('alpha[1]', 0.0))
        gamma = float(p.get('gamma[1]', 0.0))  # 0 if o=0
        beta = float(p.get('beta[1]', 0.0))
        # For GJR: persistence = α + β + γ/2 (under symmetric shock distribution)
        persistence = alpha + beta + gamma / 2

        return {
            'omega': float(p.get('omega', np.nan)),
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'persistence': persistence,
            'nu': float(p.get('nu', np.nan)),  # Student-t degrees of freedom
            'aic': float(fitted.aic),
            'bic': float(fitted.bic),
            'log_likelihood': float(fitted.loglikelihood),
            'is_igarch': persistence >= 0.995,
        }

    def get_model_diagnostics(self) -> Dict[str, float]:
        """Get diagnostics for the most recently fitted model."""
        if self.fitted_model is None:
            raise ValueError("No model fitted yet")
        return self._diagnostics_from_fit(self.fitted_model)
