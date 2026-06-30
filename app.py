#!/usr/bin/env python3
"""
Nordic Power Price Dashboard
=============================
Real-time electricity price analysis with GARCH volatility forecasting.

Features:
- Auto-updates prices on startup (cached 1 hour)
- 15 Nordic-Baltic price zones
- GJR-GARCH(1,1) with Student-t residuals
- Walk-forward backtesting

Usage:
    streamlit run app.py
    # Or: python -m streamlit run app.py

Author: Amalie Berg
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import os

# Load environment for API key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Page config
st.set_page_config(
    page_title="Nordic-Baltic Power Dashboard",
    layout="wide"
)

# Constants
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = PROJECT_ROOT / 'data' / 'prices.db'

ZONES = {
    # Norway
    'NO_1': 'Oslo (Southeast)',
    'NO_2': 'Kristiansand (South)',
    'NO_3': 'Trondheim (Central)',
    'NO_4': 'Tromsø (North)',
    'NO_5': 'Bergen (West)',
    # Sweden
    'SE_1': 'Luleå (North)',
    'SE_2': 'Sundsvall (North-Central)',
    'SE_3': 'Stockholm (Central)',
    'SE_4': 'Malmö (South)',
    # Denmark
    'DK_1': 'West Denmark (Jutland)',
    'DK_2': 'East Denmark (Zealand)',
    # Finland
    'FI': 'Finland',
    # Baltic States
    'EE': 'Estonia',
    'LV': 'Latvia',
    'LT': 'Lithuania',
}

# IGARCH detection threshold (consistent with src/models/garch_forecaster.py)
IGARCH_THRESHOLD = 0.995


# ============================================================================
# AUTO-UPDATE ON STARTUP
# ============================================================================

def initialize_database():
    """Create database and fetch initial data if empty. NO CACHING - must run each time."""

    # Step 1: Check API key
    api_key = os.getenv('ENTSOE_API_TOKEN')
    if not api_key:
        return {"status": "error", "reason": "No API key found. Add ENTSOE_API_TOKEN to Streamlit secrets."}

    # Step 2: Check entsoe-py
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return {"status": "error", "reason": "entsoe-py not installed. Add 'entsoe-py' to requirements.txt"}

    # Step 3: Create data directory
    try:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"status": "error", "reason": f"Cannot create data directory: {e}"}

    # Step 4: Create/connect database
    try:
        conn = sqlite3.connect(DATABASE_PATH)
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
        conn.commit()
    except Exception as e:
        return {"status": "error", "reason": f"Database error: {e}"}

    # Step 5: Check if we have data
    cursor.execute("SELECT COUNT(*) FROM day_ahead_prices")
    count = cursor.fetchone()[0]

    if count > 0:
        conn.close()
        return {"status": "exists", "records": count}

    # Step 6: Fetch initial data for ALL 15 Nordic-Baltic zones
    errors = []
    try:
        client = EntsoePandasClient(api_key=api_key)
        end = pd.Timestamp.now(tz='Europe/Oslo')
        start = end - pd.Timedelta(days=60)  # 60 days for backtesting

        # All 15 zones
        initial_zones = [
            'NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5',  # Norway
            'SE_1', 'SE_2', 'SE_3', 'SE_4',          # Sweden
            'DK_1', 'DK_2',                          # Denmark
            'FI',                                    # Finland
            'EE', 'LV', 'LT'                         # Baltic
        ]
        fetched = []

        for zone in initial_zones:
            try:
                prices = client.query_day_ahead_prices(zone, start=start, end=end)
                if prices is not None and len(prices) > 0:
                    now = datetime.now().isoformat()
                    records = [
                        (zone, ts.isoformat(), float(price), now, now)
                        for ts, price in prices.items()
                        if pd.notna(price)
                    ]
                    cursor.executemany("""
                        INSERT OR REPLACE INTO day_ahead_prices 
                        (zone, timestamp, price_eur_mwh, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, records)
                    conn.commit()
                    fetched.append(zone)
                import time
                time.sleep(1)  # Delay between API calls to avoid rate limiting
            except Exception as e:
                errors.append(f"{zone}: {str(e)[:50]}")

        conn.close()

        if fetched:
            return {"status": "initialized", "zones": fetched, "errors": errors}
        else:
            return {"status": "error", "reason": f"Could not fetch any zones. Errors: {errors}"}

    except Exception as e:
        conn.close()
        return {"status": "error", "reason": f"API error: {e}"}


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def auto_update_prices():
    """Fetch latest 7 days for all zones with existing data."""
    api_key = os.getenv('ENTSOE_API_TOKEN')
    if not api_key:
        return {"status": "skipped", "reason": "No API key"}

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return {"status": "skipped", "reason": "entsoe-py not installed"}

    if not DATABASE_PATH.exists():
        return {"status": "skipped", "reason": "No database"}

    # Get zones that have data
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT zone FROM day_ahead_prices")
    existing_zones = [row[0] for row in cursor.fetchall()]

    if not existing_zones:
        conn.close()
        return {"status": "skipped", "reason": "No existing data"}

    # Fetch latest 7 days for each zone
    client = EntsoePandasClient(api_key=api_key)
    end = pd.Timestamp.now(tz='Europe/Oslo')
    start = end - pd.Timedelta(days=7)

    updated = []
    errors = []

    for zone in existing_zones:
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)

            if prices is not None and len(prices) > 0:
                now = datetime.now().isoformat()
                records = [
                    (zone, ts.isoformat(), float(price), now, now)
                    for ts, price in prices.items()
                    if pd.notna(price)
                ]

                cursor.executemany("""
                    INSERT INTO day_ahead_prices (zone, timestamp, price_eur_mwh, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(zone, timestamp) DO UPDATE SET
                        price_eur_mwh = excluded.price_eur_mwh,
                        updated_at = excluded.updated_at
                """, records)

                updated.append(zone)

        except Exception as e:
            errors.append(f"{zone}: {str(e)[:50]}")

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "updated": updated,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get database connection."""
    if not DATABASE_PATH.exists():
        return None
    return sqlite3.connect(DATABASE_PATH)


def get_prices(zone: str, days: int = None) -> pd.Series:
    """Get prices from database."""
    conn = get_db_connection()
    if conn is None:
        return pd.Series(dtype=float)

    try:
        # Detect price column name
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(day_ahead_prices)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'price_eur_mwh' in columns:
            price_col = 'price_eur_mwh'
        elif 'price' in columns:
            price_col = 'price'
        else:
            st.error(f"No price column found. Columns: {columns}")
            return pd.Series(dtype=float)

        if days:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            query = f"""
                SELECT timestamp, {price_col} as price FROM day_ahead_prices 
                WHERE zone = ? AND timestamp >= ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=(zone, start_date))
        else:
            query = f"""
                SELECT timestamp, {price_col} as price FROM day_ahead_prices 
                WHERE zone = ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=(zone,))

        conn.close()

        if df.empty:
            return pd.Series(dtype=float)

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        prices = df.set_index('timestamp')['price']
        return prices.sort_index()

    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.Series(dtype=float)


def get_database_stats():
    """Get database statistics."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT zone, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM day_ahead_prices
            GROUP BY zone
        """)

        stats = {}
        for zone, count, min_ts, max_ts in cursor.fetchall():
            min_dt = pd.to_datetime(min_ts, utc=True)
            max_dt = pd.to_datetime(max_ts, utc=True)
            days = (max_dt - min_dt).days + 1
            stats[zone] = {
                'count': count,
                'min_date': min_ts,
                'max_date': max_ts,
                'days': days
            }

        conn.close()
        return stats

    except Exception:
        return None


# ============================================================================
# GARCH MODEL (GJR-GARCH with Student-t)
# ============================================================================

def fit_garch_simple(returns: pd.Series):
    """
    GJR-GARCH(1,1) with Student-t residuals for electricity price volatility.

    Why this specification:
    - GJR adds an asymmetric (gamma) term that captures leverage effects.
      Negative shocks impact volatility differently than positive ones in energy.
    - Student-t handles fat tails from spike events (Nordic prices can swing
      from negative to several hundred EUR/MWh in a single hour).
    - Constant mean is more stable than Zero mean for short estimation windows.

    Conditional variance:
        sigma^2_t = omega + alpha * eps^2_{t-1}
                  + gamma * eps^2_{t-1} * I(eps_{t-1} < 0)
                  + beta * sigma^2_{t-1}
    """
    try:
        from arch import arch_model

        # Clean the data
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_returns) < 100:
            return None

        # Scale returns to percentage (helps with numerical stability)
        scaled_returns = clean_returns * 100

        # GJR-GARCH(1,1) with Student-t: p=1, o=1, q=1, dist='t'
        model = arch_model(
            scaled_returns,
            mean='Constant',
            vol='Garch',
            p=1, o=1, q=1,    # o=1 makes this GJR-GARCH
            dist='t',          # Student-t handles fat tails
            rescale=False
        )

        result = model.fit(disp='off', show_warning=False)
        return result

    except ImportError:
        return None
    except Exception:
        return None


def forecast_volatility(prices: pd.Series, horizon: int = 24):
    """Generate volatility forecast using GJR-GARCH model."""
    if len(prices) < 100:
        return None

    # Clean prices
    clean_prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean_prices) < 100:
        return None

    # Log returns (more stable for extreme values; safe against zeros/negatives)
    safe_prices = clean_prices.where(clean_prices > 0, np.nan).dropna()
    log_prices = np.log(safe_prices)
    returns = log_prices.diff().dropna()

    # Clip extreme outliers (electricity price spikes destabilise MLE)
    std = returns.std()
    mean = returns.mean()
    returns = returns[(returns > mean - 4*std) & (returns < mean + 4*std)]

    if len(returns) < 50:
        return None

    result = fit_garch_simple(returns)

    if result is None:
        # Fallback: historical volatility
        hist_vol = returns.std()
        last_date = clean_prices.index[-1]

        if hasattr(last_date, 'tz') and last_date.tz is not None:
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=horizon, freq='h', tz=last_date.tz
            )
        else:
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=horizon, freq='h'
            )

        return {
            'volatility': pd.Series([hist_vol] * horizon, index=forecast_index),
            'last_price': clean_prices.iloc[-1],
            'last_date': last_date,
            'model_params': {
                'omega': 0, 'alpha': 0, 'gamma': 0, 'beta': 0,
                'persistence': 0, 'aic': 0, 'bic': 0,
                'note': 'Historical volatility (GARCH fit failed)'
            }
        }

    try:
        # Extract parameters from GJR-GARCH fit
        omega = float(result.params.get('omega', 0))
        alpha = float(result.params.get('alpha[1]', 0))
        gamma = float(result.params.get('gamma[1]', 0))  # GJR asymmetric term
        beta = float(result.params.get('beta[1]', 0))

        # Persistence for GJR-GARCH (under symmetric shock distribution):
        # E[sigma^2_{t+1} | F_t] depends on alpha + beta + gamma/2
        persistence = alpha + beta + gamma / 2
        is_igarch = persistence >= IGARCH_THRESHOLD

        # Last conditional volatility (used for IGARCH fallback)
        conditional_vol = result.conditional_volatility
        last_cond_vol = conditional_vol.iloc[-1] / 100  # Unscale from percentage

        if is_igarch:
            # Unit-root case: forecast is essentially constant.
            # Report this honestly rather than scaling parameters.
            volatility = np.full(horizon, last_cond_vol)
        else:
            # Stationary case: analytic h-step-ahead forecast
            forecast = result.forecast(horizon=horizon)
            variance = forecast.variance.iloc[-1].values
            volatility = np.sqrt(variance) / 100  # Unscale from percentage

        # Sanity bounds: hourly electricity volatility typically 0.1%-30%
        volatility = np.clip(volatility, 0.001, 0.30)

        last_date = clean_prices.index[-1]

        if hasattr(last_date, 'tz') and last_date.tz is not None:
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=horizon, freq='h', tz=last_date.tz
            )
        else:
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=horizon, freq='h'
            )

        return {
            'volatility': pd.Series(volatility, index=forecast_index),
            'last_price': clean_prices.iloc[-1],
            'last_date': last_date,
            'model_params': {
                'omega': omega / 10000,  # Unscale variance from percentage^2
                'alpha': alpha,
                'gamma': gamma,
                'beta': beta,
                'persistence': persistence,
                'aic': float(result.aic),
                'bic': float(result.bic),
                'igarch': is_igarch
            }
        }

    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("Nordic-Baltic Power Price Dashboard")
    st.markdown("Real-time electricity price analysis with GJR-GARCH volatility forecasting")

    # Initialize database if needed (for Streamlit Cloud deployment)
    with st.spinner("Initializing database..."):
        init_result = initialize_database()

    if init_result.get("status") == "error":
        st.error(f"""
        **Initialization failed**

        **Reason:** {init_result.get('reason')}

        **For Streamlit Cloud:**
        1. Go to App Settings, then Secrets
        2. Add: `ENTSOE_API_TOKEN = "your-token-here"`
        3. Reboot the app
        """)
        return

    if init_result.get("status") == "initialized":
        st.success(f"Fetched initial data for: {', '.join(init_result.get('zones', []))}")
        if init_result.get("errors"):
            st.warning(f"Some zones had errors: {init_result.get('errors')}")

    # Check database exists (should be created by initialize_database)
    if not DATABASE_PATH.exists():
        st.error(f"""
        **Database not found after initialization**

        Path: `{DATABASE_PATH}`
        """)
        return

    # Auto-update prices on startup (cached for 1 hour)
    with st.spinner("Checking for latest prices..."):
        update_result = auto_update_prices()

    # Get database stats
    stats = get_database_stats()

    if not stats:
        st.error("""
        **Database is empty**

        Run: `python fetch_all_nordic.py`
        """)
        return

    # Sidebar - Zone selection
    st.sidebar.header("Settings")

    available_zones = list(stats.keys())
    if not available_zones:
        st.error("No data in database")
        return

    selected_zone = st.sidebar.selectbox(
        "Select Price Zone",
        options=available_zones,
        format_func=lambda x: f"{x} - {ZONES.get(x, x)}"
    )

    # Show data status
    zone_stats = stats.get(selected_zone, {})
    days_available = zone_stats.get('days', 0)
    record_count = zone_stats.get('count', 0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Status")
    st.sidebar.markdown(f"**Records:** {record_count:,}")
    st.sidebar.markdown(f"**Days:** {days_available}")
    st.sidebar.markdown(f"**From:** {zone_stats.get('min_date', 'N/A')[:10]}")
    st.sidebar.markdown(f"**To:** {zone_stats.get('max_date', 'N/A')[:10]}")

    # Show update status
    if update_result.get("status") == "success":
        updated_zones = update_result.get("updated", [])
        if updated_zones:
            st.sidebar.success(f"Updated {len(updated_zones)} zones")
        else:
            st.sidebar.info("Data is current")
    elif update_result.get("status") == "skipped":
        st.sidebar.warning(f"Auto-update skipped: {update_result.get('reason', 'unknown')}")

    # Manual refresh button
    if st.sidebar.button("Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()

    # Check if enough data
    if days_available < 7:
        st.error(f"""
        **Insufficient data for {selected_zone}**

        You have **{days_available} days** but need at least **7 days**.

        **To fix:** Run `python fetch_all_nordic.py`
        """)
        return

    # Load prices
    prices = get_prices(selected_zone)

    if prices.empty:
        st.error(f"No price data for {selected_zone}")
        return

    # Analysis period slider
    min_days = 7
    max_days = min(days_available - 2, 90)

    if max_days <= min_days:
        analysis_days = min_days
        st.sidebar.info(f"Using all available data ({analysis_days} days)")
    else:
        analysis_days = st.sidebar.slider(
            "Analysis Period (days)",
            min_value=min_days,
            max_value=max_days,
            value=min(30, max_days),
            help="Number of days to include in analysis"
        )

    # Filter to analysis period
    cutoff = prices.index[-1] - pd.Timedelta(days=analysis_days)
    prices = prices[prices.index >= cutoff]

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Forecast", "Backtest", "About"])

    # ========================================================================
    # TAB 1: FORECAST
    # ========================================================================
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Price History")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices.index,
                y=prices.values,
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=1)
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="EUR/MWh",
                height=400,
                margin=dict(l=50, r=20, t=30, b=50)
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Summary Statistics")

            current_price = prices.iloc[-1]
            price_24h_ago = prices.iloc[-24] if len(prices) > 24 else prices.iloc[0]
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100

            st.metric("Current Price", f"{current_price:.2f} €/MWh", f"{change_24h:+.1f}%")
            st.metric("24h High", f"{prices[-24:].max():.2f} €/MWh")
            st.metric("24h Low", f"{prices[-24:].min():.2f} €/MWh")
            st.metric("Period Mean", f"{prices.mean():.2f} €/MWh")
            st.metric("Volatility (std)", f"{prices.std():.2f}")

        st.markdown("---")

        # Forecast section
        st.subheader("GJR-GARCH Volatility Forecast")

        if len(prices) < 100:
            st.warning(f"Need at least 100 data points for GARCH. You have {len(prices)}.")
        else:
            if st.button("Generate 24h Forecast", type="primary"):
                with st.spinner("Fitting GJR-GARCH model..."):
                    forecast = forecast_volatility(prices, horizon=24)

                if forecast:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=forecast['volatility'].index,
                            y=forecast['volatility'].values * 100,
                            mode='lines+markers',
                            name='Forecast Volatility',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                        fig.update_layout(
                            title="24-Hour Volatility Forecast",
                            xaxis_title="Time",
                            yaxis_title="Volatility (%)",
                            height=350
                        )
                        st.plotly_chart(fig, width="stretch")

                    with col2:
                        st.markdown("**Model Parameters**")
                        params = forecast['model_params']
                        if params.get('note'):
                            st.warning(params['note'])
                        else:
                            st.markdown(f"- ω (omega): {params['omega']:.6f}")
                            st.markdown(f"- α (alpha): {params['alpha']:.4f}")
                            st.markdown(f"- γ (gamma): {params['gamma']:.4f}")
                            st.markdown(f"- β (beta): {params['beta']:.4f}")
                            st.markdown(f"- Persistence (α + β + γ/2): {params['persistence']:.4f}")
                            st.markdown(f"- AIC: {params['aic']:.2f}")
                            st.markdown(f"- BIC: {params['bic']:.2f}")
                            if params.get('igarch'):
                                st.info(
                                    "Persistence ~ 1 indicates IGARCH behavior. "
                                    "Volatility shocks have permanent effects, so the "
                                    "forecast trajectory is essentially constant."
                                )
                else:
                    st.error("Failed to generate forecast. Check that arch library is installed.")

    # ========================================================================
    # TAB 2: BACKTEST
    # ========================================================================
    with tab2:
        st.subheader("Walk-Forward Backtest")
        st.caption(
            "Note: this tab uses rolling standard deviation as a simple volatility "
            "benchmark. Full GJR-GARCH walk-forward backtesting is available via "
            "`src/models/pipeline.py` (command-line)."
        )

        if len(prices) < 100:
            st.warning(f"""
            Need at least 100 data points for backtesting.
            You have {len(prices)} points ({len(prices)//24} days).

            Run `python fetch_all_nordic.py` to get more data.
            """)
        else:
            max_test_days = min(days_available // 3, 30)
            min_test_days = 7

            if max_test_days <= min_test_days:
                st.warning("Not enough data for configurable backtest period.")
                test_days = 7
                train_days = 14
            else:
                col1, col2 = st.columns(2)
                with col1:
                    test_days = st.slider(
                        "Test Period (days)",
                        min_value=min_test_days,
                        max_value=max_test_days,
                        value=min(14, max_test_days)
                    )
                with col2:
                    max_train = days_available - test_days - 7
                    train_days = st.slider(
                        "Training Window (days)",
                        min_value=14,
                        max_value=max(14, min(60, max_train)),
                        value=min(30, max(14, max_train))
                    )

            if st.button("Run Backtest", type="primary"):
                with st.spinner("Running walk-forward backtest..."):
                    # Clean returns data
                    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

                    test_periods = test_days * 24
                    train_periods = train_days * 24

                    if len(returns) < train_periods + test_periods:
                        st.error("Not enough data for selected parameters")
                    else:
                        results = []

                        for i in range(test_periods):
                            end_train = len(returns) - test_periods + i
                            start_train = end_train - train_periods

                            if start_train < 0:
                                continue

                            train_returns = returns.iloc[start_train:end_train]

                            if end_train < len(returns):
                                actual = returns.iloc[end_train]
                                date_val = returns.index[end_train]

                                # Convert timezone-aware to naive for plotting
                                if hasattr(date_val, 'tz') and date_val.tz is not None:
                                    date_val = date_val.tz_localize(None)

                                predicted_vol = train_returns.std()
                                actual_vol = abs(actual)

                                if pd.notna(predicted_vol) and pd.notna(actual_vol):
                                    results.append({
                                        'date': date_val,
                                        'predicted': float(predicted_vol),
                                        'actual': float(actual_vol)
                                    })

                        if results and len(results) > 10:
                            df = pd.DataFrame(results)

                            # Store in session state
                            st.session_state['backtest_results'] = df
                            st.session_state['backtest_zone'] = selected_zone
                        else:
                            st.warning(f"Backtest produced only {len(results)} valid data points. Need more data.")

            # Display results from session state
            if 'backtest_results' in st.session_state and st.session_state.get('backtest_zone') == selected_zone:
                df = st.session_state['backtest_results']

                mse = ((df['predicted'] - df['actual'])**2).mean()
                mae = (df['predicted'] - df['actual']).abs().mean()
                correlation = df['predicted'].corr(df['actual'])

                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{np.sqrt(mse):.4f}")
                col2.metric("MAE", f"{mae:.4f}")
                col3.metric("Correlation", f"{correlation:.3f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['actual'],
                    name='Actual', mode='lines',
                    line=dict(color='#3B8BD4', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['predicted'],
                    name='Predicted', mode='lines',
                    line=dict(color='#E8593C', width=1)
                ))
                fig.update_layout(
                    title="Backtest: Predicted vs Actual Volatility",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.subheader("About This Dashboard")

        st.markdown("""
        ### Methodology

        This dashboard analyses Nordic-Baltic electricity spot prices using
        GJR-GARCH(1,1) with Student-t residuals — the standard specification
        in the energy econometrics literature.

        **GJR-GARCH(1,1) Model:**

        The conditional variance follows:

        $\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\gamma \\epsilon_{t-1}^2 \\mathbb{1}(\\epsilon_{t-1} < 0) + \\beta \\sigma_{t-1}^2$

        Where:
        - $\\omega$ (omega): Long-run variance weight
        - $\\alpha$ (alpha): Impact of recent shocks
        - $\\gamma$ (gamma): Asymmetric (leverage) effect — negative shocks
          have a different impact on volatility than positive shocks
        - $\\beta$ (beta): Persistence of volatility
        - $\\mathbb{1}(\\cdot)$: Indicator function for negative shocks

        **Persistence** in GJR-GARCH is $\\alpha + \\beta + \\gamma/2$.
        If this is close to 1, the model behaves as IGARCH (integrated GARCH),
        meaning volatility shocks have permanent rather than transient effects.
        This is reported transparently rather than constrained away.

        **Why Student-t residuals:**
        Hourly electricity returns exhibit much heavier tails than the Normal
        distribution allows for. Student-t with estimated degrees of freedom
        accommodates extreme spikes that are common in spot markets.

        **Data Source:**
        - ENTSO-E Transparency Platform
        - Day-ahead electricity prices (EUR/MWh)
        - Hourly resolution

        **Nordic-Baltic Price Zones:**

        **Norway:**
        - NO_1: Oslo (Southeast)
        - NO_2: Kristiansand (South)
        - NO_3: Trondheim (Central)
        - NO_4: Tromsø (North)
        - NO_5: Bergen (West)

        **Sweden:**
        - SE_1: Luleå (North)
        - SE_2: Sundsvall (North-Central)
        - SE_3: Stockholm (Central)
        - SE_4: Malmö (South)

        **Denmark:**
        - DK_1: West Denmark (Jutland)
        - DK_2: East Denmark (Zealand)

        **Finland:**
        - FI: Finland (single zone)

        **Estonia:**
        - EE: Estonia (single zone)

        **Latvia:**
        - LV: Latvia (single zone)

        **Lithuania:**
        - LT: Lithuania (single zone)
        """)


if __name__ == "__main__":
    main()
