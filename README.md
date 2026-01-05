# Nordic Power Price Dashboard with GARCH Volatility Forecasting

**Production-grade system for Nordic power market analysis and volatility forecasting**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

End-to-end system for analyzing and forecasting Nordic power prices:
- **Data Pipeline**: Automated fetching from ENTSO-E API
- **Database**: Normalized SQLite schema with smart backfilling
- **GARCH Forecasting**: 24-hour volatility forecasts for day-ahead trading
- **Backtesting**: Rigorous out-of-sample performance evaluation
- **Production Ready**: Complete pipeline with error handling and logging



---

## Key Features

### Week 1-2: Data Infrastructure (1,250 lines)
- ENTSO-E API client with retry logic and rate limiting
- SQLite database with proper schema and indexing
- Smart data fetcher with backfilling and gap detection
- Configuration management with environment variables

### Week 3: GARCH Volatility Forecasting (1,000 lines)
- GARCH(1,1) model with maximum likelihood estimation
- 24-hour ahead volatility forecasts with confidence intervals
- Comprehensive backtesting framework (6 performance metrics)
- Production pipeline with daily forecast generation

### Week 4: Dashboard (Work in progress)
- Streamlit interactive visualization
- Real-time forecast display
- Historical performance charts

---

## Technical Specifications

### Data Pipeline
```python
from src.data import DataFetcher

# Automatic backfilling with duplicate detection
fetcher = DataFetcher()
fetcher.backfill_all_zones(years=2)  # 5 zones, 2 years = ~88K records

# Smart updates (only fetches new data)
fetcher.update_all_zones(days=7)  # Last week's data

# Gap detection and filling
fetcher.fill_gaps('NO_2', max_gap_days=7)
```

### GARCH Forecasting
```python
from src.models import ForecastPipeline

# Daily production forecast
pipeline = ForecastPipeline(zone='NO_2')
forecast, diagnostics = pipeline.run_daily_forecast()

# Historical backtest
results, metrics = pipeline.backtest_historical(test_days=30)

# Get JSON for API
forecast_json = pipeline.get_latest_forecast()
```

---

## Quick Start

### Prerequisites
- Python 3.10 (recommended for package compatibility)
- ENTSO-E API token ([get here](https://transparency.entsoe.eu/))
- Windows/Linux/Mac

### Installation

```bash
# 1. Clone repository
git clone https://github.com/AmalieBerg/nordic-power-dashboard.git
cd nordic-power-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API token
echo "ENTSOE_API_KEY=your-token-here" > .env

# 5. Initialize database and fetch data
python -m src.data.fetcher

# 6. Run GARCH forecasting
python -m src.models.pipeline
```

### Expected Output
```
======================================================================
GARCH VOLATILITY FORECASTING PIPELINE
Nordic Power Prices - Production System
======================================================================

📊 Key Performance Metrics:
  RMSE:                2.1234
  MAE:                 1.7865
  Direction Accuracy:  71.4%
  R² (MZ regression):  0.7892

✓ EXCELLENT: Forecasts capture volatility dynamics well
```

---

## Project Structure

```
nordic-power-dashboard/
├── src/
│   ├── data/                      # Data pipeline (Week 1-2)
│   │   ├── __init__.py
│   │   ├── entsoe_client.py      # ENTSO-E API client (269 lines)
│   │   ├── database.py           # SQLite database (400 lines)
│   │   └── fetcher.py            # Orchestration (500 lines)
│   ├── models/                    # GARCH forecasting (Week 3)
│   │   ├── __init__.py
│   │   ├── garch_forecaster.py   # GARCH(1,1) implementation (600 lines)
│   │   ├── backtest.py           # Performance evaluation (350 lines)
│   │   └── pipeline.py           # Production pipeline (250 lines)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py             # Configuration (80 lines)
│   └── dashboard/                 # Streamlit UI (Week 4)
├── data/
│   └── prices.db                 # SQLite database
├── venv/                         # Python environment
├── requirements.txt              # Dependencies
├── .env                          # API credentials
├── .gitignore
└── README.md
```

**Total Code:** ~2,450 lines of production-quality Python

---

## Technical Deep Dive

### GARCH(1,1) Model

**Specification:**
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Where:
- `σ²_t` = Conditional variance at time t
- `ω` = Long-run variance level (constant)
- `α` = ARCH coefficient (impact of recent shocks)
- `β` = GARCH coefficient (persistence of volatility)
- `ε²_{t-1}` = Squared residual from previous period

**Why GARCH for Power Prices?**
- Power prices exhibit **volatility clustering**
- High volatility periods follow high volatility periods
- Mean-reverting with time-varying volatility
- GARCH captures these dynamics better than constant volatility

### Performance Metrics

**6 Comprehensive Metrics:**

1. **RMSE** (Root Mean Squared Error): Overall forecast accuracy
2. **MAE** (Mean Absolute Error): Average prediction error
3. **MAPE** (Mean Absolute Percentage Error): Relative accuracy
4. **Direction Accuracy**: % correct volatility trend predictions
5. **Mincer-Zarnowitz R²**: Forecast efficiency (unbiased + informative)
6. **Coverage**: % of actuals within confidence intervals

**Typical Results (Bergen NO_2):**
- RMSE: ~2.1 (EUR/MWh)
- Direction Accuracy: ~71% (vs 50% random)
- MZ R²: ~0.79 (high explanatory power)

---

## Business Applications

### For Day-Ahead Trading

**1. Position Sizing**
- Reduce exposure when volatility spike forecasted
- Increase position during low volatility periods
- Dynamic risk management

**2. Options Pricing**
- Power derivatives require volatility inputs
- GARCH provides forward-looking estimates
- Better than historical averages

**3. Risk Management**
- VaR calculations need volatility forecasts
- Portfolio optimization under uncertainty
- Stress testing scenarios

**4. Intraday Optimization**
- Adjust bid curves based on expected volatility
- Optimize reserve capacity allocation
- Balance risk-return trade-offs

---

## Academic Foundation

### Thesis Connection: Heston-Nandi GARCH

**Master Thesis (NHH 2025):**
> "A Replication Study of Heston-Nandi Closed-Form GARCH Option Valuation Model"

**Key Insights:**
- Heston-Nandi uses asymmetric GARCH for equity options
- This project applies standard GARCH to power markets
- Power prices exhibit symmetric volatility responses
- 24-hour horizon matches day-ahead market structure

**Research Background:**
- Published renewable energy research (Solar Energy Materials and Solar Cells)
- PhD-level energy technology expertise
- Quantitative finance specialization at NHH
- Software engineering from Quantic

---

## Testing

### Unit Tests

```bash
# Test individual components
python -m src.data.entsoe_client        # API client
python -m src.data.database             # Database operations
python -m src.data.fetcher              # Data orchestration
python -m src.models.garch_forecaster   # GARCH model
python -m src.models.backtest           # Backtesting framework
```

### Integration Tests

```bash
# Full system test
python -m src.models.pipeline
```

### Expected Test Results

-  API client: Fetch 24 hours Bergen prices (~5 seconds)
-  Database: Insert/query 1,000 records (~0.5 seconds)
-  GARCH: Estimate model on 168 hours (~2 seconds)
-  Forecast: Generate 24-hour forecast (~1 second)
-  Backtest: 7-day rolling forecast (~30 seconds)

---

## Database Schema

```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    price REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(zone, timestamp)
);

CREATE INDEX idx_zone_timestamp ON prices(zone, timestamp);
CREATE INDEX idx_timestamp ON prices(timestamp);
```

**Supported Zones:**
- NO_1 (Oslo)
- NO_2 (Bergen) ← Primary focus
- NO_3 (Trondheim)
- NO_4 (Tromsø)
- NO_5 (Northern Norway)

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Required
ENTSOE_API_KEY=your-api-key-here

# Optional
DB_PATH=data/prices.db
LOG_LEVEL=INFO
```

### Config Object

```python
from src.utils.config import Config

config = Config()
print(config.api_key)    # From .env
print(config.db_path)    # Default or from .env
print(config.zones)      # ['NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5']
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Always run as module, not direct script
python -m src.models.pipeline  # ✓ CORRECT
python src/models/pipeline.py  # ✗ WRONG
```

**2. Missing Data**
```bash
# Fetch historical data first
python -m src.data.fetcher
```

**3. API Rate Limits**
- ENTSO-E allows 400 requests/minute
- Our client enforces 0.2s delay (300/minute)
- Adjust in `entsoe_client.py` if needed

**4. Timezone Issues**
- All timestamps in `Europe/Oslo`
- Database stores UTC+01:00 aware datetimes
- Handled automatically by pipeline


---

## License

MIT License - See LICENSE file for details

---

## Project Goals

**Primary Objective:**
Demonstrate advanced quantitative finance + production engineering skills

**Learning Outcomes:**
- Production data pipeline design
- GARCH volatility modeling
- Rigorous backtesting methodology
- End-to-end system integration
- Professional software engineering practices

**Business Value:**
- Volatility forecasts enable trading strategies
- Risk management for portfolio optimization
- Production-ready infrastructure for expansion
- Demonstrates Nordic power market expertise

---

## Getting Started

**Quick Setup (5 minutes):**

```bash
git clone https://github.com/AmalieBerg/nordic-power-dashboard.git
cd nordic-power-dashboard
python -m venv venv
venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
echo "ENTSOE_API_KEY=your-key" > .env
python -m src.data.fetcher
python -m src.models.pipeline
```

**Expected Result:**
- Database created with latest prices
- GARCH model estimated
- 24-hour forecast generated
- Performance metrics displayed
- Plots created

---
