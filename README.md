# NETAI Predictive Analytics & Forecasting (PoC)

Production-style Python PoC for the **Google Summer of Code (GSoC) NETAI project** focused on:

- forecasting network performance degradation from telemetry,
- comparing baseline and deep-learning forecasting approaches,
- generating automated LLM-powered incident reports for threshold breaches.

## Architecture

1. **Synthetic telemetry generation** (`data_gen.py`)
- Generates 30 days of telemetry at 5-minute intervals.
- Features: `timestamp`, `throughput_mbps`, `latency_ms`, `packet_loss_pct`.
- Injects realistic diurnal seasonality and 3-4 degradation events.

2. **Baseline forecaster (Prophet)** (`forecaster_baseline.py`)
- Trains a Prophet model on latency.
- Uses a holdout split (last 24 hours) for backtesting.
- Reports **MAE** and produces a 24-hour future forecast.

3. **Deep learning forecaster (PyTorch LSTM)** (`forecaster_dl.py`)
- Implements a lightweight LSTM for one-step time-series forecasting.
- Includes `Dataset` + `DataLoader` sliding windows.
- Trains with **MSE loss + Adam optimizer**.
- Reports holdout **MAE** and outputs a recursive 24-hour forecast.

4. **Alerting + LLM incident reporting** (`alerting.py`)
- Detects anomalies where forecasted latency exceeds threshold.
- Formats anomaly context into a prompt.
- Uses OpenAI client when API key exists, otherwise emits a mock request payload and fallback report.

## Repository Layout

```text
.
├── alerting.py
├── data_gen.py
├── forecaster_baseline.py
├── forecaster_dl.py
├── pyproject.toml
├── data/
└── outputs/
```

## Setup (uv)

```bash
# from repository root
uv venv
source .venv/bin/activate
uv sync
```

## Run End-to-End

### 1) Generate telemetry

```bash
uv run python data_gen.py \
  --output data/synthetic_network_telemetry.csv \
  --days 30 \
  --seed 42
```

### 2) Prophet baseline forecast + MAE

```bash
uv run python forecaster_baseline.py \
  --data-path data/synthetic_network_telemetry.csv \
  --backtest-output outputs/prophet_backtest.csv \
  --forecast-output outputs/prophet_forecast_24h.csv
```

### 3) LSTM forecast + MAE

```bash
uv run python forecaster_dl.py \
  --data-path data/synthetic_network_telemetry.csv \
  --backtest-output outputs/lstm_backtest.csv \
  --forecast-output outputs/lstm_forecast_24h.csv \
  --epochs 12 \
  --window-size 36
```

### 4) Alerting + LLM incident report generation

```bash
# Optional for live LLM calls
export OPENAI_API_KEY="your_api_key"

uv run python alerting.py \
  --forecast-path outputs/prophet_backtest.csv \
  --threshold-ms 35 \
  --output-path outputs/incident_reports.csv
```

## Prophet vs LSTM (PoC perspective)

- **Prophet strengths**: fast baseline, interpretable seasonality, stable on smaller datasets.
- **LSTM strengths**: better at nonlinear temporal relationships, extensible to multivariate inputs.
- **Tradeoff**: Prophet is simpler and faster to operationalize; LSTM offers flexibility but requires stronger data and tuning discipline.

## Notes for NETAI GSoC

This PoC is intentionally modular and production-oriented, enabling direct extension into:

- multivariate forecasting across interfaces/regions,
- probabilistic alerts using confidence intervals,
- streaming inference and continuous retraining,
- richer RAG-backed incident reporting with topology and runbook context.
