<div align="center">

# 🌐 NETAI — Predictive Analytics & Forecasting

**AI-powered time-series forecasting for network performance metrics on the National Research Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-76%20passed-brightgreen.svg)](#testing)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-orange.svg)](https://summerofcode.withgoogle.com/)

*Prototype for Google Summer of Code 2026 — NETAI / Predictive Analytics & Forecasting*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Early Warning System](#early-warning-system)
- [Incident Report Generation](#incident-report-generation)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Testing](#testing)
- [GSoC Proposal Context](#gsoc-proposal-context)

---

## Overview

NETAI Predictive Analytics & Forecasting is an end-to-end platform that predicts network performance degradation **before it impacts users**. It combines classical statistical methods (ARIMA), modern ML (Prophet), and deep learning (LSTM, Transformer) into an intelligent forecasting ensemble for network metrics like throughput, latency, packet loss, and retransmits.

This prototype demonstrates the complete pipeline from data ingestion through prediction to automated incident reporting, designed for deployment on the **National Research Platform (NRP)** using Kubernetes and GPU resources.

### Key Capabilities

| Capability | Description |
|---|---|
| **Multi-Model Forecasting** | ARIMA, Prophet, LSTM, Transformer with weighted ensemble |
| **Early Warning System** | Detects degradation trends before they become critical |
| **Few-Shot Adaptation** | Rapidly adapts to new network topologies with minimal data |
| **Incident Reports** | LLM-powered automated report generation (NRP LLM service) |
| **REST API** | Production-ready FastAPI service with prediction intervals |
| **Kubernetes-Native** | GPU training jobs, deployment manifests, ConfigMaps |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     NETAI Forecasting Platform                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐    │
│  │  perfSONAR   │    │         Forecasting Engine               │    │
│  │  Data Layer  │───▶│                                          │    │
│  │             │    │  ┌────────┐ ┌─────────┐ ┌──────┐        │    │
│  │ • SQLite    │    │  │ ARIMA  │ │ Prophet │ │ LSTM │        │    │
│  │ • CSV/API   │    │  └───┬────┘ └────┬────┘ └──┬───┘        │    │
│  │ • Synthetic │    │      │           │         │             │    │
│  └─────────────┘    │      │    ┌──────┴─────┐   │             │    │
│                      │      │    │Transformer │   │             │    │
│                      │      │    └──────┬─────┘   │             │    │
│                      │      └─────┬─────┴─────────┘             │    │
│                      │            ▼                              │    │
│                      │    ┌───────────────┐                     │    │
│                      │    │   Ensemble     │                     │    │
│                      │    │  (Weighted)    │                     │    │
│                      │    └───────┬───────┘                     │    │
│                      └────────────┼─────────────────────────────┘    │
│                                   │                                  │
│                      ┌────────────▼────────────┐                    │
│                      │   Early Warning System   │                    │
│                      │                          │                    │
│                      │ • Threshold Detection    │                    │
│                      │ • Trend Analysis         │                    │
│                      │ • Rate-of-Change         │                    │
│                      └────────────┬─────────────┘                    │
│                                   │                                  │
│                      ┌────────────▼────────────┐                    │
│                      │  Incident Report Gen     │                    │
│                      │  (LLM / Template)        │                    │
│                      └────────────┬─────────────┘                    │
│                                   │                                  │
│                      ┌────────────▼────────────┐                    │
│                      │    FastAPI REST API      │                    │
│                      │    /forecast             │                    │
│                      │    /early-warning        │                    │
│                      │    /incident-report      │                    │
│                      │    /evaluate             │                    │
│                      └──────────────────────────┘                    │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  Kubernetes: GPU Pods │ ConfigMaps │ PVC │ Services │ Jobs           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 🔮 Time-Series Forecasting Models

- **ARIMA/SARIMA** — Classical statistical model capturing trend and seasonality via auto-regressive integrated moving average. Supports automatic order selection.
- **Prophet** — Facebook's additive decomposition model with built-in handling of daily/weekly seasonality, trend changepoints, and holidays.
- **LSTM** — Multi-layer Long Short-Term Memory network with dropout regularization, early stopping, and MC Dropout uncertainty estimation.
- **Transformer** — Encoder-based architecture with positional encoding, multi-head self-attention, and cosine annealing learning rate schedule.
- **Ensemble** — Weighted combination of all models with inverse-MSE weight optimization on validation data.

### 🧠 Few-Shot Learning

MAML-inspired adaptation that takes a pre-trained LSTM model and fine-tunes it on a small support set (as few as 10 samples) from a new network topology, enabling rapid deployment to unseen network configurations.

### ⚠️ Early Warning System

Multi-strategy degradation detection:
- **Absolute threshold** violations (configurable per metric)
- **Relative change** from recent baseline
- **Trend analysis** (linear regression on forecast window)
- **Confidence scoring** based on prediction interval width

### 📝 Automated Incident Reports

- **Primary:** Calls NRP's managed LLM service (Qwen3-VL, GLM-4.7, GPT-OSS) for natural language reports
- **Fallback:** Structured template-based reports with metric-specific summaries, impact analysis, root cause hypotheses, and recommended actions

### 🚀 REST API

FastAPI service with endpoints for forecasting, early warning, incident reports, model comparison, and health checks. Supports prediction intervals and model selection.

---

## Project Structure

```
NETAI-Predictive-Analytics-Forecasting-Ali/
├── netai_forecast/                # Main package
│   ├── data/                      # Data generation, preprocessing, loading
│   │   ├── generator.py           # Synthetic perfSONAR data generator
│   │   ├── preprocessing.py       # Normalization, sequencing, feature engineering
│   │   └── perfsonar_loader.py    # SQLite/CSV/Parquet data loader
│   ├── models/                    # Forecasting models
│   │   ├── base.py                # Abstract base forecaster interface
│   │   ├── arima_model.py         # ARIMA/SARIMA
│   │   ├── prophet_model.py       # Facebook Prophet
│   │   ├── lstm_model.py          # LSTM with MC Dropout uncertainty
│   │   ├── transformer_model.py   # Transformer encoder forecaster
│   │   ├── ensemble.py            # Weighted ensemble with weight optimization
│   │   └── few_shot.py            # MAML-inspired few-shot adapter
│   ├── early_warning/             # Degradation detection
│   │   ├── detector.py            # Multi-strategy degradation detector
│   │   └── alerting.py            # Alert lifecycle management
│   ├── incident_report/           # Report generation
│   │   └── report_generator.py    # LLM + template report generator
│   ├── evaluation/                # Metrics and comparison
│   │   └── metrics.py             # MAE, RMSE, MAPE, sMAPE, MASE
│   └── api/                       # REST API
│       └── app.py                 # FastAPI application
├── tests/                         # Comprehensive test suite (76 tests)
│   ├── test_data.py               # Data generation & preprocessing tests
│   ├── test_models.py             # All model tests (ARIMA, LSTM, Transformer, etc.)
│   ├── test_early_warning.py      # Detector and alerting tests
│   ├── test_incident_report.py    # Report generation tests
│   ├── test_evaluation.py         # Metrics tests
│   └── test_api.py                # API endpoint tests
├── scripts/                       # CLI utilities
│   ├── train.py                   # Full training pipeline
│   └── evaluate.py                # Model evaluation & comparison
├── configs/
│   └── default_config.yaml        # All hyperparameters and settings
├── k8s/                           # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployment.yaml            # API deployment (2 replicas)
│   ├── service.yaml               # ClusterIP service
│   ├── gpu-training-job.yaml      # GPU training job (nvidia.com/gpu)
│   ├── configmap.yaml             # Configuration
│   └── pvc.yaml                   # Persistent model storage
├── Dockerfile                     # Container image
├── docker-compose.yml             # Local development
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── pyproject.toml                 # Tool configuration
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NETAI-Predictive-Analytics-Forecasting-Ali.git
cd NETAI-Predictive-Analytics-Forecasting-Ali

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Quick Start

```python
from netai_forecast.data.generator import NetworkDataGenerator
from netai_forecast.models.arima_model import ARIMAForecaster
from netai_forecast.models.lstm_model import LSTMForecaster
from netai_forecast.models.ensemble import EnsembleForecaster
from netai_forecast.early_warning.detector import DegradationDetector
from netai_forecast.evaluation.metrics import evaluate_forecast

# Generate synthetic perfSONAR data
gen = NetworkDataGenerator(profile="wan_research", num_days=30, seed=42)
data = gen.generate()

# Split chronologically
train = data.iloc[:int(len(data)*0.7)]
test = data.iloc[int(len(data)*0.85):]

# Train ARIMA
arima = ARIMAForecaster(order=(2, 1, 2), forecast_horizon=12)
arima.fit(train, metric="throughput_mbps")

# Train LSTM
lstm = LSTMForecaster(hidden_size=64, epochs=10, forecast_horizon=12)
lstm.fit(train, metric="throughput_mbps", feature_columns=["throughput_mbps"])

# Create ensemble
ensemble = EnsembleForecaster(forecast_horizon=12)
ensemble.add_model(arima, weight=0.4)
ensemble.add_model(lstm, weight=0.6)
ensemble._is_fitted = True

# Forecast with prediction intervals
result = ensemble.predict_with_intervals(steps=12, confidence=0.95)
print(f"Forecast: {result['forecast'][:5]}")
print(f"Lower CI: {result['lower'][:5]}")
print(f"Upper CI: {result['upper'][:5]}")

# Detect degradation
detector = DegradationDetector()
events = detector.detect(result, recent_history=train.tail(288), metric="throughput_mbps")
print(f"Degradation events: {len(events)}")

# Evaluate
metrics = evaluate_forecast(test["throughput_mbps"].values[:12], result["forecast"])
print(f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
```

---

## Usage

### Training Models

```bash
# Train all models with default config
python scripts/train.py --config configs/default_config.yaml --metric throughput_mbps

# Train specific models with custom data size
python scripts/train.py --models arima lstm transformer --days 60 --profile wan_research

# Evaluate models
python scripts/evaluate.py --metric latency_ms --horizon 12
```

### Running the API

```bash
# Start the API server
uvicorn netai_forecast.api.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up
```

### Docker

```bash
# Build image
docker build -t netai-forecast .

# Run container
docker run -p 8000:8000 netai-forecast
```

---

## API Reference

### `GET /health`
Health check and model status.

```json
{
  "status": "healthy",
  "models_loaded": ["arima", "prophet", "lstm", "transformer", "ensemble"],
  "version": "0.1.0"
}
```

### `POST /forecast`
Generate network metric forecasts.

**Request:**
```json
{
  "metric": "throughput_mbps",
  "steps": 12,
  "model_name": "ensemble",
  "include_intervals": true,
  "confidence": 0.95
}
```

**Response:**
```json
{
  "model": "ensemble",
  "metric": "throughput_mbps",
  "steps": 12,
  "forecast": [4987.23, 4991.45, ...],
  "lower": [4823.11, 4830.22, ...],
  "upper": [5151.35, 5152.68, ...],
  "generated_at": "2026-02-28T04:00:00.000Z"
}
```

### `POST /early-warning`
Run degradation detection on forecasted metrics.

### `POST /incident-report`
Generate automated incident report from detected degradation events.

### `GET /models`
List available models and their parameters.

### `GET /evaluate`
Evaluate all models against test data.

---

## Model Details

### ARIMA (Autoregressive Integrated Moving Average)

- **Strengths:** Fast training, interpretable, good for linear trends
- **Config:** ARIMA(2,1,2) with optional SARIMA seasonal component
- **Use case:** Baseline forecasting, short-horizon predictions

### Prophet

- **Strengths:** Handles multiple seasonalities, robust to missing data
- **Config:** Daily + weekly seasonality, 0.05 changepoint prior
- **Use case:** Capturing complex seasonal patterns in network traffic

### LSTM (Long Short-Term Memory)

- **Architecture:** 2-layer LSTM (128 hidden) → FC layers → multi-step output
- **Training:** Adam optimizer, gradient clipping, early stopping
- **Uncertainty:** Monte Carlo Dropout (100 forward passes)
- **Use case:** Complex non-linear patterns, multi-feature input

### Transformer

- **Architecture:** Linear projection → Positional encoding → 3-layer Transformer encoder → Global average pool → FC head
- **Training:** AdamW + Cosine annealing LR, gradient clipping
- **Uncertainty:** Monte Carlo Dropout
- **Use case:** Long-range dependencies, attention-based pattern recognition

### Ensemble

- **Strategy:** Weighted average with inverse-MSE optimization
- **Default weights:** ARIMA 0.15, Prophet 0.20, LSTM 0.30, Transformer 0.35
- **Adaptive:** Weights optimized on validation data automatically

### Few-Shot Adapter

- **Method:** MAML-inspired fine-tuning of pre-trained LSTM
- **Adaptation:** 5 gradient steps on 10-sample support set
- **Use case:** Rapid deployment to new NRP network topologies

---

## Early Warning System

The degradation detector combines three detection strategies:

1. **Threshold Detection** — Compares forecasted values against configurable warning/critical thresholds per metric
2. **Relative Change** — Detects significant percentage changes from the recent baseline
3. **Trend Analysis** — Fits linear regression to the forecast window to detect sustained degradation

```yaml
# Example thresholds (configs/default_config.yaml)
early_warning:
  thresholds:
    throughput_mbps:
      warning_drop_pct: 20.0    # 20% drop → WARNING
      critical_drop_pct: 50.0   # 50% drop → CRITICAL
    latency_ms:
      warning_increase_pct: 30.0
      critical_increase_pct: 100.0
    packet_loss_pct:
      warning_threshold: 1.0    # 1% loss → WARNING
      critical_threshold: 5.0   # 5% loss → CRITICAL
```

---

## Incident Report Generation

Reports are generated using a dual approach:

1. **LLM-Powered** (Primary) — Sends degradation context to NRP's managed LLM service for natural language analysis
2. **Template-Based** (Fallback) — Uses metric-specific templates when LLM is unavailable

Each report includes:
- Severity assessment
- Impact analysis on NRP infrastructure
- Root cause hypothesis
- Specific recommended actions
- Forecast data supporting the analysis

---

## Kubernetes Deployment

Deploy on NRP's Kubernetes cluster:

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy configuration and storage
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml

# Run GPU training job
kubectl apply -f k8s/gpu-training-job.yaml

# Deploy API service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The GPU training job requests `nvidia.com/gpu: 1` and includes tolerations for GPU-enabled nodes.

---

## Testing

76 comprehensive tests covering all components:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_data.py          # 21 tests — data generation & preprocessing
python -m pytest tests/test_models.py        # 16 tests — all forecasting models
python -m pytest tests/test_early_warning.py # 11 tests — degradation detection & alerting
python -m pytest tests/test_incident_report.py # 8 tests — report generation
python -m pytest tests/test_evaluation.py    # 11 tests — metrics
python -m pytest tests/test_api.py           #  9 tests — API endpoints
```

---

## GSoC Proposal Context

This prototype demonstrates the skills and implementation approach for the **NETAI / Predictive Analytics & Forecasting** project (350 hours, Hard difficulty).

### Deliverables Mapped to Proposal

| Proposal Requirement | Prototype Implementation |
|---|---|
| Time-series forecasting models (ARIMA, Prophet, DL) | ✅ ARIMA, Prophet, LSTM, Transformer + Ensemble |
| Few-shot learning for new topologies | ✅ MAML-inspired FewShotAdapter |
| Early warning system | ✅ Multi-strategy DegradationDetector |
| Automated incident reports via LLMs | ✅ LLM + template IncidentReportGenerator |
| Kubernetes deployment | ✅ Full K8s manifests with GPU support |
| GPU training | ✅ GPU training job with nvidia.com/gpu tolerations |
| REST API | ✅ FastAPI with 6 endpoints |
| Integration with perfSONAR data | ✅ PerfSONARLoader (SQLite, CSV, Parquet) |

### Technologies Demonstrated

Python, PyTorch, Prophet, statsmodels (ARIMA), scikit-learn, Pandas, NumPy, FastAPI, Docker, Kubernetes, GPU Computing

---

## License

This project is licensed under the Apache License 2.0.

---

<div align="center">

**Built for GSoC 2026 — NETAI / Predictive Analytics & Forecasting**

Mentors: Dmitry Mishin, Derek Weitzel | Organization: National Research Platform

</div>
