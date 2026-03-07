"""FastAPI REST API for NETAI Predictive Analytics & Forecasting.

Provides endpoints for:
- Generating forecasts for network metrics
- Early warning detection
- Incident report generation
- Model comparison and evaluation
- Health checks
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..data.generator import NetworkDataGenerator
from ..data.preprocessing import preprocess_timeseries, train_val_test_split
from ..models.arima_model import ARIMAForecaster
from ..models.prophet_model import ProphetForecaster
from ..models.lstm_model import LSTMForecaster
from ..models.transformer_model import TransformerForecaster
from ..models.ensemble import EnsembleForecaster
from ..early_warning.detector import DegradationDetector
from ..early_warning.alerting import AlertManager
from ..incident_report.report_generator import IncidentReportGenerator
from ..evaluation.metrics import evaluate_forecast

logger = logging.getLogger(__name__)

# ── Pydantic models ──────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    metric: str = Field(default="throughput_mbps", description="Network metric to forecast")
    steps: int = Field(default=12, ge=1, le=288, description="Number of forecast steps")
    model_name: str = Field(default="ensemble", description="Model to use: arima, prophet, lstm, transformer, ensemble")
    include_intervals: bool = Field(default=True, description="Include prediction intervals")
    confidence: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals")


class ForecastResponse(BaseModel):
    model: str
    metric: str
    steps: int
    forecast: list[float]
    lower: Optional[list[float]] = None
    upper: Optional[list[float]] = None
    generated_at: str


class WarningResponse(BaseModel):
    alerts: list[dict]
    summary: dict


class ReportResponse(BaseModel):
    report: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str


class EvaluationResponse(BaseModel):
    results: dict


# ── Application state ────────────────────────────────────────────────

class AppState:
    """Holds loaded models and data."""
    def __init__(self):
        self.models: dict[str, object] = {}
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.detector = DegradationDetector()
        self.alert_manager = AlertManager()
        self.report_generator = IncidentReportGenerator(use_fallback=True)
        self.is_initialized = False


state = AppState()


def initialize_models(metric: str = "throughput_mbps"):
    """Initialize models with synthetic data for demo purposes."""
    if state.is_initialized:
        return

    logger.info("Initializing NETAI forecasting models...")

    # Generate synthetic data
    gen = NetworkDataGenerator(profile="wan_research", num_days=30, seed=42)
    state.data = gen.generate()

    # Split data
    state.train_data, state.val_data, state.test_data = train_val_test_split(state.data)

    # Initialize ARIMA
    arima = ARIMAForecaster(order=(2, 1, 2), forecast_horizon=12)
    try:
        arima.fit(state.train_data, metric=metric)
        state.models["arima"] = arima
        logger.info("ARIMA model fitted successfully")
    except Exception as e:
        logger.warning(f"ARIMA fitting failed: {e}")

    # Initialize Prophet
    prophet = ProphetForecaster(forecast_horizon=12)
    try:
        prophet.fit(state.train_data, metric=metric)
        state.models["prophet"] = prophet
        logger.info("Prophet model fitted successfully")
    except Exception as e:
        logger.warning(f"Prophet fitting failed: {e}")

    # Initialize LSTM (lightweight for API startup)
    lstm = LSTMForecaster(
        hidden_size=64, num_layers=1, epochs=5, sequence_length=48,
        forecast_horizon=12, batch_size=32, patience=3,
    )
    try:
        lstm.fit(state.train_data, metric=metric, val_data=state.val_data,
                 feature_columns=[metric])
        state.models["lstm"] = lstm
        logger.info("LSTM model fitted successfully")
    except Exception as e:
        logger.warning(f"LSTM fitting failed: {e}")

    # Initialize Transformer (lightweight for API startup)
    transformer = TransformerForecaster(
        d_model=32, nhead=2, num_encoder_layers=2, epochs=5,
        sequence_length=48, forecast_horizon=12, batch_size=32, patience=3,
    )
    try:
        transformer.fit(state.train_data, metric=metric, val_data=state.val_data,
                        feature_columns=[metric])
        state.models["transformer"] = transformer
        logger.info("Transformer model fitted successfully")
    except Exception as e:
        logger.warning(f"Transformer fitting failed: {e}")

    # Create ensemble
    if state.models:
        ensemble = EnsembleForecaster(forecast_horizon=12)
        for model in state.models.values():
            ensemble.add_model(model)
        ensemble._is_fitted = True
        state.models["ensemble"] = ensemble
        logger.info("Ensemble model created")

    state.is_initialized = True
    logger.info(f"Models initialized: {list(state.models.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_models()
    yield


# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(
    title="NETAI Predictive Analytics API",
    description=(
        "AI-powered predictive analytics and forecasting for network performance metrics. "
        "Part of the NETAI platform for the National Research Platform (NRP)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and list loaded models."""
    return HealthResponse(
        status="healthy" if state.is_initialized else "initializing",
        models_loaded=list(state.models.keys()),
        version="0.1.0",
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Generate network metric forecasts."""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Models not yet initialized")

    model = state.models.get(request.model_name)
    if model is None:
        available = list(state.models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not available. Options: {available}",
        )

    try:
        if request.include_intervals:
            result = model.predict_with_intervals(
                steps=request.steps, confidence=request.confidence
            )
            return ForecastResponse(
                model=request.model_name,
                metric=request.metric,
                steps=request.steps,
                forecast=[round(float(v), 4) for v in result["forecast"]],
                lower=[round(float(v), 4) for v in result["lower"]],
                upper=[round(float(v), 4) for v in result["upper"]],
                generated_at=datetime.utcnow().isoformat(),
            )
        else:
            pred = model.predict(steps=request.steps)
            return ForecastResponse(
                model=request.model_name,
                metric=request.metric,
                steps=request.steps,
                forecast=[round(float(v), 4) for v in pred],
                generated_at=datetime.utcnow().isoformat(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/early-warning", response_model=WarningResponse)
async def early_warning(
    metric: str = Query(default="throughput_mbps"),
    steps: int = Query(default=12, ge=1, le=288),
):
    """Run early warning detection on forecasted metrics."""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Models not yet initialized")

    model = state.models.get("ensemble") or next(iter(state.models.values()))

    forecast_result = model.predict_with_intervals(steps=steps)
    recent_history = state.train_data.tail(288)  # Last day

    events = state.detector.detect(
        forecast=forecast_result,
        recent_history=recent_history,
        metric=metric,
        forecast_intervals=forecast_result,
    )

    alerts = state.alert_manager.process_events(events)
    summary = state.alert_manager.get_summary()

    return WarningResponse(
        alerts=[a.to_dict() for a in alerts],
        summary=summary,
    )


@app.post("/incident-report", response_model=ReportResponse)
async def incident_report(
    metric: str = Query(default="throughput_mbps"),
    steps: int = Query(default=12),
):
    """Generate an automated incident report."""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="Models not yet initialized")

    model = state.models.get("ensemble") or next(iter(state.models.values()))

    forecast_result = model.predict_with_intervals(steps=steps)
    recent_history = state.train_data.tail(288)

    events = state.detector.detect(
        forecast=forecast_result,
        recent_history=recent_history,
        metric=metric,
    )

    if not events:
        return ReportResponse(report={"message": "No degradation events detected. System healthy."})

    report = state.report_generator.generate(events)
    return ReportResponse(report=report.to_dict())


@app.get("/models", response_model=dict)
async def list_models():
    """List available models and their parameters."""
    return {
        name: model.get_params() if hasattr(model, "get_params") else {"name": name}
        for name, model in state.models.items()
    }


@app.get("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    metric: str = Query(default="throughput_mbps"),
    steps: int = Query(default=12),
):
    """Evaluate all models on test data."""
    if not state.is_initialized or state.test_data is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    actual = state.test_data[metric].values[:steps]
    results = {}

    for name, model in state.models.items():
        try:
            pred = model.predict(steps=steps)
            min_len = min(len(pred), len(actual))
            metrics = evaluate_forecast(actual[:min_len], pred[:min_len])
            results[name] = metrics
        except Exception as e:
            results[name] = {"error": str(e)}

    return EvaluationResponse(results=results)
