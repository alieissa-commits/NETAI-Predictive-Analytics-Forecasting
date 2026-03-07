"""Tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from netai_forecast.api.app import app, state, initialize_models


@pytest.fixture(scope="module")
def client():
    """Initialize models and create test client."""
    initialize_models()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["models_loaded"]) > 0
        assert data["version"] == "0.1.0"


class TestForecastEndpoint:
    def test_forecast_default(self, client):
        resp = client.post("/forecast", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast" in data
        assert len(data["forecast"]) == 12

    def test_forecast_with_model(self, client):
        resp = client.post("/forecast", json={
            "model_name": "arima",
            "steps": 6,
            "include_intervals": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "arima"
        assert len(data["forecast"]) == 6
        assert data["lower"] is not None
        assert data["upper"] is not None

    def test_forecast_no_intervals(self, client):
        resp = client.post("/forecast", json={
            "model_name": "arima",
            "steps": 6,
            "include_intervals": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["lower"] is None

    def test_forecast_invalid_model(self, client):
        resp = client.post("/forecast", json={"model_name": "nonexistent"})
        assert resp.status_code == 400


class TestEarlyWarningEndpoint:
    def test_early_warning(self, client):
        resp = client.post("/early-warning?metric=throughput_mbps&steps=12")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert "summary" in data


class TestIncidentReportEndpoint:
    def test_incident_report(self, client):
        resp = client.post("/incident-report?metric=throughput_mbps&steps=12")
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert len(data) > 0


class TestEvaluateEndpoint:
    def test_evaluate(self, client):
        resp = client.get("/evaluate?metric=throughput_mbps&steps=12")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
