import json
import pytest
import asyncio
from web.app import serve_index, get_demo, get_demo_json

def test_web_ui_loads():
    """Test that the index route serves the UI without errors."""
    response = asyncio.run(serve_index())
    assert response is not None
    text = response
    assert "Mastercard Payment Security Lab Prototype" in text
    # Ensure no external scripts are loaded (no http/https links)
    assert "https://" not in text
    assert "http://" not in text

def test_api_demo_returns_json():
    """Test that the API route returns structured JSON with all expected keys."""
    response = asyncio.run(get_demo())
    assert response.status_code == 200

    data = json.loads(response.body)
    assert "total_rounds" in data
    assert "dashboard_summary" in data
    assert "family1_results" in data
    assert "family2_results" in data
    assert "family3_results" in data
    assert "update_records" in data

    # Verify that the dashboard is present and tracks correctly
    assert data["total_rounds"] > 0
    assert "post_learning_recovery_observed" in data

def test_api_demo_is_deterministic():
    """Test that repeated calls to the API yield exactly the same byte-for-byte JSON."""
    response1 = asyncio.run(get_demo())
    response2 = asyncio.run(get_demo())

    # Ensure caching works and the output is identical
    assert response1.body == response2.body

def test_model_update_appears_in_api():
    """Verify that update records and truthful recovery data is exposed for learning panel."""
    response = asyncio.run(get_demo())
    data = json.loads(response.body)
    updates = data["update_records"]
    assert len(updates) > 0

    # Check that truthful recovery is exposed
    assert data["post_learning_recovery_observed"] is False

def test_evaluation_provenance():
    """Verify that clean evaluation data differentiates generalization provenance."""
    response = asyncio.run(get_demo())
    data = json.loads(response.body)

    clean_eval = data["clean_evaluation"]
    assert "ADAPTIVE_EVASION" in clean_eval
    assert "SYNTHETIC_IDENTITY" in clean_eval
    assert clean_eval["SYNTHETIC_IDENTITY"]["details"]["held_out_type"] == "legitimate_identities"
