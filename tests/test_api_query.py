from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_examples():
    response = client.get("/api/v1/examples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 4
    assert "time_series_pembrolizumab" in [ex["id"] for ex in data]

def test_post_query_pembrolizumab_time_series():
    payload = {
        "query": "How has the number of trials for Pembrolizumab changed over time?",
        "drug_name": "Pembrolizumab",
        "max_trials_to_analyze": 30
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "visualization" in data
    assert "meta" in data
    assert "citations" in data
    assert data["visualization"]["type"] == "time_series"
    assert len(data["citations"]) > 0
    assert data["citations"][0]["url"].startswith("https://clinicaltrials.gov")

def test_post_query_self_comparison_collapses_to_single_entity():
    payload = {
        "query": "Compare trials",
        "drug_name": "Pembrolizumab",
        "drug_name_b": "Pembrolizumab",
        "max_trials_to_analyze": 30
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Must not be a grouped comparison with duplicate (phase, series) pairs --
    # collapses to a normal single-entity distribution instead.
    assert data["visualization"]["type"] != "grouped_bar_chart"
    if data["visualization"]["data"]:
        assert "series" not in data["visualization"]["data"][0]
    assert any("same entity" in note for note in data["meta"]["notes"])

def test_post_query_network_graph():
    payload = {
        "query": "Show a network of sponsors to drugs for Lung Cancer trials.",
        "condition": "Lung Cancer",
        "visualization_override": "network_graph",
        "max_trials_to_analyze": 30
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visualization"]["type"] == "network_graph"
    assert "nodes" in data["visualization"]
    assert "edges" in data["visualization"]
