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
