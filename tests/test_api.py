from fastapi.testclient import TestClient

import app.main as main
from app.store import Store


def client_with_store(tmp_path):
    main.store = Store(str(tmp_path / "api.db"))
    return TestClient(main.app)


def test_health_and_ready(tmp_path):
    client = client_with_store(tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["status"] == "ready"


def test_demo_summary_graph_and_alerts(tmp_path):
    client = client_with_store(tmp_path)
    response = client.post("/api/demo?accounts=40&transactions=220&seed=11")
    assert response.status_code == 200
    summary = client.get("/api/summary").json()
    assert summary["transactions"] == 220
    assert summary["accounts"] > 0
    assert summary["flagged_accounts"] > 0
    graph = client.get("/api/graph").json()
    assert graph["nodes"] and graph["edges"]
    alerts = client.get("/api/alerts?limit=5").json()
    assert len(alerts) == 5
    account = alerts[0]["account"]
    detail = client.get(f"/api/accounts/{account}")
    assert detail.status_code == 200
    assert detail.json()["risk"]["account"] == account


def test_transaction_ingestion_and_investigation(tmp_path):
    client = client_with_store(tmp_path)
    payload = {
        "id": "TX-1",
        "source": "alice",
        "target": "bob",
        "amount": 8500,
        "timestamp": 10,
        "type": "TRANSFER",
        "is_fraud": False,
    }
    response = client.post("/api/transactions", json=payload)
    assert response.status_code == 201
    assert response.json()["transaction"]["id"] == "TX-1"
    inv = client.post("/api/investigations", json={"account": "alice", "status": "reviewing", "note": "check KYC"})
    assert inv.status_code == 201
    assert client.get("/api/investigations").json()[0]["account"] == "alice"


def test_metrics(tmp_path):
    client = client_with_store(tmp_path)
    client.post("/api/demo?accounts=20&transactions=100")
    text = client.get("/metrics").text
    assert "fraud_transactions 100" in text
    assert "fraud_flagged_accounts" in text
