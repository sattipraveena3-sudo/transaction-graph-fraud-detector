import os
import time
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core import DetectionConfig, analyze_transactions, generate_transactions
from app.store import Store


DB_PATH = os.getenv("DATABASE_PATH", "data/fraud.db")
API_KEY = os.getenv("INGESTION_API_KEY", "")
store = Store(DB_PATH)
config = DetectionConfig(
    contamination=float(os.getenv("CONTAMINATION", "0.08")),
    high_amount_threshold=float(os.getenv("HIGH_AMOUNT_THRESHOLD", "7500")),
    velocity_threshold=int(os.getenv("VELOCITY_THRESHOLD", "8")),
)


class TransactionIn(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    amount: float = Field(gt=0)
    timestamp: float | None = None
    type: str = "TRANSFER"
    is_fraud: bool = False


class InvestigationIn(BaseModel):
    account: str = Field(min_length=1, max_length=120)
    status: Literal["open", "reviewing", "confirmed_fraud", "cleared"] = "open"
    note: str = Field(default="", max_length=2000)


def require_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid API key")


def current_analysis():
    frame = store.frame()
    return analyze_transactions(frame, config)


app = FastAPI(
    title="Transaction Graph Fraud Detector",
    version="2.0.0",
    description="Graph-aware fraud detection and investigation API using NetworkX and Isolation Forest.",
)
static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(static / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "transaction-graph-fraud-detector", "transactions": store.transaction_count()}


@app.get("/ready")
def ready():
    try:
        store.transaction_count()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(503, f"database unavailable: {exc}")


@app.get("/api/summary")
def summary():
    result = current_analysis()
    scored = result["scored"]
    graph = result["graph"]
    return {
        "transactions": store.transaction_count(),
        "accounts": int(graph.number_of_nodes()),
        "relationships": int(graph.number_of_edges()),
        "flagged_accounts": int(scored.flagged.sum()) if not scored.empty else 0,
        "critical_accounts": int((scored.risk_level == "critical").sum()) if not scored.empty else 0,
        "metrics": result["metrics"],
    }


@app.get("/api/alerts")
def alerts(limit: int = Query(20, ge=1, le=200)):
    scored = current_analysis()["scored"]
    if scored.empty:
        return []
    return scored.head(limit).to_dict("records")


@app.get("/api/accounts/{account}")
def account(account: str):
    result = current_analysis()
    scored = result["scored"]
    row = scored[scored.account == account]
    if row.empty:
        raise HTTPException(404, "account not found")
    transactions = store.frame()
    related = transactions[(transactions.source == account) | (transactions.target == account)].tail(100)
    return {"risk": row.iloc[0].to_dict(), "transactions": related.to_dict("records")}


@app.get("/api/graph")
def graph_data(limit_edges: int = Query(500, ge=1, le=3000)):
    result = current_analysis()
    graph = result["graph"]
    scored = result["scored"].set_index("account") if not result["scored"].empty else None
    nodes = []
    for node in graph.nodes:
        risk = float(scored.loc[node, "risk_score"]) if scored is not None and node in scored.index else 0.0
        level = str(scored.loc[node, "risk_level"]) if scored is not None and node in scored.index else "low"
        nodes.append({"id": node, "risk_score": risk, "risk_level": level})
    edges = [
        {"source": a, "target": b, "amount": float(data.get("amount", 0)), "count": int(data.get("count", 1))}
        for a, b, data in list(graph.edges(data=True))[:limit_edges]
    ]
    return {"nodes": nodes, "edges": edges}


@app.post("/api/transactions", status_code=201, dependencies=[Depends(require_key)])
def ingest(payload: TransactionIn):
    item = payload.model_dump()
    item["timestamp"] = item["timestamp"] or time.time()
    store.add_transaction(item)
    result = current_analysis()["scored"]
    source = result[result.account == payload.source]
    return {"transaction": item, "source_risk": source.iloc[0].to_dict() if not source.empty else None}


@app.post("/api/demo", dependencies=[Depends(require_key)])
def demo(accounts: int = Query(100, ge=10, le=500), transactions: int = Query(700, ge=50, le=5000), seed: int = 11):
    store.clear_transactions()
    frame = generate_transactions(accounts=accounts, transactions=transactions, seed=seed)
    count = store.add_frame(frame)
    return {"created": count, "accounts": accounts, "seed": seed}


@app.post("/api/investigations", status_code=201)
def add_investigation(payload: InvestigationIn):
    return store.add_investigation(payload.account, payload.status, payload.note)


@app.get("/api/investigations")
def investigations(limit: int = Query(100, ge=1, le=500)):
    return store.investigations(limit)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    data = summary()
    return "\n".join([
        "# TYPE fraud_transactions gauge",
        f"fraud_transactions {data['transactions']}",
        "# TYPE fraud_flagged_accounts gauge",
        f"fraud_flagged_accounts {data['flagged_accounts']}",
        "# TYPE fraud_graph_accounts gauge",
        f"fraud_graph_accounts {data['accounts']}",
        "# TYPE fraud_graph_relationships gauge",
        f"fraud_graph_relationships {data['relationships']}",
        "",
    ])


# Backward-compatible endpoints.
@app.get("/alerts", include_in_schema=False)
def legacy_alerts(limit: int = 20):
    return alerts(limit)


@app.get("/account/{account}/explanation", include_in_schema=False)
def legacy_explanation(account: str):
    return account_detail(account)


def account_detail(account: str):
    return globals()["account"](account)


@app.get("/graph", include_in_schema=False)
def legacy_graph():
    return graph_data()
