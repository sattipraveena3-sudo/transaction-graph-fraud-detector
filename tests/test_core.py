import pandas as pd

from app.core import DetectionConfig, analyze_transactions, generate_transactions, graph_features, score_accounts
from app.store import Store


def test_graph_features_known_cycle():
    frame = pd.DataFrame(
        [
            ("T1", "A", "B", 10, 1, "X", False),
            ("T2", "B", "A", 12, 2, "X", True),
        ],
        columns=["id", "source", "target", "amount", "timestamp", "type", "is_fraud"],
    )
    graph, features = graph_features(frame)
    assert graph.number_of_nodes() == 2
    assert features.cycle_count.sum() >= 2


def test_scoring_has_risk_and_explanation():
    _, features = graph_features(generate_transactions(40, 300))
    scored = score_accounts(features)
    assert scored.risk_score.between(0, 1).all()
    assert scored.explanation.str.len().gt(5).all()
    assert set(scored.risk_level.unique()) <= {"low", "medium", "high", "critical"}


def test_synthetic_ring_is_detectable():
    result = analyze_transactions(generate_transactions(80, 600), DetectionConfig(contamination=0.1))
    scored = result["scored"].set_index("account")
    ring = [f"A{i}" for i in range(5)]
    assert scored.loc[ring, "risk_score"].mean() > scored.risk_score.median()
    assert result["metrics"]["precision"] >= 0
    assert result["metrics"]["recall"] >= 0


def test_store_persists_and_investigates(tmp_path):
    store = Store(str(tmp_path / "fraud.db"))
    frame = generate_transactions(20, 50)
    assert store.add_frame(frame) == 50
    assert store.transaction_count() == 50
    assert len(store.frame()) == 50
    item = store.add_investigation("A0", "reviewing", "manual review")
    assert item["account"] == "A0"
    assert store.investigations()[0]["status"] == "reviewing"
