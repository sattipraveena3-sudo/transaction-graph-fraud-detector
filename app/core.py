import math
from dataclasses import dataclass, asdict
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "total_amount",
    "transaction_count",
    "mean_amount",
    "max_amount",
    "unique_targets",
    "unique_sources",
    "pagerank",
    "betweenness",
    "cycle_count",
    "in_degree",
    "out_degree",
]


@dataclass
class DetectionConfig:
    contamination: float = 0.08
    high_amount_threshold: float = 7500.0
    velocity_threshold: int = 8
    cycle_weight: float = 0.16
    amount_weight: float = 0.18
    velocity_weight: float = 0.14
    model_weight: float = 0.52


def generate_transactions(accounts: int = 120, transactions: int = 900, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(transactions):
        source = f"A{rng.integers(accounts)}"
        target = f"A{rng.integers(accounts)}"
        while target == source:
            target = f"A{rng.integers(accounts)}"
        amount = float(rng.lognormal(4.0, 1.0))
        fraud = False
        tx_type = "TRANSFER"
        if i < max(12, transactions // 20):
            source = f"A{i % 5}"
            target = f"A{(i + 1) % 5}"
            amount = float(rng.uniform(8000, 15000))
            fraud = True
        elif i % 97 == 0:
            amount = float(rng.uniform(9000, 18000))
        rows.append((f"T{i}", source, target, amount, i, tx_type, fraud))
    return pd.DataFrame(rows, columns=["id", "source", "target", "amount", "timestamp", "type", "is_fraud"])


def graph_features(frame: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame]:
    if frame.empty:
        return nx.DiGraph(), pd.DataFrame(columns=["account", *FEATURE_COLUMNS, "fraud_label"])

    graph = nx.DiGraph()
    for row in frame.itertuples():
        current = graph.get_edge_data(row.source, row.target, {})
        graph.add_edge(
            row.source,
            row.target,
            amount=float(current.get("amount", 0.0)) + float(row.amount),
            count=int(current.get("count", 0)) + 1,
        )

    pagerank = nx.pagerank(graph, weight="amount") if graph.number_of_nodes() else {}
    betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_nodes() else {}
    cycles = {node: 0 for node in graph.nodes}
    for cycle in list(nx.simple_cycles(graph, length_bound=5))[:2000]:
        for node in cycle:
            cycles[node] += 1

    outgoing = frame.groupby("source").agg(
        total_amount=("amount", "sum"),
        transaction_count=("id", "count"),
        mean_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        unique_targets=("target", "nunique"),
        fraud_label=("is_fraud", "max"),
    )
    incoming = frame.groupby("target").agg(unique_sources=("source", "nunique"))
    accounts = sorted(set(frame.source) | set(frame.target))
    rows = []
    for account in accounts:
        out = outgoing.loc[account] if account in outgoing.index else None
        rows.append(
            {
                "account": account,
                "total_amount": float(out.total_amount) if out is not None else 0.0,
                "transaction_count": int(out.transaction_count) if out is not None else 0,
                "mean_amount": float(out.mean_amount) if out is not None else 0.0,
                "max_amount": float(out.max_amount) if out is not None else 0.0,
                "unique_targets": int(out.unique_targets) if out is not None else 0,
                "unique_sources": int(incoming.loc[account].unique_sources) if account in incoming.index else 0,
                "pagerank": float(pagerank.get(account, 0.0)),
                "betweenness": float(betweenness.get(account, 0.0)),
                "cycle_count": int(cycles.get(account, 0)),
                "in_degree": int(graph.in_degree(account)) if account in graph else 0,
                "out_degree": int(graph.out_degree(account)) if account in graph else 0,
                "fraud_label": bool(out.fraud_label) if out is not None else False,
            }
        )
    return graph, pd.DataFrame(rows)


def _minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def score_accounts(features: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    config = config or DetectionConfig()
    if features.empty:
        return features.assign(score=pd.Series(dtype=float), flagged=pd.Series(dtype=bool), explanation=pd.Series(dtype=str))

    x = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0)
    model = IsolationForest(
        n_estimators=200,
        contamination=min(max(config.contamination, 0.01), 0.49),
        random_state=11,
    ).fit(x)
    model_score = _minmax(pd.Series(-model.score_samples(x), index=features.index))
    amount_signal = _minmax(features["max_amount"].astype(float))
    velocity_signal = _minmax(features["transaction_count"].astype(float))
    cycle_signal = _minmax(features["cycle_count"].astype(float))
    score = (
        config.model_weight * model_score
        + config.amount_weight * amount_signal
        + config.velocity_weight * velocity_signal
        + config.cycle_weight * cycle_signal
    )
    result = features.copy()
    result["model_score"] = model_score
    result["risk_score"] = score.clip(0, 1)
    result["flagged"] = (model.predict(x) == -1) | (result["risk_score"] >= 0.72)
    result["risk_level"] = pd.cut(
        result["risk_score"], bins=[-0.01, 0.35, 0.6, 0.8, 1.01], labels=["low", "medium", "high", "critical"]
    ).astype(str)

    def explain(row) -> str:
        reasons = []
        if row.max_amount >= config.high_amount_threshold:
            reasons.append("unusually high transaction amount")
        if row.transaction_count >= config.velocity_threshold:
            reasons.append("high transfer velocity")
        if row.cycle_count > 0:
            reasons.append(f"participates in {int(row.cycle_count)} short payment cycles")
        if row.pagerank > features.pagerank.quantile(0.9):
            reasons.append("high graph centrality")
        if not reasons:
            reasons.append("multivariate anomaly pattern")
        return "; ".join(reasons)

    result["explanation"] = result.apply(explain, axis=1)
    return result.sort_values("risk_score", ascending=False).reset_index(drop=True)


def evaluation_metrics(scored: pd.DataFrame) -> dict:
    if scored.empty or "fraud_label" not in scored:
        return {"precision": 0.0, "recall": 0.0, "flagged": 0}
    truth = scored.fraud_label.astype(bool)
    pred = scored.flagged.astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "flagged": int(pred.sum())}


def analyze_transactions(frame: pd.DataFrame, config: DetectionConfig | None = None) -> dict:
    graph, features = graph_features(frame)
    scored = score_accounts(features, config)
    return {"graph": graph, "features": features, "scored": scored, "metrics": evaluation_metrics(scored)}
