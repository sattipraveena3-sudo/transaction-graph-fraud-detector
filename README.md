# Transaction Graph Fraud Detector

I built this project to combine structural graph evidence with an Isolation Forest account-risk model. The frictionless demo uses a deterministic PaySim-style synthetic mobile-money dataset with explicit ground-truth labels. NetworkX calculates PageRank, betweenness, and short-cycle participation; the model combines those signals with amount and frequency features. Neo4j is included for exploration and production-oriented graph storage.

```bash
docker compose up --build
```

Open `http://localhost:8000`. `GET /alerts` returns ranked accounts, `GET /account/{id}/explanation` returns signal-level evidence, and `/health` reports precision, recall, and ROC-AUC calculated at startup on the generated labeled data. Numbers are computed, not estimates; they may change if generator parameters change.

Isolation Forest is lightweight and explainable for tabular graph features. A GNN could learn richer neighborhoods but needs more data, more tuning, and careful leakage control. Current limitations include synthetic data, account-level labels, capped cycle enumeration, and in-memory NetworkX scoring. Next steps are a licensed PaySim download workflow, Neo4j bulk ingestion, temporal splits, PR-AUC, SHAP, GDS projections, and a graph neural baseline.

Suggested commits: `set up graph project`, `add transaction generator`, `build graph schema`, `extract centrality features`, `detect transaction cycles`, `add Isolation Forest`, `add explanations`, `add evaluation metrics`, `add FastAPI endpoints`, `build Cytoscape dashboard`, `add Neo4j Compose`, `write README`.

```bash
git init -b main
git add app/core.py && git commit -m "add graph features and anomaly scoring"
git add app/main.py app/static && git commit -m "add alert API and graph dashboard"
git add tests requirements.txt && git commit -m "add graph and evaluation tests"
git add Dockerfile docker-compose.yml && git commit -m "add Neo4j Docker setup"
git add README.md && git commit -m "document evaluation and limitations"
gh repo create transaction-graph-fraud-detector --public --source=. --remote=origin
git push -u origin main
```

MIT licensed. Research demonstration only; not a financial decision system.
