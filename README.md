# Transaction Graph Fraud Detector

A complete, runnable graph-based fraud detection and investigation platform. It ingests transaction telemetry, persists it locally, builds a directed payment network, extracts graph/behavioral features, scores account risk with an Isolation Forest + rule signals, exposes operational APIs, and presents an interactive analyst dashboard.

`transactions → SQLite → NetworkX graph → graph/behavior features → anomaly scoring → alerts + investigations + dashboard`

## Features

- Real transaction ingestion over REST
- Persistent SQLite transaction and investigation storage
- Directed transaction graph construction
- PageRank, betweenness, in/out-degree and short-cycle features
- Amount, transfer velocity and counterparty diversity signals
- Isolation Forest anomaly detection
- Composite 0–1 account risk score
- Low / medium / high / critical risk tiers
- Human-readable account explanations
- Ranked fraud alert feed
- Account-level transaction drilldowns
- Investigation case records with analyst status and notes
- Interactive Cytoscape transaction network
- Synthetic fraud-ring generator for demos and testing
- Liveness `/health` and readiness `/ready` probes
- Prometheus-compatible `/metrics`
- Optional API-key protection for transaction/demo write endpoints
- Docker and Docker Compose startup
- Optional Neo4j exploration profile
- Unit + API integration tests
- GitHub Actions compile, test, container-build and live smoke validation
- Interactive OpenAPI documentation at `/docs`

## Fastest start

Requirements: Docker and Docker Compose.

```bash
git clone https://github.com/sattipraveena3-sudo/transaction-graph-fraud-detector.git
cd transaction-graph-fraud-detector
docker compose up --build
```

Open `http://localhost:8000` and click **Generate demo network**.

Useful endpoints:

- Dashboard: `http://localhost:8000/`
- Swagger / OpenAPI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`
- Metrics: `http://localhost:8000/metrics`

## Local Python development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
make run
```

Then in another terminal:

```bash
make demo
make smoke
```

## Ingest a transaction

```bash
curl -X POST http://localhost:8000/api/transactions \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "TX-1001",
    "source": "acct-103",
    "target": "acct-772",
    "amount": 9200,
    "timestamp": 1786857600,
    "type": "TRANSFER",
    "is_fraud": false
  }'
```

The service stores the transaction and recomputes account risk from the current transaction graph.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness and transaction count |
| GET | `/ready` | Storage readiness |
| GET | `/metrics` | Prometheus-style service metrics |
| GET | `/api/summary` | Graph and risk KPIs |
| GET | `/api/alerts` | Ranked account-risk alerts |
| GET | `/api/accounts/{account}` | Account risk + related transactions |
| GET | `/api/graph` | Network nodes/edges for visualization |
| POST | `/api/transactions` | Ingest a transaction |
| POST | `/api/demo` | Reset and generate a synthetic fraud network |
| POST | `/api/investigations` | Create an analyst case record |
| GET | `/api/investigations` | List analyst case records |
| GET | `/docs` | Interactive API docs |

## Detection model

For every account, the service derives behavioral and graph features including:

- total outgoing value
- transaction count
- mean and maximum transaction amount
- unique incoming/outgoing counterparties
- PageRank
- betweenness centrality
- short payment-cycle participation
- graph in-degree and out-degree

An Isolation Forest produces the unsupervised anomaly signal. The service combines that with amount, velocity and cycle signals into a normalized account risk score. This keeps the model lightweight enough for a one-command portfolio project while still demonstrating why graph context adds information that flat transaction rules miss.

The synthetic demo deliberately creates a small high-value circular transfer ring so graph-cycle and risk-alert behavior can be exercised reliably.

## Investigation workflow

1. Generate or ingest transactions.
2. Review ranked accounts in the dashboard.
3. Select an account to inspect graph evidence and related transfers.
4. Create an investigation record via `POST /api/investigations` with status `open`, `reviewing`, `confirmed_fraud`, or `cleared`.
5. Integrate `/metrics` and API endpoints into external monitoring if desired.

## Configuration

Copy the example file if you want custom settings:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---:|---|
| `DATABASE_PATH` | `data/fraud.db` | SQLite database file |
| `CONTAMINATION` | `0.08` | Isolation Forest expected anomaly fraction |
| `HIGH_AMOUNT_THRESHOLD` | `7500` | High-value evidence threshold |
| `VELOCITY_THRESHOLD` | `8` | High outgoing transaction-count threshold |
| `INGESTION_API_KEY` | empty | Optional write-endpoint API key |
| `NEO4J_PASSWORD` | `portfolio-password` | Optional Neo4j profile password |

When `INGESTION_API_KEY` is set, include it as `X-API-Key` for `/api/transactions` and `/api/demo`.

## Optional Neo4j

The application itself does not require Neo4j, which keeps default startup fast and fully self-contained. For graph exploration or future Graph Data Science work:

```bash
docker compose --profile neo4j up --build
```

Neo4j Browser will be available at `http://localhost:7474`.

## Tests and CI

```bash
pytest -q
```

GitHub Actions validates the repository by running:

1. dependency installation
2. Python source compilation
3. full pytest suite
4. Docker image build
5. actual container startup
6. live readiness, health, demo generation, summary, alerts, graph and metrics requests

This means CI checks the same runnable service a user starts locally rather than only testing isolated functions.

## Architecture

- **FastAPI** — ingestion, investigation, risk and graph APIs
- **SQLite** — zero-setup persistent transaction/case storage
- **NetworkX** — directed graph analytics and structural features
- **scikit-learn Isolation Forest** — unsupervised anomaly signal
- **Pandas / NumPy** — feature aggregation and data processing
- **Cytoscape.js** — interactive graph visualization
- **Docker / Compose** — repeatable startup
- **GitHub Actions** — automated end-to-end verification
- **Neo4j (optional)** — exploration / future production graph storage

## Repository commands

```bash
make install
make test
make run
make demo
make smoke
make docker-up
make docker-down
make neo4j
```

## Production-scale extensions

The project is complete and runnable as-is. Larger deployments could add Kafka streaming ingestion, PostgreSQL or Neo4j persistence, feature-store snapshots, temporal windows, supervised fraud labels, PR-AUC monitoring, SHAP explanations, Graph Data Science pipelines, RBAC/SSO, alert delivery to Slack/PagerDuty, and a GNN baseline.

This repository is an engineering/research demonstration and should not be used as the sole basis for real financial decisions.

MIT licensed.
