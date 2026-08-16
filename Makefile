PYTHON ?= python3

.PHONY: install test run demo smoke docker-up docker-down neo4j

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

demo:
	curl -fsS -X POST 'http://localhost:8000/api/demo?accounts=100&transactions=700&seed=11'

smoke:
	curl -fsS http://localhost:8000/health
	curl -fsS http://localhost:8000/ready
	curl -fsS http://localhost:8000/api/summary
	curl -fsS http://localhost:8000/metrics

docker-up:
	docker compose up --build

docker-down:
	docker compose down

neo4j:
	docker compose --profile neo4j up --build
