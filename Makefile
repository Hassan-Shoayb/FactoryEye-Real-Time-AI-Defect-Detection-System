# FactoryEye — Developer Automation & Operational Task Runner

.PHONY: help install run test benchmark gate docker-up docker-down clean

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║            FactoryEye AI Platform Developer Tasks            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo "  make install      Install Python production dependencies"
	@echo "  make run          Start FastAPI service with live reload (:8000)"
	@echo "  make test         Run automated API & regression test suite"
	@echo "  make benchmark    Run ONNX vs PyTorch latency & FPS benchmark"
	@echo "  make gate         Run Champion vs Challenger SLA model gate"
	@echo "  make batch        Run offline batch prediction on sample data"
	@echo "  make synthetic    Generate synthetic defect training samples"
	@echo "  make docker-up    Start API and MLflow with Docker Compose"
	@echo "  make docker-down  Stop Docker Compose services"
	@echo "  make clean        Remove cache files, databases, and logs"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	python3 tests/test_api.py

benchmark:
	python3 scripts/benchmark_export.py

gate:
	python3 scripts/model_gate.py --champion training/runs/train/weights/best.pt --challenger yolov8n.pt --max-latency 30.0

batch:
	python3 scripts/batch_predict.py --input-dir data/samples --output-dir data/batch_results

synthetic:
	python3 scripts/generate_synthetic.py --count 10 --output-dir data/synthetic_samples

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
