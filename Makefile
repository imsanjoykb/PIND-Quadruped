.PHONY: help install install-dev test lint format clean data train eval benchmark docker-build docker-run

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=pind --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 pind/ scripts/ tests/
	mypy pind/ --ignore-missing-imports

format:  ## Format code
	black pind/ scripts/ tests/
	isort pind/ scripts/ tests/

clean:  ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ .pytest_cache/ .coverage htmlcov/

data:  ## Generate training data
	python -m pind.data.generator \
		--num-trajectories 20 \
		--trajectory-length 500 \
		--output data/go1_dataset.pt

train:  ## Train PIND model (quick test)
	python scripts/train.py \
		--config pind/configs/default.yaml \
		--data-path data/go1_dataset.pt \
		--data-fraction 0.1 \
		--epochs 10

train-full:  ## Train PIND model (full dataset)
	python scripts/train.py \
		--config pind/configs/default.yaml \
		--data-path data/go1_dataset.pt

eval:  ## Evaluate trained model
	@echo "Please specify checkpoint path:"
	@echo "  make eval CHECKPOINT=path/to/model.pt"

benchmark:  ## Run full benchmark
	python run_experiments.py --config pind/configs/data_efficiency.yaml

docker-build:  ## Build Docker image
	docker-compose build

docker-run:  ## Run Docker container (GPU)
	docker-compose up -d pind
	docker exec -it pind-research bash

docker-run-cpu:  ## Run Docker container (CPU only)
	docker-compose up -d pind-cpu
	docker exec -it pind-research-cpu bash

docker-stop:  ## Stop Docker containers
	docker-compose down

tensorboard:  ## Launch TensorBoard
	tensorboard --logdir experiments/

docs:  ## Generate documentation
	cd docs && make html

check:  ## Run all checks (lint, test)
	make lint
	make test

setup-dev:  ## Setup development environment
	python -m venv venv
	. venv/bin/activate && make install-dev
	@echo "✓ Development environment ready!"
	@echo "  Activate with: source venv/bin/activate"

quick-test:  ## Quick test run (1% data, 3 epochs)
	python scripts/train.py --data-fraction 0.01 --epochs 3

# Evaluation with checkpoint
ifdef CHECKPOINT
eval:
	python scripts/evaluate.py \
		--checkpoint $(CHECKPOINT) \
		--data-path data/go1_dataset.pt \
		--horizon 50
endif
