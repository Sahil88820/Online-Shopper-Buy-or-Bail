# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER_COMPOSE := docker-compose

# Python environment setup
.PHONY: setup
setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-streamlit.txt

# Run tests
.PHONY: test
test:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing

# Run API server
.PHONY: run-api
run-api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit dashboard
.PHONY: run-dashboard
run-dashboard:
	streamlit run src/streamlit_app.py

# Docker commands
.PHONY: docker-build
docker-build:
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml build

.PHONY: docker-up
docker-up:
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml up -d

.PHONY: docker-down
docker-down:
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml down

# Clean up
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -delete
	find . -type f -name "*.pyc" -delete

# Training pipeline
.PHONY: train
train:
	$(PYTHON) src/data_preprocessing.py
	$(PYTHON) src/persona_clustering.py
	$(PYTHON) src/purchase_prediction.py
	$(PYTHON) src/incentive_recommendation.py

# Format code
.PHONY: format
format:
	black src/ tests/
	isort src/ tests/

# Lint code
.PHONY: lint
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

# All-in-one development setup
.PHONY: dev-setup
dev-setup: clean setup format lint test