.PHONY: help setup data db train anomaly drift explain summary reports all dashboard test clean

PYTHON := venv/bin/python3
PIP := venv/bin/pip
STREAMLIT := venv/bin/streamlit

help:
	@echo "GearGuard commands:"
	@echo "  make setup      Create venv and install dependencies"
	@echo "  make data       Generate synthetic dataset"
	@echo "  make db         Load dataset into SQLite"
	@echo "  make train      Train defect-risk model"
	@echo "  make anomaly    Score anomaly detection outputs"
	@echo "  make drift      Generate drift report"
	@echo "  make explain    Generate feature attribution and item explanations"
	@echo "  make summary    Generate quality triage summary"
	@echo "  make reports    Run anomaly, drift, explain, and summary"
	@echo "  make all        Run full non-dashboard pipeline"
	@echo "  make dashboard  Launch Streamlit dashboard"
	@echo "  make test       Run tests"
	@echo "  make clean      Remove generated model/report/cache files"

setup:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) src/data_generator.py

db:
	$(PYTHON) src/db.py

train:
	$(PYTHON) src/train.py

anomaly:
	$(PYTHON) src/anomaly.py

drift:
	$(PYTHON) src/drift.py

explain:
	$(PYTHON) src/explain.py

summary:
	$(PYTHON) src/summary.py

reports: anomaly drift explain summary

all: data db train reports

dashboard:
	$(STREAMLIT) run app/streamlit_app.py

test:
	$(PYTHON) -m pytest

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf app/__pycache__
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -f models/*.joblib
	rm -f reports/*.json
	rm -f reports/*.csv
	rm -f reports/*.md
	rm -f data/processed/equipment_quality_scored.csv
	rm -f data/processed/gear_guard.sqlite
