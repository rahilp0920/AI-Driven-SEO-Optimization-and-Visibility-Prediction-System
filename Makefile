# One-command pipeline. See USER_MANUAL.md for the full walk-through.
#
# Quick start:   make install && make all
# Smoke test:    make test
# Live demo:     make dashboard

.PHONY: all install scrape queries serp features graph train train-fast dashboard test lint clean help

PYTHON ?= python3

help:
	@echo "Targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  all         scrape → queries → serp → features → graph → train (full pipeline)"
	@echo "  scrape      crawl all 5 dev-doc domains (~25 min)"
	@echo "  queries     derive queries from scraped <title> tags"
	@echo "  serp        fetch top-10 Google rankings via Brave Search (~25 min)"
	@echo "  features    build data/processed/features.csv"
	@echo "  graph       build link graph + merge graph features"
	@echo "  train       train all 4 models (LR + RF + XGBoost + MLP)"
	@echo "  train-fast  train XGBoost only (the cheapest end-to-end for tonight)"
	@echo "  dashboard   streamlit run src/dashboard/app.py"
	@echo "  test        pytest -v (synthetic-input smoke tests)"
	@echo "  clean       remove data/, models/, caches"

install:
	pip install -r requirements.txt

# Full data pipeline. Order matters; each step depends on the previous output.
all: scrape queries serp features graph train

scrape:
	$(PYTHON) -m src.scraping.doc_scraper --domain docs.python.org       --limit 300
	$(PYTHON) -m src.scraping.doc_scraper --domain developer.mozilla.org --limit 300
	$(PYTHON) -m src.scraping.doc_scraper --domain react.dev             --limit 250
	$(PYTHON) -m src.scraping.doc_scraper --domain nodejs.org            --limit 200
	$(PYTHON) -m src.scraping.doc_scraper --domain kubernetes.io         --limit 250

queries:
	$(PYTHON) -m src.scraping.serp_client build-queries

serp:
	$(PYTHON) -m src.scraping.serp_client fetch

features:
	$(PYTHON) -m src.features.build_features

graph:
	$(PYTHON) -m src.graph.build_graph
	$(PYTHON) -m src.graph.graph_features

train:
	$(PYTHON) -m src.models.baseline
	$(PYTHON) -m src.models.tree_models
	$(PYTHON) -m src.models.boosting
	$(PYTHON) -m src.models.neural

# Cheapest path to a working dashboard tonight.
train-fast:
	$(PYTHON) -m src.models.boosting

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest -v

clean:
	rm -rf data/raw data/interim data/processed models/*.joblib models/*.pt
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
