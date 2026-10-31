PY ?= python

.PHONY: data features train eval app test lint all

data:
	$(PY) -m dengue.cli data

features:
	$(PY) -m dengue.cli features

train:
	$(PY) -m dengue.cli train

eval:
	$(PY) -m dengue.cli eval

app:
	uvicorn app.api.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check src tests app scripts

all:
	$(PY) -m dengue.cli all
