.PHONY: install dev test lint demo train-synthetic serve docker

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[all]"

test:
	pytest

lint:
	ruff check src tests

# End-to-end on synthetic light curves (no internet needed)
demo: train-synthetic serve

train-synthetic:
	exofinder make-synthetic --n 1500 --workers 4
	exofinder train --epochs 25

serve:
	exofinder serve --port 8000

docker:
	docker build -t exoplanet-finder .
