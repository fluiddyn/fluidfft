.PHONY: clean build install test

clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

build:
	python -m build --no-isolation

install:
	pip install --no-build-isolation -e .

install_dev:
	pip install --no-build-isolation -e .[dev]

test:
	pytest tests/
