.PHONY: test build lint format typecheck qa check  publish clean

test:
	@uv run pytest tests/ -v

lint:
	@uv run ruff check --fix src/ tests/

format:
	@uv run ruff format src/ tests/

typecheck:
	@uv run mypy src/

qa: test lint typecheck format

build: clean
	@uv build
	@uv run twine check dist/*

check:
	@uv run twine check dist/*

publish: check
	@uv run twine upload dist/*

clean:
	@rm -rf dist/ build/ src/*.egg-info
