# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Run linting

```bash
ruff check .
```

## Suggested workflow

1. Fork the repository.
2. Create a feature branch.
3. Make the smallest relevant change.
4. Add or update tests for the behavior you change.
5. Run the relevant checks.
6. Open a pull request with a clear summary.

## Coding expectations

- Keep changes focused and readable.
- Prefer small, testable functions.
- Document public behavior when it matters.
- Maintain compatibility with the project’s supported Python versions.
