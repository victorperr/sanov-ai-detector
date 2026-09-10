install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

demo:
	python -m ai_detector.cli demo --seed 7

# app:
#	streamlit run src/ai_detector/app.py
