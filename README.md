# AI Text Detector: Types, Sanov, and Watermarks

A project for **detecting AI-generated text** with empirical types and large deviations.

## Research question

Given a text represented as a sequence of discrete events (word n-grams, token IDs, or token probability bins), test:

- **$H_0$ (human):** the empirical type is sampled around a human reference distribution $`P`$.
- **$H_1$ (AI):** the empirical type is sampled around an LLM reference distribution $`Q`$.

> [!NOTE]
> $`R_n`$ = type of the sequence (also called empirical distribution)

> For 2 distributions $p$ and $q$, $`D(p || q)`$ is called relative entropy (sometimes called **Kullback-Leibler divergence**). It measures of how much an approximating probability distribution $q$ is different from a true probability distribution $p$.

We estimate $`R_n`$ from the sequence and compute the difference of relative entropy $`D(R_n || P) - D(R_n || Q)`$ (called score) and compare it to a threshold.

For a rejection region `E`, **Sanov's theorem** gives
$$P(R_n \in E) ~= \exp(-n \times inf_{R \in E} D(R || P))$$
The implementation reports this exponent as an interpretable upper-tail rarity estimate.

## Quick start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
python -m ai_detector.cli demo --seed 7
python -m ai_detector.cli analyze --text "Language models generate fluent sequences and humans edit them." --json
```

The demo is fully offline and uses built-in synthetic reference distributions. Optional Hugging Face integration is available with `pip install -e ".[ml]"`.

## Commands

- `demo`: run a calibrated human-vs-AI experiment, report error rates and Sanov exponents.
- `analyze`: score one text using a reference model saved as JSON or the built-in demo references.
- `watermark`: generate or analyze a green-list watermark toy sequence and compare its z-score with the large-deviation exponent.
- `benchmark`: compare the statistical detector with a TF-IDF logistic-regression baseline when scikit-learn is installed.

## Architecture

```text
src/ai_detector/
  types.py       Empirical types, smoothing, KL/JS divergence
  detector.py    Sanov detector, calibration, operating-point metrics
  watermark.py   Kirchenbauer-style green-list watermark abstraction
  corpora.py     Offline references and n-gram extraction
  baselines.py   Optional TF-IDF neural/classical baseline adapters
  hf.py          Optional Hugging Face generation and token probabilities (in development)
  cli.py         Reproducible command line interface
  app.py         Streamlit research dashboard (in development)
```


