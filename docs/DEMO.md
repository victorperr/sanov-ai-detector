# Demo guide

This project already includes ready-to-run demos for the main ideas:

- statistical detection with empirical types and Sanov exponents
- one-text analysis using built-in reference distributions
- a toy green-list watermark detector
- an optional TF-IDF baseline comparison

## 1. Install the project

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

If you also want the optional ML baseline:

```bash
pip install -e ".[ml]"
```

---

## 2. Run the main offline demo

This is the easiest and most representative demo. It generates synthetic human and AI samples, calibrates a threshold, and reports summary metrics.

```bash
python -m ai_detector.cli demo --seed 7
```

Typical output:

```json
{
  "threshold": 0.123,
  "false_positive_rate": 0.05,
  "false_negative_rate": 0.08,
  "accuracy": 0.93,
  "mean_human_exponent": 0.42,
  "mean_ai_exponent": 0.81
}
```

This demonstrates:

- comparing a sample against human and AI reference distributions
- calibrating a decision threshold
- measuring type I / type II errors
- reporting Sanov-style exponents

---

## 3. Analyze one custom text

You can score a text using the built-in demo references.

```bash
python -m ai_detector.cli analyze --text "Language models generate fluent sequences and humans edit them."
```

With JSON output:

```bash
python -m ai_detector.cli analyze --text "Language models generate fluent sequences and humans edit them." --json
```

This prints a classification and related stats such as:

```text
AI score=0.7421 KL(P)=0.4123 KL(Q)=0.9801
```

---

## 4. Run the watermark toy demo

This illustrates the green-list watermark concept. It generates a toy token sequence with a secret key and checks whether the green-token rate is unusually high.

```bash
python -m ai_detector.cli watermark --length 200 --seed 7
```

To generate an unmarked sequence for comparison:

```bash
python -m ai_detector.cli watermark --length 200 --seed 7 --unmarked
```

This reports:

- green hits
- green rate
- z-score
- p-value
- Sanov exponent
- whether the sequence is flagged as watermarked

---

## 5. Run the TF-IDF baseline (optional)

If scikit-learn is installed, you can compare the statistical detector to a simple supervised baseline.

```bash
python -m ai_detector.cli benchmark
```

This trains a TF-IDF + logistic regression model on a small synthetic dataset and prints accuracy and ROC-AUC.

---

## 6. Notes

- The watermark implementation is intentionally toy-like, to illustrate the mechanism without requiring a full production generator setup.
- Real research experiments should use held-out corpora, fixed tokenization, and proper calibration.

