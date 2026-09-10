"""Command-line entry points for reproducible demos."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .corpora import demo_references, tokenize
from .detector import SanovDetector
from .watermark import GreenlistWatermarker


def _demo_detector(seed: int) -> SanovDetector:
    alphabet, human, ai = demo_references()
    return SanovDetector(alphabet, human, ai, alpha=0.25)


def run_demo(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    alphabet, human, ai = demo_references()
    detector = SanovDetector(alphabet, human, ai, alpha=0.25)
    human_samples = [rng.choice(
        alphabet, size=120, p=human).tolist() for _ in range(40)]
    ai_samples = [rng.choice(alphabet, size=120, p=ai).tolist()
                  for _ in range(40)]
    calibration = detector.calibrate(
        human_samples[:20], ai_samples[:20], target_fpr=0.05)
    human_results = [detector.score(sample) for sample in human_samples[20:]]
    ai_results = [detector.score(sample) for sample in ai_samples[20:]]
    return {
        "threshold": calibration.threshold,
        "false_positive_rate": float(np.mean([result.label == "ai" for result in human_results])),
        "false_negative_rate": float(np.mean([result.label == "human" for result in ai_results])),
        "accuracy": calibration.accuracy,
        "mean_human_exponent": float(np.mean([result.sanov_exponent_human for result in human_results])),
        "mean_ai_exponent": float(np.mean([result.sanov_exponent_ai for result in ai_results])),
    }


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(
        description="AI text detection with types and Sanov's theorem")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo", help="run an offline calibrated experiment")
    demo.add_argument("--seed", type=int, default=7)
    analyze = subparsers.add_parser("analyze", help="score a text")
    analyze.add_argument("--text", required=True)
    analyze.add_argument("--json", action="store_true")
    watermark = subparsers.add_parser(
        "watermark", help="run green-list watermark analysis")
    watermark.add_argument("--length", type=int, default=200)
    watermark.add_argument("--seed", type=int, default=7)
    watermark.add_argument("--unmarked", action="store_true")
    subparsers.add_parser(
        "benchmark", help="run the optional TF-IDF supervised baseline")
    return command_parser


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "demo":
        print(json.dumps(run_demo(args.seed), indent=2))
    elif args.command == "analyze":
        detector = _demo_detector(0)
        tokens = [token for token in tokenize(
            args.text) if token in detector.estimator.alphabet]
        if not tokens:
            raise SystemExit("No demo-vocabulary tokens found in the text")
        result = detector.score(tokens)
        payload = result.__dict__
        print(json.dumps(payload, indent=2)
              if args.json else f"{result.label.upper()} score={result.score:.4f} KL(P)={result.kl_to_human:.4f} KL(Q)={result.kl_to_ai:.4f}")
    elif args.command == "watermark":
        detector = GreenlistWatermarker(256)
        result = detector.analyze(detector.generate_toy(
            args.length, seed=args.seed, watermarked=not args.unmarked))
        print(json.dumps(result.__dict__, indent=2))
    elif args.command == "benchmark":
        from .baselines import TfidfLogisticBaseline

        human = [
            "the data and research are reviewed by a human editor" for _ in range(20)]
        ai = [
            "the model generates fluent text for the research and data" for _ in range(20)]
        texts = human + ai
        labels = [0] * len(human) + [1] * len(ai)
        result = TfidfLogisticBaseline().fit(texts, labels).evaluate(texts, labels)
        print(json.dumps({"accuracy": result.accuracy,
              "roc_auc": result.roc_auc}, indent=2))


if __name__ == "__main__":
    main()
