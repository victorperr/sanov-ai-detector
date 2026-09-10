"""Optional Hugging Face adapters for generation and token-probability types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeneratedText:
    text: str
    token_ids: list[int]
    token_probabilities: list[float]


def load_causal_lm(model_name: str = "distilgpt2") -> tuple[Any, Any]:
    """Load a causal LM lazily; model downloads are never required by core tests."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Install the 'ml' extra to use Hugging Face adapters") from error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def generate_text(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int = 80, seed: int = 0) -> GeneratedText:
    import torch

    torch.manual_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**encoded, max_new_tokens=max_new_tokens,
                            return_dict_in_generate=True, output_scores=True, do_sample=True, top_p=0.95)
    sequence = output.sequences[0]
    new_token_ids = sequence[encoded["input_ids"].shape[1]:].tolist()
    probabilities: list[float] = []
    for step, token_id in enumerate(new_token_ids):
        distribution = torch.softmax(output.scores[step][0], dim=-1)
        probabilities.append(float(distribution[token_id]))
    return GeneratedText(tokenizer.decode(sequence, skip_special_tokens=True), new_token_ids, probabilities)


def probability_type(probabilities: list[float], bins: int = 10) -> np.ndarray:
    """Convert token probabilities into an empirical type over fixed bins."""
    if not probabilities or bins < 2:
        raise ValueError(
            "probabilities must be non-empty and bins must be >= 2")
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(np.clip(probabilities, 0, 1), bins=edges)
    return (counts / counts.sum()).astype(float)
