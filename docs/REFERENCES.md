# REFERENCES

## Watermarking `watermark.py`

[1] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. (2023) A Watermark for Large Language Models arXiv:2301.10226 

"A watermark is a hidden pattern in text that is imperceptible to humans, while making the text algorithmically identifiable as synthetic."
They propose "a statistical test for detecting the watermark with interpretable p-values, and derive an information-theoretic framework for analyzing the sensitivity of the watermark."

`watermark.py` code creates a secret **“green list”** of tokens for each position, using a hash of the key and recent context. During generation, it prefers tokens from that green list, so the text contains more green tokens than normal. 

During detection, it checks whether the observed text has an unusually high green-token rate and computes a **z-score, p-value, and divergence-based signal** to decide if the watermark is likely present.

```
secret key → decides which tokens are “green”
generation → picks green tokens more often
detection → counts green tokens and tests if they are unusually frequent
```

## TfidfLogisticBaseline `baselines.py`

TF-IDF is a way of counting words. It assigns higher importance to words that are rare across the corpus but frequent within a document.

Libraries scikit-learn TfidfVectorizer gives a fixed-length document representation. **TF-IDF-weighted embeddings** compute the TF-IDF scores for the same tokenised words used to obtain the embeddings. Each word embedding is then multiplied by its corresponding TF-IDF weight, and the document representation is obtained as a weighted average of the word embeddings.

`baselines.py` vectorize the text with TF-IDF, implement a **logistic regression** (max iter=1000 allows convergence of the regression), traning of the logisic regression is made with `fit()` function. Finally, `predict()` apply this model on new texts.

```
score > 0.5 => label 1 
score < 0.5 => label 0
with 
accuracy_score: percentage of good predictions
roc_auc_score: quality of ranking between classes
```

## Hugging Face adapters for generation and token-probability types `hf.py`