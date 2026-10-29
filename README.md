# Codeforces Tag Prediction — ILLUIN Technology

Multi-label classification of **4,982 Codeforces programming problems** across **37 algorithmic tags**, with a particular focus on 8 priority tags.

## Approaches

**Supervised Machine Learning**
- TF-IDF (unigrams/bigrams) on problem statements and source code
- MiniLM sentence embeddings (384 dimensions)
- Early fusion with One-vs-Rest Logistic Regression
- Comparison of feature representations, regularization and classification thresholds

**LLM — Gemini**
- Zero-shot classification
- Fixed few-shot prompting (3 examples)
- Retrieval-based few-shot prompting using MiniLM cosine similarity

## Evaluation

Models are evaluated using **macro-F1 and micro-F1**, both on the 8 priority tags and across all 37 tags.

Experiments use a fixed train/validation/test split (64% / 16% / 20%).

## Key Findings

- Combining lexical and semantic features improves the supervised approach.
- Gemini zero-shot substantially outperforms the supervised baseline.
- Few-shot prompting does not consistently improve over zero-shot.
- Predictive performance must be balanced against inference latency and API cost.

## Repository

The notebook covers data exploration, feature engineering, model training, prompting experiments and evaluation.

*Technical assessment — ILLUIN Technology*
