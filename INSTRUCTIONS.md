# Assignment — Recommender Systems (Surprise library)

You must work in teams of 2 or 3.

## Overview

Use the provided data set and the Surprise Python library to implement and evaluate collaborative filtering approaches. Submit a zip/rar containing:
- A report explaining results and discussion (include graphs).
- All source code used to produce results.

---

## Tasks

### 1. K-NN (user-based CF)
Using the user-based K-NN algorithm explained in class:

- a) With 25% of ratings removed (missing), find the value of K that minimizes MAE.
- b) Sparsity problem: with 75% of ratings removed, find the value of K that minimizes MAE.

### 2. Mitigation of sparsity: SVD (Funk variant)
- Show how SVD (Funk variant) can provide a better MAE than user-based K-NN on the provided data set (compare MAE values, especially under high sparsity).

### 3. Top-N recommendations
For both 25% and 75% missing ratings, evaluate Top-N recommendation quality:

- Use user-based K-NN (with the best K found) and SVD.
- For N in {10, 20, ..., 100} compute:
  - Precision@N
  - Recall@N
  - F1@N
- Define relevant items for a user as those with rating 4 or 5 in the original data set.
- Explain why the results make sense.

### 4. (HCID master students only)
Repeat task 3 for N = 10 and 25% missing ratings using a *different* data set. Explain and justify differences in precision, recall and F1 between data sets.

---

## Requirements & Notes

- Use the Surprise library for implementations of the algorithms.
- Include plots/graphs in the report to illustrate results (recommended).
- Discuss why observed behavior is reasonable (effects of sparsity, choice of K, differences between algorithms and data sets).

--- 

Deliverables: zipped report + source code.