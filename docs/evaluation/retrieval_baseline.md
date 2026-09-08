# Candidate-to-skill retrieval baseline

This milestone tests whether the frozen tower embeddings recover each
candidate's synthetic ground-truth skills. It deliberately evaluates the
validation split by default; the test split remains sealed until model and
fusion choices are finalized.

## Compared strategies

| Strategy | Candidate vector |
| --- | --- |
| `reliability_weighted` | Existing reliability-weighted five-tower vector |
| `uniform_mean` | Equal mean of all available tower vectors |

Both strategies use the same normalized skill vectors. Retrieval is exact
cosine similarity, implemented as batched matrix multiplication. This gives an
unambiguous baseline before FAISS approximation is introduced.

## Metrics

Metrics are reported at 10, 30, and 100:

- **Precision@K**: relevant retrieved skills divided by K;
- **Recall@K**: retrieved truth skills divided by all candidate truth skills;
- **NDCG@K**: ranking quality using graded synthetic `relevance_score`;
- **HitRate@K**: fraction of candidates with at least one relevant result;
- **MRR@K**: reciprocal rank of the first relevant skill;
- **Catalog coverage@K**: fraction of canonical skills retrieved at least once.

The profiler also slices metrics by the number of available evidence towers:
0–2, 3–4, or all 5. This exposes whether reliability-aware fusion is helping
sparse profiles rather than only improving an overall average.
It prints the theoretical random-ranking Recall@30 as a basic sanity reference.

## Outputs

| Artifact | Grain |
| --- | --- |
| `retrieval_predictions.parquet` | Candidate × strategy × rank |
| `retrieval_candidate_metrics.parquet` | Candidate × strategy × K |
| `retrieval_metrics.parquet` | Strategy × split × K |
| `retrieval_metrics_by_coverage.parquet` | Strategy × split × K × coverage bucket |
| `retrieval_manifest.json` | Configuration, input hashes, and output hashes |

All outputs live under `data/evaluation/` and remain outside Git.

## Commands

```bash
uv run ruff check .
uv run pytest -q
uv run python -m talentid.evaluation.evaluate_retrieval
uv run python scripts/profile_retrieval.py
```

Only after model decisions are frozen should `configs/retrieval.toml` be changed
to `splits = ["test"]` for the final unbiased report.
