# Tower embeddings

This milestone freezes a compact Sentence Transformer and creates aligned,
versioned vectors for the five candidate evidence towers and the canonical
skill tower.

## Model and artifacts

The baseline uses `sentence-transformers/all-MiniLM-L6-v2` at a pinned model
revision. It produces 384-dimensional vectors. The repository lockfile pins the
Python implementation; `embedding_manifest.json` records the model revision,
configuration, feature hashes, output checksums, shapes, and dtypes.

| Artifact | Shape for v1 | Purpose |
| --- | ---: | --- |
| `candidate_tower_embeddings.npy` | 50,000 × 384 | One row per candidate and source |
| `candidate_tower_embedding_index.parquet` | 50,000 rows | Stable row-to-candidate/source mapping |
| `skill_embeddings.npy` | 8,821 × 384 | Canonical skill vectors |
| `skill_embedding_index.parquet` | 8,821 rows | Stable row-to-skill mapping |
| `candidate_embeddings.npy` | 10,000 × 384 | Reliability-weighted baseline candidate vectors |
| `candidate_embedding_index.parquet` | 10,000 rows | Candidate and split mapping |

The exact row counts come from the feature snapshot. NumPy arrays avoid costly
object-array columns in Parquet; the adjacent index tables make every row
explicit and joinable.

## Missingness and fusion

Missing towers are retained in the index and represented by all-zero vectors.
Available tower and skill vectors are L2-normalized. The baseline candidate
vector is:

\[
z_c = \operatorname{normalize}\left(
  \frac{\sum_t r_{ct} e_{ct}}{\sum_t r_{ct}}
\right),
\]

where `r` is the feature-layer reliability score and `e` is the normalized
tower embedding. This is a transparent baseline, not the final learned fusion
model.

## Leakage and evaluation boundary

The encoder reads only `tower_text` and `skill_text`. Candidate truth labels,
canonical IDs hidden by the feature layer, occupation codes, and split labels
are never passed to it. Splits remain in index metadata for later retrieval
evaluation.

The first model download requires internet access. CI uses a deterministic fake
encoder, so it validates alignment, normalization, missingness, and fusion math
without downloading model weights.

## Commands

```bash
uv add "sentence-transformers>=5,<7"
uv run ruff check .
uv run pytest -q
uv run python -m talentid.embeddings.build_embeddings
uv run python scripts/profile_embeddings.py
```

Embedding outputs live under `data/embeddings/` and must stay out of Git.

