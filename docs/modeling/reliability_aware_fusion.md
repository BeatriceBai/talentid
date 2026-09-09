# Trainable reliability-aware fusion

This milestone learns a shared 128-dimensional candidate-skill space from the
frozen 384-dimensional tower embeddings. It starts from the transparent
reliability-weighted baseline, then learns how source quality and hard mistakes
should change the representation.

## Architecture

Each candidate has five frozen source vectors. A shared projection preserves a
common geometry, while zero-initialized source-specific residual adapters learn
source corrections. The gate uses six structured features per source:

1. reliability score;
2. exponential freshness;
3. log-volume score;
4. stale flag;
5. high-volume flag;
6. availability flag.

The gate starts at the supplied reliability weights. A small MLP learns an
additive log-weight correction, unavailable towers are masked to zero, and the
remaining weights sum to one. A separate skill projection maps frozen skill
vectors into the same normalized 128-dimensional space.

## Objective

For every train positive, the model contrasts its skill against four mined
hard negatives and four random negatives. Cross-entropy is weighted by the
positive's relevance/proficiency score. AdamW, gradient clipping, deterministic
seeding, validation NDCG@30 checkpointing, and early stopping are configured in
`configs/fusion_model.toml`.

The encoder text model stays frozen. This keeps laptop training practical and
isolates whether learned fusion and projection improve retrieval.

## Leakage boundary

- Only train pairs update model parameters.
- Validation labels choose the checkpoint.
- Test labels are absent from the pair artifacts and rejected if encountered.
- Test profile features may be transformed at export, but no test outcome is
  inspected.

## Outputs

All generated outputs live in `data/models/fusion/v1/`:

- `best_model.pt` — state dictionary and architecture metadata;
- `learned_candidate_embeddings.npy` — 10,000 × 128;
- `learned_skill_embeddings.npy` — 8,821 × 128;
- candidate and skill row-index Parquet files;
- `gate_weights.parquet` — candidate × source learned weights;
- `training_history.parquet` — loss and validation metrics by epoch;
- `fusion_manifest.json` — configuration, hashes, best epoch, and metrics.

## Commands

```bash
uv add --optional training "torch>=2.3,<3"
uv run --extra training ruff check .
uv run --extra training pytest -q
uv run --extra training python -m talentid.training.train_fusion
uv run python scripts/profile_fusion_model.py
```

Apple Silicon uses PyTorch's MPS backend when available; otherwise the trainer
falls back to CPU. The test split stays sealed until all modeling decisions are
frozen.

