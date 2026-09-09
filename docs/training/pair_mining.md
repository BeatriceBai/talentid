# Contrastive training-pair construction

This milestone converts the frozen retrieval baseline into auditable training
supervision for learned fusion. It writes positive candidate-skill pairs and a
mixture of difficult and broad negatives without exposing test labels.

## Split boundary

- Train truth supplies training positives and negative exclusions.
- Validation truth is retained only for model selection.
- Test truth is excluded from every output and rejected by configuration.
- Negatives are mined only for training candidates.

## Pair types

| Table | Construction | Intended use |
| --- | --- | --- |
| `positive_pairs` | Synthetic truth for train and validation | Weighted positive loss and validation |
| `negative_pairs: hard_retrieval` | Highest-scoring frozen retrieval mistakes | Decision-boundary learning |
| `negative_pairs: random` | Seeded catalog samples excluding truth and hard negatives | Global separation and stability |
| `training_candidate_index` | Visible candidates and pair counts | Joins and data-contract checks |

Each positive receives a bounded weight combining 70% relevance and 30%
normalized proficiency. These weights preserve graded supervision without
turning lower-relevance truth into negatives.

The default configuration mines 30 hard and 10 random negatives per training
candidate. Hard negatives are false skills that the frozen baseline ranks
highly, making them more informative than arbitrary catalog skills.

## Determinism and leakage protection

Candidate and skill arrays are aligned through explicit `embedding_row`
indices. Random sampling uses a fixed seed. The builder validates key
uniqueness, catalog foreign keys, negative/truth disjointness, fixed negative
counts, and the absence of test candidates. A manifest stores all input and
output hashes.

## Commands

```bash
uv run ruff check .
uv run pytest -q
uv run python -m talentid.training.build_pairs
uv run python scripts/profile_training_pairs.py
```

Generated tables live under `data/training/` and must stay outside Git.

