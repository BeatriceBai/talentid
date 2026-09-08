# Feature pipeline

The feature layer converts normalized candidate evidence into model-ready text
and reliability metadata. It produces a fixed candidate-by-source grid so the
embedding and fusion layers receive the same schema even when towers are
missing.

## Artifacts

| Artifact | Grain | Model use |
| --- | --- | --- |
| `candidate_tower_features.parquet` | Candidate × source | Tower encoders and reliability-aware fusion |
| `skill_features.parquet` | Canonical skill | Skill-tower encoder and retrieval index |
| `candidate_index.parquet` | Candidate | Joins, dataset splits, and evaluation slices only |
| `feature_manifest.json` | Feature snapshot | Lineage, configuration, hashes, and row counts |

The five candidate sources are resume, projects, job history, courses, and
self-reported skills. Missing sources remain present as rows with empty text,
zero reliability, and `is_available = false`.

## Leakage boundary

Candidate model text is built only from observable profile fields. The pipeline
explicitly excludes:

- `candidate_skill_truth`;
- `self_reported_skills.canonical_skill_id`;
- `self_reported_skills.oracle_is_true_skill`;
- `courses.skill_id`;
- job-history and primary O*NET occupation codes.

`candidate_index.primary_onet_soc_code` exists only for joins and evaluation
slices. It must not be passed to the candidate encoder or reranker.

## Reliability metadata

Every tower row includes availability, age, item count, reliability score,
log-volume, staleness, and high-volume flags. These structured values are kept
separate from text so the fusion model can learn reliability-aware weights
without embedding numeric metadata as prose.

## Determinism and lineage

Text normalization, ordering, truncation, and feature versions are
deterministic. The manifest records the feature configuration, input file sizes,
SHA-256 hashes, and output row counts. Any data or configuration change creates
a traceable new feature snapshot.
