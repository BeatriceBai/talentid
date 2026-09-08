# Synthetic candidate data contract

TalentID generates candidate profiles from the processed O*NET 31.0 snapshot.
The generator is deterministic for a fixed configuration and seed. It does not
call an LLM, so the first dataset is reproducible and free to rebuild.

## Output tables

| Table | Grain | Purpose |
| --- | --- | --- |
| `candidates` | One row per candidate | Population attributes, occupation, split, and taxonomy version |
| `resumes` | Zero or one row per candidate | Timestamped resume text |
| `job_history` | Zero or more rows per candidate | Job titles, dates, descriptions, and O*NET occupation links |
| `projects` | Zero or more rows per candidate | Timestamped project evidence |
| `courses` | Zero or more rows per candidate | Course evidence linked to canonical skills |
| `self_reported_skills` | Zero or more rows per candidate | Raw aliases, canonical mappings, and reported proficiency |
| `candidate_skill_truth` | Candidate-skill pair | Latent evaluation truth and proficiency |
| `source_status` | Candidate-source pair | Availability, freshness, volume, and reliability metadata |

All output tables are written to `data/synthetic/candidates/v1/`. Generated
Parquet files are local artifacts and should remain excluded from Git.

## Simulated production problems

- Missingness is correlated through complete, partial, and sparse profile
  segments instead of being purely independent.
- Resume dates follow a fresh/stale mixture.
- Self-reported profiles follow terse, typical, and exhaustive claim-volume
  styles.
- High-volume profiles receive extra noisy claims and a reliability penalty.
- Skill surface forms include canonical names, case variants, shortened vendor
  names, symbols, and selected manual synonyms.
- Every row carries or inherits the taxonomy version so a later v2 snapshot can
  test version alignment.
- Train, validation, and test assignment happens at candidate level to prevent
  profile evidence from leaking across splits.

## Leakage boundary

`candidate_skill_truth` and `self_reported_skills.oracle_is_true_skill` are
oracle evaluation fields. Feature pipelines, embedding models, retrieval, and
reranking must not consume them. They may be used only to construct training
labels under an explicitly documented experiment or to evaluate predictions.

## Reproducibility

Generation settings live in `configs/synthetic_candidates.toml`. Running the
generator twice with the same processed O*NET snapshot, configuration, and seed
produces identical tables. Change the taxonomy version or seed when producing a
new benchmark snapshot.
