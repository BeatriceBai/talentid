# Coverage and candidate-pool audit

Catalog coverage alone cannot distinguish model concentration from limited
validation truth diversity. This audit separates the two and measures whether
the learned top-100 pool provides enough candidates for downstream reranking.

## Reported stages

- Frozen reliability-weighted top 30
- Learned exact-FAISS top 30
- Learned exact-FAISS top-100 candidate pool
- Selected evidence-aware reranked top 30

For every stage, the audit reports catalog coverage, coverage of distinct
validation-truth skills, micro truth-pair recall, mean candidate recall,
top-1%-skill exposure share, Herfindahl concentration, and normalized entropy.
It also reports catalog and truth coverage by O*NET skill type and the most
frequently exposed skills.

## Decision rule

Hybrid candidate generation is recommended when any of the following fails:

- top-100 truth-pair recall of 75%;
- top-100 coverage of 80% of distinct validation-truth skills;
- reranked top-30 coverage of 50% of distinct validation-truth skills.

Otherwise, the pipeline can proceed to the LLM judge. Only train and validation
pair artifacts are loaded; test labels remain sealed.
