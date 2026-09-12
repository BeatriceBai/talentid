# Hybrid-pool reranking

This milestone reruns the reliability-aware evidence reranker over the expanded
top-200 hybrid candidate pool. It writes a separate v2 snapshot, leaving the
original learned-top-100 reranker under `data/evaluation/reranking/v1`.

## Inputs and scoring

The candidate pool is the union produced by hybrid retrieval: the protected
learned top 100 plus 100 source-tower and occupation candidates. The reranker
uses the same independently computed signals as v1:

1. hybrid reciprocal-rank score;
2. reliability-gated source-tower evidence;
3. train-only skill-popularity penalty;
4. MMR-style semantic diversity penalty.

The validation search evaluates 28 deterministic configurations. The selected
configuration maximizes catalog coverage among trials whose NDCG@30 is no lower
than the unmodified hybrid-pool ranking. Test labels remain sealed.

## Evaluation

The profiler compares three top-30 lists:

- selected v1 reranker over learned top 100;
- unmodified hybrid top-200 scoring;
- selected hybrid-aware v2 reranker.

The comparison reports precision, recall, NDCG, hit rate, MRR, catalog coverage,
mean train popularity, and intra-list similarity. The result determines whether
the pipeline is ready for LLM judging or needs further reranker calibration.

