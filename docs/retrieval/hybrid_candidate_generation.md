# Hybrid candidate generation

The learned dense retriever has strong micro recall, but the coverage audit
shows that its top-100 pools reach only a small fraction of distinct validation
truth skills. The hybrid generator expands each pool to 200 without removing
any learned top-100 candidate.

## Channels

- The learned fusion model contributes its complete exact-FAISS top 100.
- Each available source tower independently retrieves semantically similar
  skills from the frozen embedding space. Its reciprocal-rank contribution is
  weighted by that candidate-source reliability score.
- The candidate's primary O*NET occupation contributes high-importance,
  high-level, hot, and in-demand skills from the public taxonomy.

The extra candidates are ordered with weighted reciprocal-rank fusion and
deterministic skill-ID tie breaking. Candidate-skill truth is used only to
evaluate the completed pools, never to generate them. Test labels remain
sealed.

## Recall-preserving contract

Ranks 1-100 are exactly the original learned candidates. Ranks 101-200 are the
highest-scoring new candidates from the tower and occupation channels. This
means the hybrid pool cannot reduce the learned top-100 recall.

After profiling, the next milestone is to rerun the evidence-aware reranker on
the hybrid top-200 pool and compare it with the existing learned-top-100
reranker under the same NDCG constraint.

