# Evidence- and diversity-aware reranking

The learned fusion model substantially improves validation relevance but
concentrates its top-30 predictions in a small fraction of the skill catalog.
This milestone reranks the exact FAISS top-100 pool into a top-30 list.

## Scoring signals

1. **Learned retrieval score** from the trainable 128-dimensional fusion model.
2. **Independent source evidence** from cosine similarity between each original
   tower and the original skill embedding, aggregated with learned tower gates.
3. **Train-only popularity penalty** derived exclusively from training-positive
   skill frequency.
4. **MMR-style diversity penalty** based on the maximum similarity to a skill
   already selected for the list.

Candidate-level retrieval and evidence scores are min-max normalized before
combination. Greedy selection is deterministic.

## Model selection

The validation grid evaluates 13 configurations, including the unmodified
retrieval ranking. The selected trial maximizes catalog coverage while requiring
NDCG@30 to remain within 0.02 absolute of the retrieval-only result. NDCG,
precision, recall, MRR, hit rate, catalog coverage, mean train popularity, and
mean intra-list similarity are retained for every trial.

Only train labels are used for popularity and validation labels for tuning.
Test labels remain sealed.
