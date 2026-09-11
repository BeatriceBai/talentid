# Learned-embedding retrieval and FAISS index

This milestone turns the trained 128-dimensional fusion embeddings into a
serving-style candidate-to-skill retrieval layer. It evaluates only validation
labels from the mined positive-pair artifact; test labels remain sealed.

## Search designs

- **NumPy exact** is the deterministic correctness reference.
- **FAISS IndexFlatIP** must recover the same cosine neighbors after L2
  normalization.
- **FAISS HNSW** is the serving candidate. It trades a small amount of neighbor
  recall for faster search as the skill catalog grows.

The configuration uses `M=32`, `efConstruction=200`, and `efSearch=256`. The
pipeline fails when HNSW recall@100 against exact FAISS falls below 0.99.

## Evaluation contract

The report includes precision, recall, NDCG, hit rate, MRR, catalog coverage,
ANN neighbor recall, index construction time, search throughput, and absolute
and relative lift over the frozen reliability-weighted baseline. The saved
HNSW index contains skill row positions aligned to the learned skill index.

Both Flat and HNSW indexes are persisted. At the current 8,821-skill catalog
size, latency and fidelity measurements determine which index should serve;
the generated indexes and evaluation tables are excluded from Git.
