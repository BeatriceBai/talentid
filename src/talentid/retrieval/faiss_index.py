"""Build and query exact or HNSW FAISS indexes with cosine similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "FAISS is required; run `uv add --optional retrieval faiss-cpu`"
        ) from error
    return faiss


def normalize_embeddings(values: np.ndarray) -> np.ndarray:
    """Return contiguous float32 unit vectors, rejecting malformed rows."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("embeddings must be a nonempty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("embeddings must not contain zero rows")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def build_index(
    skill_embeddings: np.ndarray,
    *,
    index_type: str,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 256,
    threads: int = 1,
) -> Any:
    """Build a cosine-search FAISS index over normalized skill embeddings."""
    if threads < 1:
        raise ValueError("threads must be positive")
    faiss = _faiss()
    faiss.omp_set_num_threads(threads)
    vectors = normalize_embeddings(skill_embeddings)
    dimension = vectors.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        if min(hnsw_m, ef_construction, ef_search) < 1:
            raise ValueError("HNSW parameters must be positive")
        index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError("index_type must be 'flat' or 'hnsw'")
    index.add(vectors)
    return index


def search_index(
    index: Any,
    query_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return skill row positions and cosine scores, ordered best first."""
    if not 1 <= k <= int(index.ntotal):
        raise ValueError("k must be between 1 and the indexed skill count")
    scores, indices = index.search(normalize_embeddings(query_embeddings), k)
    if (indices < 0).any():
        raise RuntimeError("FAISS returned an incomplete result")
    return indices.astype(np.int64), scores.astype(np.float32)


def neighbor_recall(reference: np.ndarray, observed: np.ndarray) -> float:
    """Calculate average top-k set recall against reference neighbors."""
    if reference.shape != observed.shape or reference.ndim != 2:
        raise ValueError("neighbor matrices must have the same 2-D shape")
    if reference.shape[1] == 0:
        raise ValueError("neighbor matrices must contain at least one column")
    recalls = [
        len(set(expected).intersection(actual)) / reference.shape[1]
        for expected, actual in zip(reference, observed, strict=True)
    ]
    return float(np.mean(recalls))


def save_index(index: Any, path: Path) -> None:
    """Persist an index for the serving layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _faiss().write_index(index, str(path))


def load_index(path: Path, *, ef_search: int | None = None) -> Any:
    """Load a persisted FAISS index and optionally set HNSW search effort."""
    if not path.exists():
        raise FileNotFoundError(path)
    index = _faiss().read_index(str(path))
    if ef_search is not None:
        if ef_search < 1 or not hasattr(index, "hnsw"):
            raise ValueError("ef_search requires a positive value and an HNSW index")
        index.hnsw.efSearch = ef_search
    return index
