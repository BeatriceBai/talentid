"""Encode tower text and build a reliability-weighted candidate baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "embeddings.toml"

SOURCES = (
    "resume",
    "projects",
    "job_history",
    "courses",
    "self_reported_skills",
)


class TextEncoder(Protocol):
    """Small interface that keeps unit tests independent of model downloads."""

    @property
    def dimension(self) -> int:
        """Return the output embedding dimension."""

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """Return one float vector per text."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Frozen encoder and artifact settings."""

    embedding_version: str = "talentskill-embeddings-v1"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    batch_size: int = 64
    device: str = "auto"
    normalize: bool = True
    candidate_prefix: str = "Candidate {source} evidence: "
    skill_prefix: str = "Skill: "

    def validate(self) -> None:
        if not self.embedding_version.strip():
            raise ValueError("embedding_version must be nonempty")
        if not self.model_name.strip() or not self.model_revision.strip():
            raise ValueError("model name and revision must be nonempty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.normalize:
            raise ValueError("v1 requires normalized embeddings")
        if "{source}" not in self.candidate_prefix:
            raise ValueError("candidate prefix must contain {source}")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> EmbeddingConfig:
    """Load embedding settings from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    settings = raw["embeddings"]
    prefixes = raw["text_prefixes"]
    config = EmbeddingConfig(
        embedding_version=str(settings["embedding_version"]),
        model_name=str(settings["model_name"]),
        model_revision=str(settings["model_revision"]),
        batch_size=int(settings["batch_size"]),
        device=str(settings["device"]),
        normalize=bool(settings["normalize"]),
        candidate_prefix=str(prefixes["candidate"]),
        skill_prefix=str(prefixes["skill"]),
    )
    config.validate()
    return config


class SentenceTransformerEncoder:
    """Lazy adapter around Sentence Transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required; run "
                "`uv add 'sentence-transformers>=5,<7'`"
            ) from error

        device = None if config.device == "auto" else config.device
        self._model = SentenceTransformer(
            config.model_name,
            revision=config.model_revision,
            device=device,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("encoder did not report an embedding dimension")
        self._dimension = int(dimension)
        self._normalize = config.normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return np.asarray(vectors, dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > 0,
    )


def reliability_weighted_fusion(
    tower_embeddings: np.ndarray,
    reliability_scores: np.ndarray,
    candidate_ids: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Fuse available tower vectors, then L2-normalize each candidate vector."""
    vectors = np.asarray(tower_embeddings, dtype=np.float32)
    weights = np.asarray(reliability_scores, dtype=np.float32)
    ids = np.asarray(candidate_ids, dtype=str)
    if vectors.ndim != 2 or len(vectors) != len(weights) or len(ids) != len(weights):
        raise ValueError("tower vectors, weights, and candidate ids must align")
    if np.any(weights < 0):
        raise ValueError("reliability scores cannot be negative")

    ordered_ids = sorted(set(ids.tolist()))
    fused = np.zeros((len(ordered_ids), vectors.shape[1]), dtype=np.float32)
    for output_row, candidate_id in enumerate(ordered_ids):
        mask = ids == candidate_id
        candidate_vectors = vectors[mask]
        candidate_weights = weights[mask]
        positive = candidate_weights > 0
        if positive.any():
            fused[output_row] = np.average(
                candidate_vectors[positive],
                axis=0,
                weights=candidate_weights[positive],
            )
        elif np.any(np.linalg.norm(candidate_vectors, axis=1) > 0):
            fused[output_row] = candidate_vectors.mean(axis=0)
    return _l2_normalize(fused), ordered_ids


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _encode_available(
    table: pd.DataFrame,
    text_column: str,
    available_column: str,
    prefix: Sequence[str],
    encoder: TextEncoder,
    batch_size: int,
) -> np.ndarray:
    output = np.zeros((len(table), encoder.dimension), dtype=np.float32)
    available = table[available_column].astype(bool).to_numpy()
    positions = np.flatnonzero(available)
    if len(positions) == 0:
        return output
    texts = [
        f"{prefix[position]}{table.iloc[position][text_column]}"
        for position in positions
    ]
    encoded = np.asarray(encoder.encode(texts, batch_size), dtype=np.float32)
    expected_shape = (len(positions), encoder.dimension)
    if encoded.shape != expected_shape:
        raise ValueError(
            f"encoder returned {encoded.shape}; expected {expected_shape}"
        )
    output[positions] = _l2_normalize(encoded)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_array(path: Path, array: np.ndarray) -> dict[str, object]:
    np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": "float32",
        "sha256": _sha256(path),
    }


def build_embedding_artifacts(
    candidate_towers: pd.DataFrame,
    skills: pd.DataFrame,
    candidate_index: pd.DataFrame,
    encoder: TextEncoder,
    config: EmbeddingConfig,
) -> dict[str, object]:
    """Build aligned tower, skill, and fused candidate arrays in memory."""
    config.validate()
    _require_columns(
        candidate_towers,
        {
            "candidate_id",
            "source",
            "tower_text",
            "is_available",
            "reliability_score",
            "split",
            "feature_version",
        },
        "candidate_tower_features",
    )
    _require_columns(
        skills,
        {"skill_id", "skill_text", "feature_version"},
        "skill_features",
    )
    _require_columns(candidate_index, {"candidate_id", "split"}, "candidate_index")

    unknown_sources = sorted(set(candidate_towers["source"]) - set(SOURCES))
    if unknown_sources:
        raise ValueError(f"unknown candidate sources: {unknown_sources}")
    if candidate_towers.duplicated(["candidate_id", "source"]).any():
        raise ValueError("candidate tower keys must be unique")
    if skills["skill_id"].duplicated().any():
        raise ValueError("skill ids must be unique")
    source_sets = candidate_towers.groupby("candidate_id")["source"].agg(set)
    incomplete = source_sets[source_sets.map(lambda values: values != set(SOURCES))]
    if not incomplete.empty:
        raise ValueError("each candidate must have exactly one row for every source")
    if candidate_index["candidate_id"].duplicated().any():
        raise ValueError("candidate index ids must be unique")

    tower_order = pd.Categorical(candidate_towers["source"], categories=SOURCES)
    towers = candidate_towers.assign(_source_order=tower_order).sort_values(
        ["candidate_id", "_source_order"], kind="stable"
    ).drop(columns="_source_order").reset_index(drop=True)
    skill_table = skills.sort_values("skill_id", kind="stable").reset_index(drop=True)

    tower_prefixes = [
        config.candidate_prefix.format(source=source.replace("_", " "))
        for source in towers["source"]
    ]
    tower_vectors = _encode_available(
        towers,
        "tower_text",
        "is_available",
        tower_prefixes,
        encoder,
        config.batch_size,
    )
    skill_vectors = _encode_available(
        skill_table.assign(_available=True),
        "skill_text",
        "_available",
        [config.skill_prefix] * len(skill_table),
        encoder,
        config.batch_size,
    )
    candidate_vectors, ordered_candidate_ids = reliability_weighted_fusion(
        tower_vectors,
        towers["reliability_score"].to_numpy(dtype=np.float32),
        towers["candidate_id"].astype(str).tolist(),
    )

    split_lookup = candidate_index.set_index("candidate_id")["split"]
    fused_index = pd.DataFrame({"candidate_id": ordered_candidate_ids})
    fused_index["split"] = fused_index["candidate_id"].map(split_lookup)
    if fused_index["split"].isna().any():
        raise ValueError("candidate index does not cover every tower candidate")

    tower_index = towers[
        [
            "candidate_id",
            "source",
            "is_available",
            "reliability_score",
            "split",
            "feature_version",
        ]
    ].copy()
    tower_index.insert(0, "embedding_row", np.arange(len(tower_index)))
    skill_index = skill_table[["skill_id", "feature_version"]].copy()
    skill_index.insert(0, "embedding_row", np.arange(len(skill_index)))
    fused_index.insert(0, "embedding_row", np.arange(len(fused_index)))
    fused_index["embedding_version"] = config.embedding_version

    return {
        "candidate_tower_embeddings": tower_vectors,
        "candidate_tower_embedding_index": tower_index,
        "skill_embeddings": skill_vectors,
        "skill_embedding_index": skill_index,
        "candidate_embeddings": candidate_vectors,
        "candidate_embedding_index": fused_index,
    }


def load_feature_tables(feature_dir: Path) -> dict[str, pd.DataFrame]:
    required = (
        "candidate_tower_features",
        "skill_features",
        "candidate_index",
    )
    paths = {name: feature_dir / f"{name}.parquet" for name in required}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Missing feature inputs:\n" + lines)
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def write_embedding_artifacts(
    artifacts: dict[str, object],
    feature_dir: Path,
    output_dir: Path,
    config: EmbeddingConfig,
) -> dict[str, object]:
    """Write aligned arrays, indices, and a reproducibility manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "embedding_version": config.embedding_version,
        "model": {
            "name": config.model_name,
            "revision": config.model_revision,
            "dimension": int(
                np.asarray(artifacts["skill_embeddings"]).shape[1]
            ),
            "normalized": config.normalize,
        },
        "config": asdict(config),
        "inputs": {},
        "outputs": {},
    }
    inputs = manifest["inputs"]
    assert isinstance(inputs, dict)
    for name in (
        "candidate_tower_features",
        "skill_features",
        "candidate_index",
    ):
        path = feature_dir / f"{name}.parquet"
        inputs[name] = {"path": str(path), "sha256": _sha256(path)}

    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    for name in (
        "candidate_tower_embeddings",
        "skill_embeddings",
        "candidate_embeddings",
    ):
        array = np.asarray(artifacts[name], dtype=np.float32)
        outputs[name] = _write_array(output_dir / f"{name}.npy", array)
    for name in (
        "candidate_tower_embedding_index",
        "skill_embedding_index",
        "candidate_embedding_index",
    ):
        table = artifacts[name]
        assert isinstance(table, pd.DataFrame)
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        outputs[name] = {
            "path": path.name,
            "rows": len(table),
            "sha256": _sha256(path),
        }

    manifest_path = output_dir / "embedding_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tables = load_feature_tables(args.feature_dir)
    encoder = SentenceTransformerEncoder(config)
    artifacts = build_embedding_artifacts(
        tables["candidate_tower_features"],
        tables["skill_features"],
        tables["candidate_index"],
        encoder,
        config,
    )
    manifest = write_embedding_artifacts(
        artifacts,
        args.feature_dir,
        args.output_dir,
        config,
    )
    print(f"Embedding snapshot: {args.output_dir.resolve()}")
    for name, metadata in manifest["outputs"].items():
        if name.endswith("_embeddings"):
            print(f"{name}: {metadata['shape']}")


if __name__ == "__main__":
    main()
