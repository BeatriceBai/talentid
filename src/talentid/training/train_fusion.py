"""Train and export the reliability-aware candidate-skill fusion model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from talentid.evaluation.evaluate_retrieval import (
    compute_candidate_metrics,
    retrieve_top_k,
)
from talentid.modeling.fusion import (
    ReliabilityAwareFusion,
    weighted_contrastive_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "v1"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "v1"
DEFAULT_PAIR_DIR = PROJECT_ROOT / "data" / "training" / "pairs" / "v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "fusion" / "v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "fusion_model.toml"

SOURCES = (
    "resume",
    "projects",
    "job_history",
    "courses",
    "self_reported_skills",
)
QUALITY_COLUMNS = (
    "reliability_score",
    "freshness_score",
    "volume_score",
    "is_stale",
    "is_high_volume",
    "is_available",
)


@dataclass(frozen=True)
class FusionTrainingConfig:
    """Model, optimization, validation, and quality-transform settings."""

    model_version: str = "talentskill-fusion-v1"
    input_dimension: int = 384
    output_dimension: int = 128
    quality_dimension: int = 6
    gate_hidden_dimension: int = 32
    dropout: float = 0.05
    epochs: int = 5
    batch_size: int = 512
    hard_negatives_per_example: int = 4
    random_negatives_per_example: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    gradient_clip_norm: float = 1.0
    patience: int = 2
    seed: int = 20_260_909
    device: str = "auto"
    validation_cutoff: int = 30
    query_batch_size: int = 256
    stale_after_days: int = 730
    volume_scale: int = 40

    def validate(self) -> None:
        if not self.model_version.strip():
            raise ValueError("model_version must be nonempty")
        dimensions = (
            self.input_dimension,
            self.output_dimension,
            self.quality_dimension,
            self.gate_hidden_dimension,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        counts = (
            self.epochs,
            self.batch_size,
            self.hard_negatives_per_example,
            self.query_batch_size,
            self.stale_after_days,
            self.volume_scale,
        )
        if any(value < 1 for value in counts):
            raise ValueError("training counts and scales must be positive")
        if self.random_negatives_per_example < 0 or self.patience < 0:
            raise ValueError("random-negative count and patience cannot be negative")
        if self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("learning rate and temperature must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("weight decay and gradient clipping are invalid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.validation_cutoff < 1:
            raise ValueError("validation cutoff must be positive")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> FusionTrainingConfig:
    """Load the model configuration from TOML."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    model = raw["model"]
    training = raw["training"]
    validation = raw["validation"]
    quality = raw["quality"]
    config = FusionTrainingConfig(
        model_version=str(model["model_version"]),
        input_dimension=int(model["input_dimension"]),
        output_dimension=int(model["output_dimension"]),
        quality_dimension=int(model["quality_dimension"]),
        gate_hidden_dimension=int(model["gate_hidden_dimension"]),
        dropout=float(model["dropout"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        hard_negatives_per_example=int(
            training["hard_negatives_per_example"]
        ),
        random_negatives_per_example=int(
            training["random_negatives_per_example"]
        ),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        temperature=float(training["temperature"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        patience=int(training["patience"]),
        seed=int(training["seed"]),
        device=str(training["device"]),
        validation_cutoff=int(validation["cutoff"]),
        query_batch_size=int(validation["query_batch_size"]),
        stale_after_days=int(quality["stale_after_days"]),
        volume_scale=int(quality["volume_scale"]),
    )
    config.validate()
    if config.quality_dimension != len(QUALITY_COLUMNS):
        raise ValueError("quality_dimension does not match engineered features")
    return config


@dataclass
class FusionData:
    """Aligned arrays and index mappings used by training and export."""

    candidate_ids: list[str]
    candidate_splits: list[str]
    tower_embeddings: np.ndarray
    quality_features: np.ndarray
    availability: np.ndarray
    skill_ids: list[str]
    skill_embeddings: np.ndarray
    positive_pairs: pd.DataFrame
    negative_pairs: pd.DataFrame


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _complete_rows(table: pd.DataFrame, length: int, name: str) -> None:
    if sorted(table["embedding_row"].tolist()) != list(range(length)):
        raise ValueError(f"{name} embedding rows must be a complete permutation")


def _quality_vector(row: pd.Series, config: FusionTrainingConfig) -> list[float]:
    available = bool(row["is_available"])
    if not available:
        return [0.0] * len(QUALITY_COLUMNS)
    age_days = float(row["age_days"]) if pd.notna(row["age_days"]) else 0.0
    freshness = math.exp(-max(age_days, 0.0) / config.stale_after_days)
    item_count = max(float(row["item_count"]), 0.0)
    volume = min(
        math.log1p(item_count) / math.log1p(config.volume_scale),
        2.0,
    )
    return [
        float(np.clip(row["reliability_score"], 0, 1)),
        freshness,
        volume,
        float(bool(row["is_stale"])),
        float(bool(row["is_high_volume"])),
        1.0,
    ]


def prepare_fusion_data(
    candidate_tower_embeddings: np.ndarray,
    candidate_tower_index: pd.DataFrame,
    candidate_index: pd.DataFrame,
    candidate_tower_features: pd.DataFrame,
    skill_embeddings: np.ndarray,
    skill_index: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    negative_pairs: pd.DataFrame,
    config: FusionTrainingConfig,
) -> FusionData:
    """Align row-indexed artifacts into dense candidate-by-source tensors."""
    config.validate()
    _require_columns(
        candidate_index,
        {"embedding_row", "candidate_id", "split"},
        "candidate_embedding_index",
    )
    _require_columns(
        candidate_tower_index,
        {"embedding_row", "candidate_id", "source", "is_available"},
        "candidate_tower_embedding_index",
    )
    _require_columns(
        candidate_tower_features,
        {
            "candidate_id",
            "source",
            "is_available",
            "reliability_score",
            "age_days",
            "item_count",
            "is_stale",
            "is_high_volume",
        },
        "candidate_tower_features",
    )
    _require_columns(
        skill_index,
        {"embedding_row", "skill_id"},
        "skill_embedding_index",
    )
    _require_columns(
        positive_pairs,
        {"candidate_id", "skill_id", "split", "sample_weight"},
        "positive_pairs",
    )
    _require_columns(
        negative_pairs,
        {"candidate_id", "skill_id", "split", "negative_type"},
        "negative_pairs",
    )
    if len(candidate_tower_embeddings) != len(candidate_tower_index):
        raise ValueError("candidate tower embeddings and index do not align")
    if len(skill_embeddings) != len(skill_index):
        raise ValueError("skill embeddings and index do not align")
    if candidate_tower_embeddings.shape[1] != config.input_dimension:
        raise ValueError("candidate tower embedding dimension does not match config")
    if skill_embeddings.shape[1] != config.input_dimension:
        raise ValueError("skill embedding dimension does not match config")
    _complete_rows(candidate_index, len(candidate_index), "candidate")
    _complete_rows(
        candidate_tower_index, len(candidate_tower_embeddings), "candidate tower"
    )
    _complete_rows(skill_index, len(skill_embeddings), "skill")
    if candidate_tower_index.duplicated(["candidate_id", "source"]).any():
        raise ValueError("candidate tower index keys must be unique")

    candidates = candidate_index.sort_values("embedding_row").reset_index(drop=True)
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate ids must be unique")
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    candidate_splits = candidates["split"].astype(str).tolist()
    candidate_position = {
        candidate_id: position for position, candidate_id in enumerate(candidate_ids)
    }
    source_position = {source: position for position, source in enumerate(SOURCES)}

    feature_keys = candidate_tower_features.copy()
    feature_keys["candidate_id"] = feature_keys["candidate_id"].astype(str)
    if feature_keys.duplicated(["candidate_id", "source"]).any():
        raise ValueError("candidate tower feature keys must be unique")
    feature_lookup = feature_keys.set_index(["candidate_id", "source"])

    towers = np.zeros(
        (len(candidates), len(SOURCES), config.input_dimension), dtype=np.float32
    )
    quality = np.zeros(
        (len(candidates), len(SOURCES), config.quality_dimension), dtype=np.float32
    )
    availability = np.zeros((len(candidates), len(SOURCES)), dtype=bool)
    tower_index = candidate_tower_index.sort_values("embedding_row")
    for index_row in tower_index.itertuples(index=False):
        candidate_id = str(index_row.candidate_id)
        source = str(index_row.source)
        if candidate_id not in candidate_position or source not in source_position:
            raise ValueError(f"unknown candidate tower key: {(candidate_id, source)}")
        key = (candidate_id, source)
        if key not in feature_lookup.index:
            raise ValueError(f"missing candidate tower features for {key}")
        candidate_row = candidate_position[candidate_id]
        source_row = source_position[source]
        feature_row = feature_lookup.loc[key]
        if bool(index_row.is_available) != bool(feature_row["is_available"]):
            raise ValueError(f"availability disagreement for {key}")
        towers[candidate_row, source_row] = candidate_tower_embeddings[
            int(index_row.embedding_row)
        ]
        quality[candidate_row, source_row] = _quality_vector(feature_row, config)
        availability[candidate_row, source_row] = bool(index_row.is_available)
    if (~availability.any(axis=1)).any():
        raise ValueError("every candidate must have at least one available tower")

    skills = skill_index.sort_values("embedding_row").reset_index(drop=True)
    skill_rows = skills["embedding_row"].to_numpy(dtype=int)
    ordered_skill_embeddings = skill_embeddings[skill_rows].astype(np.float32)
    skill_ids = skills["skill_id"].astype(str).tolist()

    positives = positive_pairs.copy()
    negatives = negative_pairs.copy()
    for table in (positives, negatives):
        table["candidate_id"] = table["candidate_id"].astype(str)
        table["skill_id"] = table["skill_id"].astype(str)
    if "test" in set(positives["split"]) or "test" in set(negatives["split"]):
        raise ValueError("training artifacts expose the sealed test split")
    split_lookup = dict(zip(candidate_ids, candidate_splits, strict=True))
    for name, table in (("positive", positives), ("negative", negatives)):
        expected_splits = table["candidate_id"].map(split_lookup)
        if expected_splits.isna().any():
            raise ValueError(f"{name} pairs contain unknown candidate ids")
        if not expected_splits.eq(table["split"]).all():
            raise ValueError(f"{name} pair split disagrees with candidate index")
    known_skills = set(skill_ids)
    if not set(positives["skill_id"]) <= known_skills:
        raise ValueError("positive pairs contain unknown skill ids")
    if not set(negatives["skill_id"]) <= known_skills:
        raise ValueError("negative pairs contain unknown skill ids")

    return FusionData(
        candidate_ids=candidate_ids,
        candidate_splits=candidate_splits,
        tower_embeddings=towers,
        quality_features=quality,
        availability=availability,
        skill_ids=skill_ids,
        skill_embeddings=ordered_skill_embeddings,
        positive_pairs=positives,
        negative_pairs=negatives,
    )


def resolve_device(requested: str) -> torch.device:
    """Select CUDA, Apple MPS, or CPU without making hardware assumptions."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible_seed(seed: int) -> None:
    """Seed NumPy and PyTorch and request deterministic algorithms when possible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _position_maps(data: FusionData) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {value: index for index, value in enumerate(data.candidate_ids)},
        {value: index for index, value in enumerate(data.skill_ids)},
    )


def _negative_pools(
    negative_pairs: pd.DataFrame, skill_position: dict[str, int]
) -> dict[str, dict[str, np.ndarray]]:
    pools: dict[str, dict[str, np.ndarray]] = {}
    for (candidate_id, negative_type), group in negative_pairs.groupby(
        ["candidate_id", "negative_type"], sort=False
    ):
        pools.setdefault(str(candidate_id), {})[str(negative_type)] = np.asarray(
            [skill_position[str(skill_id)] for skill_id in group["skill_id"]],
            dtype=np.int64,
        )
    return pools


def _sample_negative_batch(
    candidate_ids: list[str],
    pools: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    hard_count: int,
    random_count: int,
) -> np.ndarray:
    rows = []
    for candidate_id in candidate_ids:
        candidate_pools = pools.get(candidate_id, {})
        hard = candidate_pools.get("hard_retrieval", np.asarray([], dtype=np.int64))
        random = candidate_pools.get("random", np.asarray([], dtype=np.int64))
        if len(hard) < hard_count or len(random) < random_count:
            raise ValueError(f"insufficient negative pool for {candidate_id}")
        hard_sample = rng.choice(hard, size=hard_count, replace=False)
        random_sample = rng.choice(random, size=random_count, replace=False)
        rows.append(np.concatenate([hard_sample, random_sample]))
    return np.asarray(rows, dtype=np.int64)


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def encode_all(
    model: ReliabilityAwareFusion,
    data: FusionData,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode every candidate and skill and collect candidate gate weights."""
    model.eval()
    candidate_vectors = []
    gate_weights = []
    skill_vectors = []
    with torch.inference_mode():
        for start in range(0, len(data.candidate_ids), batch_size):
            stop = min(start + batch_size, len(data.candidate_ids))
            vectors, gates = model.encode_candidates(
                _tensor(data.tower_embeddings[start:stop], device),
                _tensor(data.quality_features[start:stop], device),
                torch.as_tensor(data.availability[start:stop], device=device),
            )
            candidate_vectors.append(vectors.cpu().numpy())
            gate_weights.append(gates.cpu().numpy())
        for start in range(0, len(data.skill_ids), batch_size * 4):
            stop = min(start + batch_size * 4, len(data.skill_ids))
            vectors = model.encode_skills(
                _tensor(data.skill_embeddings[start:stop], device)
            )
            skill_vectors.append(vectors.cpu().numpy())
    return (
        np.concatenate(candidate_vectors).astype(np.float32),
        np.concatenate(skill_vectors).astype(np.float32),
        np.concatenate(gate_weights).astype(np.float32),
    )


def validation_metrics(
    model: ReliabilityAwareFusion,
    data: FusionData,
    device: torch.device,
    config: FusionTrainingConfig,
) -> dict[str, float]:
    """Evaluate the validation split against the full skill catalog."""
    candidate_vectors, skill_vectors, _ = encode_all(
        model, data, device, config.batch_size
    )
    validation_positions = np.asarray(
        [
            position
            for position, split in enumerate(data.candidate_splits)
            if split == "validation"
        ],
        dtype=np.int64,
    )
    if len(validation_positions) == 0:
        raise ValueError("no validation candidates were found")
    validation_ids = [data.candidate_ids[position] for position in validation_positions]
    top_indices, _ = retrieve_top_k(
        candidate_vectors[validation_positions],
        skill_vectors,
        config.validation_cutoff,
        config.query_batch_size,
    )
    retrieved_ids = np.asarray(data.skill_ids, dtype=str)[top_indices]
    validation_truth = data.positive_pairs[
        data.positive_pairs["split"].eq("validation")
    ]
    candidate_metrics = compute_candidate_metrics(
        retrieved_ids,
        validation_truth,
        validation_ids,
        (config.validation_cutoff,),
    )
    return {
        "precision": float(candidate_metrics["precision"].mean()),
        "recall": float(candidate_metrics["recall"].mean()),
        "ndcg": float(candidate_metrics["ndcg"].mean()),
        "hit_rate": float(candidate_metrics["hit_rate"].mean()),
        "mrr": float(candidate_metrics["mrr"].mean()),
    }


def train_model(
    data: FusionData,
    config: FusionTrainingConfig,
) -> tuple[ReliabilityAwareFusion, pd.DataFrame, torch.device]:
    """Train with candidate-specific hard/random negatives and early stopping."""
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    model = ReliabilityAwareFusion(
        input_dimension=config.input_dimension,
        output_dimension=config.output_dimension,
        quality_dimension=config.quality_dimension,
        source_count=len(SOURCES),
        gate_hidden_dimension=config.gate_hidden_dimension,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    candidate_position, skill_position = _position_maps(data)
    pools = _negative_pools(data.negative_pairs, skill_position)
    train_pairs = data.positive_pairs[
        data.positive_pairs["split"].eq("train")
    ].reset_index(drop=True)
    if train_pairs.empty:
        raise ValueError("no training positive pairs were found")

    history_rows = []
    initial_metrics = validation_metrics(model, data, device, config)
    history_rows.append(
        {"epoch": 0, "train_loss": np.nan, "device": str(device), **initial_metrics}
    )
    print(
        f"epoch 0 validation: recall@{config.validation_cutoff}="
        f"{initial_metrics['recall']:.4f}, ndcg@{config.validation_cutoff}="
        f"{initial_metrics['ndcg']:.4f}"
    )
    best_ndcg = initial_metrics["ndcg"]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        rng = np.random.default_rng(config.seed + epoch)
        order = rng.permutation(len(train_pairs))
        total_weighted_loss = 0.0
        total_examples = 0
        for start in range(0, len(order), config.batch_size):
            batch_rows = order[start : start + config.batch_size]
            batch = train_pairs.iloc[batch_rows]
            batch_candidate_ids = batch["candidate_id"].astype(str).tolist()
            candidate_rows = np.asarray(
                [candidate_position[value] for value in batch_candidate_ids],
                dtype=np.int64,
            )
            positive_rows = np.asarray(
                [skill_position[str(value)] for value in batch["skill_id"]],
                dtype=np.int64,
            )
            negative_rows = _sample_negative_batch(
                batch_candidate_ids,
                pools,
                rng,
                config.hard_negatives_per_example,
                config.random_negatives_per_example,
            )
            sample_weights = batch["sample_weight"].to_numpy(dtype=np.float32)

            optimizer.zero_grad(set_to_none=True)
            candidate_vectors, _ = model.encode_candidates(
                _tensor(data.tower_embeddings[candidate_rows], device),
                _tensor(data.quality_features[candidate_rows], device),
                torch.as_tensor(data.availability[candidate_rows], device=device),
            )
            positive_vectors = model.encode_skills(
                _tensor(data.skill_embeddings[positive_rows], device)
            )
            flat_negative_vectors = model.encode_skills(
                _tensor(
                    data.skill_embeddings[negative_rows.reshape(-1)],
                    device,
                )
            )
            negative_vectors = flat_negative_vectors.reshape(
                len(batch), len(negative_rows[0]), config.output_dimension
            )
            loss = weighted_contrastive_loss(
                candidate_vectors,
                positive_vectors,
                negative_vectors,
                _tensor(sample_weights, device),
                config.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            total_weighted_loss += float(loss.detach().cpu()) * len(batch)
            total_examples += len(batch)

        metrics = validation_metrics(model, data, device, config)
        epoch_loss = total_weighted_loss / total_examples
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "device": str(device),
                **metrics,
            }
        )
        print(
            f"epoch {epoch}: loss={epoch_loss:.4f}, "
            f"recall@{config.validation_cutoff}={metrics['recall']:.4f}, "
            f"ndcg@{config.validation_cutoff}={metrics['ndcg']:.4f}"
        )
        if metrics["ndcg"] > best_ndcg + 1e-6:
            best_ndcg = metrics["ndcg"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > config.patience:
                print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    history["is_best"] = history["epoch"].eq(best_epoch)
    return model, history, device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(
    model: ReliabilityAwareFusion,
    data: FusionData,
    history: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
    config: FusionTrainingConfig,
    input_paths: list[Path],
) -> dict[str, Any]:
    """Save the best checkpoint, learned vectors, gates, history, and lineage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_vectors, skill_vectors, gates = encode_all(
        model, data, device, config.batch_size
    )

    candidate_path = output_dir / "learned_candidate_embeddings.npy"
    skill_path = output_dir / "learned_skill_embeddings.npy"
    np.save(candidate_path, candidate_vectors, allow_pickle=False)
    np.save(skill_path, skill_vectors, allow_pickle=False)

    candidate_index = pd.DataFrame(
        {
            "embedding_row": np.arange(len(data.candidate_ids)),
            "candidate_id": data.candidate_ids,
            "split": data.candidate_splits,
            "model_version": config.model_version,
        }
    )
    skill_index = pd.DataFrame(
        {
            "embedding_row": np.arange(len(data.skill_ids)),
            "skill_id": data.skill_ids,
            "model_version": config.model_version,
        }
    )
    gate_rows = []
    for candidate_row, candidate_id in enumerate(data.candidate_ids):
        for source_row, source in enumerate(SOURCES):
            gate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": data.candidate_splits[candidate_row],
                    "source": source,
                    **{
                        name: float(
                            data.quality_features[
                                candidate_row, source_row, feature_position
                            ]
                        )
                        for feature_position, name in enumerate(QUALITY_COLUMNS)
                    },
                    # Keep the exported availability column Boolean. The quality
                    # vector also contains this signal as a numeric model input.
                    "is_available": bool(
                        data.availability[candidate_row, source_row]
                    ),
                    "gate_weight": float(gates[candidate_row, source_row]),
                    "model_version": config.model_version,
                }
            )
    gate_table = pd.DataFrame(gate_rows)

    table_paths = {
        "candidate_embedding_index": output_dir
        / "candidate_embedding_index.parquet",
        "skill_embedding_index": output_dir / "skill_embedding_index.parquet",
        "gate_weights": output_dir / "gate_weights.parquet",
        "training_history": output_dir / "training_history.parquet",
    }
    candidate_index.to_parquet(table_paths["candidate_embedding_index"], index=False)
    skill_index.to_parquet(table_paths["skill_embedding_index"], index=False)
    gate_table.to_parquet(table_paths["gate_weights"], index=False)
    history.to_parquet(table_paths["training_history"], index=False)

    checkpoint_path = output_dir / "best_model.pt"
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": cpu_state,
            "config": asdict(config),
            "sources": SOURCES,
            "quality_columns": QUALITY_COLUMNS,
            "best_epoch": int(history.loc[history["is_best"], "epoch"].iloc[0]),
        },
        checkpoint_path,
    )

    output_paths = {
        "learned_candidate_embeddings": candidate_path,
        "learned_skill_embeddings": skill_path,
        **table_paths,
        "best_model": checkpoint_path,
    }
    manifest: dict[str, Any] = {
        "model_version": config.model_version,
        "config": asdict(config),
        "best_epoch": int(history.loc[history["is_best"], "epoch"].iloc[0]),
        "best_validation": history.loc[history["is_best"]].iloc[0].to_dict(),
        "inputs": {
            str(path): {"sha256": _sha256(path)} for path in input_paths
        },
        "outputs": {},
    }
    for name, path in output_paths.items():
        metadata: dict[str, Any] = {
            "path": path.name,
            "sha256": _sha256(path),
        }
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            metadata.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        manifest["outputs"][name] = metadata
    manifest_path = output_dir / "fusion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_data(
    embedding_dir: Path,
    feature_dir: Path,
    pair_dir: Path,
    config: FusionTrainingConfig,
) -> tuple[FusionData, list[Path]]:
    """Load all frozen embeddings, feature metadata, and mined supervision."""
    paths = {
        "candidate_tower_embeddings": embedding_dir
        / "candidate_tower_embeddings.npy",
        "candidate_tower_index": embedding_dir
        / "candidate_tower_embedding_index.parquet",
        "candidate_index": embedding_dir / "candidate_embedding_index.parquet",
        "skill_embeddings": embedding_dir / "skill_embeddings.npy",
        "skill_index": embedding_dir / "skill_embedding_index.parquet",
        "candidate_tower_features": feature_dir
        / "candidate_tower_features.parquet",
        "positive_pairs": pair_dir / "positive_pairs.parquet",
        "negative_pairs": pair_dir / "negative_pairs.parquet",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Missing fusion-training inputs:\n" + lines)
    data = prepare_fusion_data(
        np.load(paths["candidate_tower_embeddings"], allow_pickle=False),
        pd.read_parquet(paths["candidate_tower_index"]),
        pd.read_parquet(paths["candidate_index"]),
        pd.read_parquet(paths["candidate_tower_features"]),
        np.load(paths["skill_embeddings"], allow_pickle=False),
        pd.read_parquet(paths["skill_index"]),
        pd.read_parquet(paths["positive_pairs"]),
        pd.read_parquet(paths["negative_pairs"]),
        config,
    )
    return data, list(paths.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data, input_paths = load_data(
        args.embedding_dir, args.feature_dir, args.pair_dir, config
    )
    model, history, device = train_model(data, config)
    manifest = export_model(
        model,
        data,
        history,
        device,
        args.output_dir,
        config,
        input_paths,
    )
    best = manifest["best_validation"]
    print(f"Fusion model snapshot: {args.output_dir.resolve()}")
    print(f"Best epoch: {manifest['best_epoch']}")
    print(
        f"Validation Recall@{config.validation_cutoff}: {best['recall']:.4f}; "
        f"NDCG@{config.validation_cutoff}: {best['ndcg']:.4f}"
    )


if __name__ == "__main__":
    main()
