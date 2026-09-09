"""Reliability-aware fusion and skill projection model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ReliabilityAwareFusion(nn.Module):
    """Fuse five evidence towers with metadata-conditioned reliability gates."""

    def __init__(
        self,
        input_dimension: int = 384,
        output_dimension: int = 128,
        quality_dimension: int = 6,
        source_count: int = 5,
        gate_hidden_dimension: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.quality_dimension = quality_dimension
        self.source_count = source_count

        self.shared_projection = nn.Linear(
            input_dimension, output_dimension, bias=False
        )
        self.skill_projection = nn.Linear(
            input_dimension, output_dimension, bias=False
        )
        self.source_residuals = nn.ModuleList(
            nn.Linear(input_dimension, output_dimension, bias=False)
            for _ in range(source_count)
        )
        self.gate_adjustment = nn.Sequential(
            nn.Linear(quality_dimension + source_count, gate_hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dimension, 1),
        )
        self.register_buffer("source_identity", torch.eye(source_count))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start from a shared geometry and the supplied reliability scores."""
        nn.init.orthogonal_(self.shared_projection.weight)
        with torch.no_grad():
            self.skill_projection.weight.copy_(self.shared_projection.weight)
        for residual in self.source_residuals:
            nn.init.zeros_(residual.weight)
        nn.init.xavier_uniform_(self.gate_adjustment[0].weight)
        nn.init.zeros_(self.gate_adjustment[0].bias)
        nn.init.zeros_(self.gate_adjustment[-1].weight)
        nn.init.zeros_(self.gate_adjustment[-1].bias)

    def encode_candidates(
        self,
        tower_embeddings: torch.Tensor,
        quality_features: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized candidate vectors and interpretable tower weights."""
        if tower_embeddings.ndim != 3:
            raise ValueError("tower_embeddings must have shape [batch, source, dim]")
        if tower_embeddings.shape[1:] != (
            self.source_count,
            self.input_dimension,
        ):
            raise ValueError("tower embedding shape does not match model dimensions")
        if quality_features.shape != (
            len(tower_embeddings),
            self.source_count,
            self.quality_dimension,
        ):
            raise ValueError("quality feature shape does not match model dimensions")
        if availability.shape != tower_embeddings.shape[:2]:
            raise ValueError("availability shape must match batch and source axes")
        available = availability.bool()
        if (~available.any(dim=1)).any():
            raise ValueError("every candidate must have at least one available tower")

        shared = self.shared_projection(tower_embeddings)
        residuals = torch.stack(
            [
                adapter(tower_embeddings[:, source])
                for source, adapter in enumerate(self.source_residuals)
            ],
            dim=1,
        )
        projected_towers = F.normalize(shared + residuals, dim=-1)

        source_features = self.source_identity.unsqueeze(0).expand(
            len(tower_embeddings), -1, -1
        )
        gate_input = torch.cat([quality_features, source_features], dim=-1)
        learned_adjustment = self.gate_adjustment(gate_input).squeeze(-1)
        reliability = quality_features[..., 0].clamp_min(1e-6)
        gate_logits = reliability.log() + learned_adjustment
        gate_logits = gate_logits.masked_fill(~available, float("-inf"))
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(gate_weights.unsqueeze(-1) * projected_towers, dim=1)
        return F.normalize(fused, dim=-1), gate_weights

    def encode_skills(self, skill_embeddings: torch.Tensor) -> torch.Tensor:
        """Project frozen skill vectors into the learned shared space."""
        if skill_embeddings.ndim != 2:
            raise ValueError("skill_embeddings must have shape [skills, dim]")
        if skill_embeddings.shape[1] != self.input_dimension:
            raise ValueError("skill embedding dimension does not match model")
        return F.normalize(self.skill_projection(skill_embeddings), dim=-1)


def weighted_contrastive_loss(
    candidate_vectors: torch.Tensor,
    positive_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
    sample_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross-entropy contrastive loss with the positive at logit position zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive_logits = torch.sum(candidate_vectors * positive_vectors, dim=1)
    negative_logits = torch.einsum("bd,bnd->bn", candidate_vectors, negative_vectors)
    logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
    targets = torch.zeros(len(candidate_vectors), dtype=torch.long, device=logits.device)
    losses = F.cross_entropy(logits / temperature, targets, reduction="none")
    weights = sample_weights.clamp_min(0)
    return torch.sum(losses * weights) / weights.sum().clamp_min(1e-8)

