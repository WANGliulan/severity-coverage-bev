"""Reliability-gated cross-modal fusion for BEV occupancy under degradation.

Why this module exists
----------------------
The baseline (:class:`~src.models.network.BevOccupancyNet`) receives every
selected modality concatenated as input channels, so the network itself has to
work out which of those channels is currently trustworthy. That is exactly the
information degradation destroys: a rain-blurred camera plane and a clean one are
different data, and a model trained on mild degradation has no explicit cue that
the camera channel has gone bad.

This module makes reliability explicit and learnable:

    one stream per modality
      -> per-modality severity head (auxiliary, supervised only while training)
      -> learnable (modality x severity) reliability table
      -> softmax gate over modalities
      -> gated sum of the streams -> shared U-Net decoder -> occupancy logits

The auxiliary supervision is the point. During simulation the injected severity
is known, so the severity heads get free labels and the reliability table is
anchored to something meaningful; at test time on real weather nobody supplies
that label, so the heads must infer it from the input. That is what makes the
gate a mechanism rather than an oracle.

Capacity is deliberately matched to the baseline -- one stream per modality with
the same three resolutions, and the same shared decoder -- so a comparison
isolates the fusion mechanism rather than the parameter count.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .network import (MODALITY_CHANNELS, ConvBlock, TrainConfig, bev_iou,
                      occupancy_loss)


def channel_slices(modalities: tuple[str, ...]) -> dict[str, slice]:
    """Which input channels belong to which modality.

    ``assemble_inputs`` concatenates the modality planes in the order of the
    modality tuple, so the slices follow from the per-modality channel counts.
    """
    slices: dict[str, slice] = {}
    offset = 0
    for name in modalities:
        width = MODALITY_CHANNELS[name]
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


class ReliabilityTable(nn.Module):
    """A learnable (modality x severity) table of raw trust scores.

    Each modality predicts its own severity class from its own features; the
    table then says how much that modality is worth *at that severity*. Keeping
    the table separate from the encoders is what makes the learned behaviour
    readable: the trained table can be printed and checked against the measured
    retention curve of each modality.

    It returns raw scores, not probabilities, so the caller can add the
    feature-based term of :class:`GatedFusionNet` before the softmax.
    """

    def __init__(self, num_modalities: int, num_levels: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.zeros(num_modalities, num_levels))

    def forward(self, severity_probabilities: torch.Tensor) -> torch.Tensor:
        """Map ``(N, M, K)`` severity probabilities to ``(N, M)`` raw scores."""
        return torch.einsum("nmk,mk->nm", severity_probabilities, self.table)


class SensorStream(nn.Module):
    """The three-resolution encoder of one modality, mirroring the baseline."""

    def __init__(self, in_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level1 = self.encoder1(x)
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        return level1, level2, level3


class GatedFusionNet(nn.Module):
    """Per-modality streams, a learned reliability gate, and the shared decoder."""

    def __init__(
        self,
        modalities: tuple[str, ...],
        num_levels: int = 5,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("at least one modality is required")
        self.modalities = tuple(modalities)
        self.slices = channel_slices(self.modalities)
        self.num_levels = int(num_levels)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.streams = nn.ModuleDict(
            {name: SensorStream(MODALITY_CHANNELS[name], base_channels)
             for name in self.modalities})
        self.severity_heads = nn.ModuleDict(
            {name: nn.Linear(c3, self.num_levels) for name in self.modalities})
        # A direct, feature-based trust score per modality, added to the
        # severity-table term. The table answers "what is a camera worth in rain
        # in general"; this answers "what is *this* camera frame worth right now",
        # which is the part a table cannot know and the part degradation changes.
        self.reliability_heads = nn.ModuleDict(
            {name: nn.Linear(c3, 1) for name in self.modalities})
        self.reliability = ReliabilityTable(len(self.modalities), self.num_levels)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.decoder2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.decoder1 = ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def _fuse(self, x: torch.Tensor, severity_override: int | None = None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(decoded features, gate weights, severity logits)``.

        The body of :meth:`forward`, factored out so the detector can read the
        same features the occupancy head does without duplicating the gate.
        ``severity_override`` feeds a *known* severity to the reliability table
        instead of the predicted one. It exists for the oracle experiment -- how
        much of the degradation-induced error is recoverable if the severity were
        known perfectly -- which bounds what any severity estimator can win.
        """
        levels = {name: stream(x[:, self.slices[name]])
                  for name, stream in self.streams.items()}
        severity = torch.stack(
            [self.severity_heads[name](levels[name][2].mean(dim=(2, 3)))
             for name in self.modalities], dim=1)
        table_input = torch.softmax(severity, dim=-1)
        if severity_override is not None:
            table_input = torch.zeros_like(table_input)
            table_input[:, :, int(severity_override)] = 1.0
        scores = self.reliability(table_input)
        scores = scores + torch.stack(
            [self.reliability_heads[name](levels[name][2].mean(dim=(2, 3))).squeeze(1)
             for name in self.modalities], dim=1)
        gates = torch.softmax(scores, dim=1)
        fused = [
            sum(gates[:, index].view(-1, 1, 1, 1) * levels[name][resolution]
                for index, name in enumerate(self.modalities))
            for resolution in range(3)
        ]
        decoded2 = self.decoder2(torch.cat([self.up2(fused[2]), fused[1]], dim=1))
        decoded1 = self.decoder1(torch.cat([self.up1(decoded2), fused[0]], dim=1))
        return decoded1, gates, severity

    def forward_features(self, x: torch.Tensor,
                         severity_override: int | None = None) -> torch.Tensor:
        """The decoder output the occupancy head reads, for the detector to share."""
        return self._fuse(x, severity_override)[0]

    def occupancy_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Occupancy logits from the shared features."""
        return self.head(features)

    def fuse(self, x: torch.Tensor, severity_override: int | None = None
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The shared interface the detector uses: ``(features, severity logits)``.

        The severity logits come along because they carry the auxiliary loss that
        anchors the reliability table; a detector trained without it would be
        training a different gate from the one the occupancy tables describe.
        """
        features, _, severity = self._fuse(x, severity_override)
        return features, severity

    def forward(self, x: torch.Tensor,
                severity_override: int | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(occupancy logits, gate weights, severity logits)``."""
        features, gates, severity = self._fuse(x, severity_override)
        return self.head(features), gates, severity


def fusion_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    severity: torch.Tensor,
    levels: int | np.ndarray | torch.Tensor,
    auxiliary_weight: float = 0.5,
) -> torch.Tensor:
    """Occupancy loss plus the severity classification that anchors the table.

    ``levels`` may be a single severity (one level per batch) or one level per
    sample, because training mixes levels 0--2 in one tensor.
    """
    task = occupancy_loss(logits, target)
    labels = levels if torch.is_tensor(levels) else torch.as_tensor(
        levels, dtype=torch.long, device=severity.device)
    if labels.ndim == 0:
        labels = labels.reshape(1).expand(severity.shape[0])
    if labels.ndim == 1:
        # One severity for the whole sample: every modality was degraded alike.
        labels = labels.unsqueeze(1).expand(-1, severity.shape[1])
    # With shape (N, M) every modality head is told *its own* severity, which is
    # the supervision the asymmetric augmentation depends on: a camera pushed to
    # level 3 is labelled 3 even on a frame whose LiDAR stayed clean.
    auxiliary = sum(nn.functional.cross_entropy(severity[:, index, :], labels[:, index])
                    for index in range(severity.shape[1])) / severity.shape[1]
    return task + auxiliary_weight * auxiliary


def fit_gated(
    model: GatedFusionNet,
    inputs: np.ndarray,
    targets: np.ndarray,
    levels: int | np.ndarray,
    config: TrainConfig | None = None,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, float]]:
    """Train the gated model, returning the per-epoch history.

    Kept separate from :func:`src.models.network.fit` because this forward pass
    returns three tensors, only one of which the baseline's loop expects.
    """
    config = config or TrainConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.manual_seed(config.seed)
    x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    level_values = np.asarray(levels, dtype=np.int64)
    if level_values.ndim == 0:
        level_values = np.full(x.shape[0], int(level_values), dtype=np.int64)
    if level_values.ndim == 1:
        # Broadcast a per-sample severity across the modalities; a (N, M) array is
        # passed through unchanged, which is what the asymmetric augmentation
        # supplies.
        level_values = np.repeat(level_values[:, None], len(model.modalities), axis=1)
    level_tensor = torch.as_tensor(level_values, dtype=torch.long)
    # AdamW with the same weight decay as the baseline's training loop. The two
    # loops originally disagreed here (Adam against AdamW), which is an accident
    # of how they were written rather than a design choice: with that difference
    # in place, comparing the two fusion mechanisms would also be comparing two
    # optimisers, and the gate would start from a disadvantage that has nothing
    # to do with gating.
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(x.shape[0])
        running = 0.0
        for start in range(0, x.shape[0], config.batch_size):
            index = order[start:start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits, _, severity = model(x[index].to(device))
            loss = fusion_loss(logits, y[index].to(device), severity,
                               level_tensor[index].to(device),
                               auxiliary_weight=config.auxiliary_weight
                               if hasattr(config, "auxiliary_weight") else 0.5)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * index.numel()
        row = {"epoch": float(epoch), "loss": running / max(x.shape[0], 1)}
        if validation is not None:
            row["val_iou"] = evaluate_gated(model, validation[0], validation[1])
        history.append(row)
    return history


@torch.no_grad()
def evaluate_gated(model: GatedFusionNet, inputs: np.ndarray, targets: np.ndarray,
                   severity_override: int | None = None) -> float:
    """Occupancy IoU of the gated model, defined exactly as for the baseline."""
    device = next(model.parameters()).device
    model.eval()
    logits, _, _ = model(torch.as_tensor(np.asarray(inputs), dtype=torch.float32,
                                         device=device),
                         severity_override=severity_override)
    return bev_iou(logits, torch.as_tensor(np.asarray(targets), dtype=torch.float32,
                                           device=device))


@torch.no_grad()
def gate_table(model: GatedFusionNet) -> dict[str, list[float]]:
    """The learned trust table, ready to print beside the retention ladder."""
    model.eval()
    rows = model.reliability.table.detach().cpu().numpy()
    return {name: [float(value) for value in rows[index]]
            for index, name in enumerate(model.modalities)}


@torch.no_grad()
def mean_gates(model: GatedFusionNet, inputs: np.ndarray,
               severity_override: int | None = None) -> dict[str, float]:
    """Average gate per modality over a batch: the readable summary of the gate."""
    device = next(model.parameters()).device
    model.eval()
    _, gates, _ = model(torch.as_tensor(np.asarray(inputs), dtype=torch.float32,
                                        device=device),
                        severity_override=severity_override)
    return {name: float(gates[:, index].mean())
            for index, name in enumerate(model.modalities)}