"""The occupancy network used by the ablation.

The architecture is deliberately small. The paper does not aim to win a
segmentation benchmark; it measures how the *error-growth rate* changes as
modalities and the closed loop are added. The network therefore only has to be
capable enough to benefit from a better input, and small enough to train
several configurations on a laptop GPU.

The input channels are whatever :func:`src.models.bev.assemble_inputs`
produced for the selected modalities, which is how the modality ladder is
implemented without touching the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

#: number of BEV channels each modality contributes to the network input.
#: ``contact`` and ``memory`` are not sensors: they are the traversability
#: attribute plane written by Stage II and the map prior fed back by Stage III.
MODALITY_CHANNELS = {"lidar": 4, "radar": 1, "camera": 2, "contact": 1, "memory": 1}


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BevOccupancyNet(nn.Module):
    """A small U-Net over the BEV grid."""

    def __init__(self, in_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.pool = nn.MaxPool2d(2)
        self.encoder1 = ConvBlock(in_channels, c1)
        self.encoder2 = ConvBlock(c1, c2)
        self.encoder3 = ConvBlock(c2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.decoder2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.decoder1 = ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward_features(self, x: torch.Tensor,
                         severity_override: int | None = None) -> torch.Tensor:
        """The decoder output the occupancy head reads.

        Factored out so the detector of ``src.models.detection`` can put a second
        head on the same features, which is what makes the detection comparison a
        comparison of fusion structures rather than of two different tasks. The
        baseline has no severity input, so ``severity_override`` is accepted and
        ignored, keeping the signature the detector calls uniform across trunks.
        """
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(self.pool(encoded1))
        encoded3 = self.encoder3(self.pool(encoded2))
        decoded2 = self.decoder2(torch.cat([self.up2(encoded3), encoded2], dim=1))
        return self.decoder1(torch.cat([self.up1(decoded2), encoded1], dim=1))

    def occupancy_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Occupancy logits from the shared features."""
        return self.head(features)

    def fuse(self, x: torch.Tensor, severity_override: int | None = None
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The shared interface the detector uses: ``(features, auxiliary logits)``.

        The baseline has no auxiliary head, so the second element is ``None``,
        which is exactly what its detector does not need.
        """
        return self.forward_features(x, severity_override), None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.occupancy_from_features(self.forward_features(x))


def input_channels(modalities: tuple[str, ...]) -> int:
    """How many BEV channels a modality subset contributes."""
    return sum(MODALITY_CHANNELS[m] for m in modalities)


def build_model(
    modalities: tuple[str, ...],
    base_channels: int = 16,
    device: str | torch.device = "cuda",
) -> BevOccupancyNet:
    """Instantiate the network for one modality subset."""
    model = BevOccupancyNet(input_channels(modalities), base_channels)
    return model.to(device)


@dataclass
class TrainConfig:
    """Optimisation settings, kept small enough for a laptop GPU."""

    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    #: Weight on the severity-classification auxiliary loss. Consumed by the
    #: gated model in :mod:`src.models.gated`; the baseline ignores it.
    auxiliary_weight: float = 0.5


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice, which counteracts the class imbalance of occupancy."""
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=(2, 3)) + eps
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1.0 - numerator / denominator).mean()


def occupancy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus Dice."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


@torch.no_grad()
def bev_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Occupancy IoU over a batch, at a fixed decision threshold."""
    prediction = (torch.sigmoid(logits) > threshold).float()
    intersection = float((prediction * target).sum())
    union = float(prediction.sum() + target.sum()) - intersection
    return intersection / (union + eps)


def fit(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: TrainConfig | None = None,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, float]]:
    """Train on in-memory BEV tensors and return the per-epoch history."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model.to(config.device)

    x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    count = x.shape[0]

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(count)
        total = 0.0
        for start in range(0, count, config.batch_size):
            index = order[start:start + config.batch_size]
            batch_x = x[index].to(config.device)
            batch_y = y[index].to(config.device)
            optimiser.zero_grad(set_to_none=True)
            loss = occupancy_loss(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * index.numel()
        record = {"epoch": float(epoch), "train_loss": total / max(count, 1)}
        if validation is not None:
            val_x, val_y = validation
            model.eval()
            with torch.no_grad():
                logits = model(torch.as_tensor(np.asarray(val_x), dtype=torch.float32)
                               .to(config.device))
                record["val_iou"] = bev_iou(
                    logits,
                    torch.as_tensor(np.asarray(val_y), dtype=torch.float32).to(config.device))
        history.append(record)
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: TrainConfig | None = None,
) -> float:
    """Occupancy IoU on a held-out set."""
    config = config or TrainConfig()
    model.eval()
    logits = model(torch.as_tensor(np.asarray(inputs), dtype=torch.float32).to(config.device))
    return bev_iou(logits, torch.as_tensor(np.asarray(targets), dtype=torch.float32)
                   .to(config.device))
