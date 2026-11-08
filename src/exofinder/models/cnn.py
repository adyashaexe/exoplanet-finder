"""Two-branch 1D CNN (global + local phase-folded views) with a small scalar-feature branch.

Follows the AstroNet idea: the *global* view shows the whole orbit (secondary
eclipses, variability), the *local* view zooms on the transit (U vs V shape).
BLS-derived scalars (period, depth, SNR, odd/even difference...) are fused in
the classification head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import AUX_FEATURES


def _block(cin: int, cout: int, k: int = 5, pool: int = 5, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
        nn.Conv1d(cout, cout, k, padding=k // 2),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool, stride=stride, padding=pool // 2),
    )


class _Branch(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        blocks, cin = [], 1
        for c in channels:
            blocks.append(_block(cin, c))
            cin = c
        self.net = nn.Sequential(*blocks)
        self.out_dim = cin * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))
        return torch.cat([h.mean(-1), h.amax(-1)], dim=1)  # length-agnostic pooling


class ExoNet(nn.Module):
    def __init__(
        self,
        n_aux: int = len(AUX_FEATURES),
        global_channels: tuple[int, ...] = (16, 32, 64, 128),
        local_channels: tuple[int, ...] = (16, 32, 64),
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_args = dict(
            n_aux=n_aux,
            global_channels=tuple(global_channels),
            local_channels=tuple(local_channels),
            hidden=hidden,
            dropout=dropout,
        )
        self.global_branch = _Branch(tuple(global_channels))
        self.local_branch = _Branch(tuple(local_channels))
        self.aux_branch = nn.Sequential(nn.Linear(n_aux, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(self.global_branch.out_dim + self.local_branch.out_dim + 32, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, g: torch.Tensor, loc: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.global_branch(g), self.local_branch(loc), self.aux_branch(aux)], dim=1)
        return self.head(z).squeeze(-1)  # logits


def save_checkpoint(path: str | Path, model: ExoNet, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_args": model.model_args, **meta}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ExoNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ExoNet(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "model_args")}
    return model, meta


def scale_aux(aux: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((aux - mean) / np.where(std < 1e-6, 1.0, std)).astype(np.float32)
