from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

# Local Import(s)
from src.models.dit.model import MultiViewDiT
from src.models.flow import (
  interpolant,
  apply_conditioning,
  masked_flow_matching_loss,
)
def train_overfit(
  model: MultiViewDiT,
  z0: Tensor,
  rays: Tensor,
  cond_mask: Tensor,
  num_steps: int = 5000,
  lr: float = 1e-4
) -> None:
  pass