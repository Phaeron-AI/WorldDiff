from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

def apply_conditioning(
  zt: Tensor,
  z0: Tensor,
  cond_mask: Tensor
) -> Tensor:
  mask = cond_mask[:, :, None, None]

  return torch.where(mask, z0, zt)