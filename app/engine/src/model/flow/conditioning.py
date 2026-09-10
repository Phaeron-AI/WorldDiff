from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

def apply_conditioning(
  zt: Tensor,
  z0: Tensor,
  cond_mask: Tensor
) -> Tensor:
  """
    This is where we add conditioning mask: 
      - mv = 0 (for i/p views)
      - mv = 1 (for target views)
    
    mv = [B, V]
  """
  mask = cond_mask[:, :, None, None]  # [B, V]

  return torch.where(mask, z0, zt)