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
  if zt.ndim != 5:
    raise ValueError(f"Expected: [B, V, C, h, w]; Got: {tuple(zt.shape)}")

  if zt.shape != z0.shape:
    raise ValueError(
      f"Both z0 and zt: [B, V, C, h, w]"
      f"z0 shape: {tuple(z0.shape)}"
      f"zt shape: {tuple(zt.shape)}" 
    )

  if cond_mask.ndim != 2:
    raise ValueError(f"Expected: [B, V]; Got: {tuple(cond_mask.shape)}")

  if cond_mask.shape[0] != z0.shape[0] or cond_mask.shape[1] != z0.shape[1]:
    raise ValueError(f"Expected Shape: [B, V]; Got: {tuple(cond_mask.shape)}")
  
  mask = cond_mask[:, :, None, None, None]  # [B, V]

  return torch.where(mask, z0, zt)