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
    Keep conditioning views fixed at their clean latent z0.

    Args:
      zt: Current latent state [B, V, C, h, w].
      z0: Clean input latent [B, V, C, h, w].
      cond_mask: Boolean mask [B, V].
        True  = known / conditioning view.
        False = target / generated view.

    Returns:
      Latent with conditioning views replaced by z0.
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

  B, V = zt.shape[:2]

  if cond_mask.shape != (B, V):
    raise ValueError(
      f"Expected Shape: {(B, V)}" 
      f"Got: {tuple(cond_mask.shape)}"
    )

  cond_mask = cond_mask.bool()
  mask = cond_mask[:, :, None, None, None]  # [B, V]

  return torch.where(mask, z0, zt)