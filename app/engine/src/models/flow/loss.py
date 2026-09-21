from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

def masked_flow_matching_loss(
  v_pred: Tensor,
  target: Tensor,
  target_mask: Tensor
) -> Tensor:
  if v_pred.ndim != 5:
    raise ValueError(
      f"v_pred must have shape [B, V, C, h, w], "
      f"got {tuple(v_pred.shape)}"
    )

  if target.shape != v_pred.shape:
    raise ValueError(
      f"target must have the same shape as v_pred. "
      f"Got v_pred={tuple(v_pred.shape)}, "
      f"target={tuple(target.shape)}"
    )

  if target_mask.ndim != 2:
    raise ValueError(
      f"target_mask must have shape [B, V], "
      f"got {tuple(target_mask.shape)}"
    )

  B, V, C, h, w = v_pred.shape

  if target_mask.shape != (B, V):
    raise ValueError(
      f"target_mask must have shape {(B, V)}, "
      f"got {tuple(target_mask.shape)}"
    )

  target_mask = target_mask.bool()

  target_views_per_sample = target_mask.sum(dim=1)

  if torch.any(target_views_per_sample == 0):
    bad_samples = torch.where(
      target_views_per_sample == 0
    )[0].tolist()

    raise ValueError(
      "Flow-matching loss received samples with no target "
      f"views: batch indices {bad_samples}. "
      "Each training sample must contain at least one "
      "target view."
    )
  
  mean_squared_error = (v_pred - target) ** 2

  per_view_mse = mean_squared_error.mean(dim=(2, 3, 4))

  masked_mse = per_view_mse * target_mask

  per_sample_target_mean = (
    masked_mse.sum(dim=1)
    / target_views_per_sample.to(dtype=masked_mse.dtype)
  )

  return per_sample_target_mean.mean()