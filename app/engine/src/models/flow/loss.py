from __future__ import annotations

import torch
from torch import Tensor


def _masked_flow_matching_loss_per_sample(
  v_pred: Tensor,
  target: Tensor,
  target_mask: Tensor,
) -> Tensor:
  if v_pred.ndim != 5:
    raise ValueError(
      "v_pred must have shape [B, V, C, h, w], "
      f"got ndim={v_pred.ndim}"
    )

  if target.shape != v_pred.shape:
    raise ValueError(
      f"target shape {target.shape} must match "
      f"v_pred shape {v_pred.shape}"
    )

  if target_mask.ndim != 2:
    raise ValueError(
      "target_mask must have shape [B, V], "
      f"got ndim={target_mask.ndim}"
    )

  B, V, C, h, w = v_pred.shape

  if target_mask.shape != (B, V):
    raise ValueError(
      f"target_mask shape {target_mask.shape} must be "
      f"(B, V)=({B}, {V})"
    )

  target_mask = target_mask.bool()

  target_views_per_sample = target_mask.sum(dim=1)

  if torch.any(target_views_per_sample == 0):
    bad_samples = torch.where(
      target_views_per_sample == 0
    )[0].tolist()

    raise ValueError(
      "Every sample must have at least one target view; "
      f"no target views for samples {bad_samples}"
    )

  mean_squared_error = (
    v_pred.float() - target.float()
  ) ** 2

  per_view_mse = mean_squared_error.mean(
    dim=(2, 3, 4)
  )

  masked_mse = (
    per_view_mse * target_mask
  )

  per_sample_target_mean = (
    masked_mse.sum(dim=1)
    / target_views_per_sample.to(
      dtype=masked_mse.dtype
    )
  )

  return per_sample_target_mean


def masked_flow_matching_loss(
  v_pred: Tensor,
  target: Tensor,
  target_mask: Tensor,
) -> Tensor:
  per_sample_loss = (
    _masked_flow_matching_loss_per_sample(
      v_pred,
      target,
      target_mask,
    )
  )

  return per_sample_loss.mean()