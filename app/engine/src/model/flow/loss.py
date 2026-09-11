from __future__ import annotations

# Third Party Import(s)
from torch import Tensor

def masked_flow_matching_mask(
  v_pred: Tensor,
  target: Tensor,
  target_mask: Tensor
) -> Tensor:
  mean_squared_error = (v_pred - target) ** 2

  mask = target_mask[:, :, None, None, None]

  error = mean_squared_error * mask
  denom = mask.sum() * v_pred.shape[2] * v_pred.shape[3] * v_pred.shape[4]

  return error.sum() / denom