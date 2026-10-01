from __future__ import annotations

import math

import torch
from torch import Tensor


def psnr(
  prediction: Tensor,
  target: Tensor,
  *,
  max_val: float = 1.0,
) -> float:
  if max_val <= 0:
    raise ValueError(
      f"max_val must be positive, got {max_val}"
    )

  if prediction.shape != target.shape:
    raise ValueError(
      "Prediction and Target shape must match. "
      f"Predicted: {tuple(prediction.shape)} "
      f"Target: {tuple(target.shape)}"
    )

  prediction = prediction.float().clamp(0.0, 1.0)
  target = target.float().clamp(0.0, 1.0)

  mse = torch.mean(
    (prediction - target) ** 2
  )

  if not torch.isfinite(mse):
    raise ValueError("MSE not finite")

  mse_value = mse.item()

  if mse_value == 0.0:
    return float("inf")

  return float(
    10.0
    * math.log10(
      (max_val * max_val) / mse_value
    )
  )


def pooled_mse(
  prediction: Tensor,
  target: Tensor,
) -> tuple[float, int]:
  if prediction.shape != target.shape:
    raise ValueError(
      "Prediction and Target shape must match. "
      f"Predicted: {tuple(prediction.shape)} "
      f"Target: {tuple(target.shape)}"
    )

  prediction = prediction.float().clamp(0.0, 1.0)
  target = target.float().clamp(0.0, 1.0)

  error = prediction - target

  squared_error_sum = float(
    torch.sum(error * error).item()
  )

  count = prediction.numel()

  return squared_error_sum, count


def pooled_psnr(
  squared_error_sum: float,
  count: int,
  *,
  max_val: float = 1.0,
) -> float:
  if count <= 0:
    raise ValueError("count must be positive")

  if max_val <= 0:
    raise ValueError(
      "max_val must be positive"
    )

  mse = squared_error_sum / count

  if mse == 0.0:
    return float("inf")

  return float(
    10.0
    * math.log10(
      (max_val * max_val) / mse
    )
  )