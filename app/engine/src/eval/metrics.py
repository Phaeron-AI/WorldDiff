from __future__ import annotations

# Native Import(s)
import math

# Third Party Import(s)
import torch
from torch import Tensor

def psnr(
  prediction: Tensor,
  target: Tensor,
  *,
  max_val: float = 1.0
) -> float:
  if max_val <= 0:
    raise ValueError(f"max_value must be positive, got {max_val}")

  if prediction.shape != target.shape:
    raise ValueError(
      f"Prediction and Target shape must match."
      f"Predicted: {tuple(prediction.shape)}"
      f"Target: {tuple(target.shape)}"
    )

  mse = torch.mean((prediction.float() - target.float()) ** 2)

  if not torch.isfinite(mse):
    raise ValueError("MSE not finite")

  if mse.item() == 0:
    return float("inf")

  return float(10.0 * math.log10( (max_val * max_val) / mse.item()))

def pooled_mse(
  prediction: Tensor,
  target: Tensor
) -> tuple[float, int]:
  
  if prediction.shape != target.shape:
    raise ValueError(
      f"Prediction and Target shape must match."
      f"Predicted: {tuple(prediction.shape)}"
      f"Target: {tuple(target.shape)}"
    )

  error = (prediction.float() - target.float())

  squared_error_sum = float(torch.sum(error * error).item())

  count = prediction.numel()

  return squared_error_sum, count

def pooled_psnr(
  squared_error_sum: float,
  count: int,
  *,
  max_val: float = 1.0
) -> float:
  if count <= 0:
    raise ValueError(f"count must be positive")

  if max_val <= 0:
    raise ValueError(f"max_val must be positive")

  mse = squared_error_sum / count

  if mse == 0.0:
    return float("inf")

  return float(10.0 * math.log10( (max_val * max_val) / mse))