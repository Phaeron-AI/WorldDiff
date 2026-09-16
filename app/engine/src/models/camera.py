from __future__ import annotations

# third Party Import(s)
import torch
from torch import Tensor

def rescale_intrinsics(intrinsics: Tensor, factor: int) -> Tensor:
  if factor <= 0:
    raise ValueError(f"Expected factor > 0; Got: {factor}")

  s = 1.0 / factor

  K = intrinsics.clone()
  K[..., 0, 0] *= s; K[..., 0, 2] *= s
  K[..., 1, 1] *= s; K[..., 1, 2] *= s

  return K