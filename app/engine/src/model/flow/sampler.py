from __future__ import annotations

# Third Party Import(s)
from torch import Tensor

def euler_step(
  z: Tensor,
  velocity: Tensor,
  dt: float
) -> Tensor:
  
  if z.ndim != 5:
    raise ValueError(f"Expected: [B, V, C, h, w]; Got: {tuple(z.shape)}")

  if velocity.shape != z.shape:
    raise ValueError(f"Velocity shape must match z shape.")

  return z - dt * velocity