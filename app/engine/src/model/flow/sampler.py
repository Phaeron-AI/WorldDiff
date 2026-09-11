from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

# Local Import(s)
from src.model.flow.interpolant import sample_noise

def euler_step(
  z: Tensor,
  velocity: Tensor,
  dt: float
) -> Tensor:
  """
    Euler Step Method:
      zt-del_t = zt - del_t * v(zt, t)
  """

  if z.ndim != 5:
    raise ValueError(f"Expected: [B, V, C, h, w]; Got: {tuple(z.shape)}")

  if z.shape != velocity.shape:
    raise ValueError(
      f"Velocity shape must match z shape."
      f"Got z={tuple(z.shape)}, "
      f"velocity={tuple(velocity.shape)}"
    )

  if dt <= 0.0:
    raise ValueError(
      f"dt must be positive, got {dt}"
    )

  return z - dt * velocity

def init_target_noise(z0: Tensor, cond_mask: Tensor, *, rng: torch.Generator | None = None) -> Tensor:
  if z0.ndim != 5:
    raise ValueError(
      f"z0 must have shape [B,V,C,h,w], "
      f"got {tuple(z0.shape)}"
    )

  B, V = z0.shape[:2]

  if cond_mask.shape != (B, V):
    raise ValueError(
      f"cond_mask must have shape {(B, V)}, "
      f"got {tuple(cond_mask.shape)}"
    )

  cond_mask = cond_mask.bool()

  z = sample_noise(z0, rng=rng)

  mask = cond_mask[:, :, None, None, None]

  return torch.where(mask, z0, z)

def sample(
  model, 
  z0: Tensor, 
  cond_mask: Tensor, 
  *, 
  num_steps: int = 20, 
  rng: torch.Generator | None = None
) -> Tensor:
  if num_steps <= 0:
    raise ValueError(f"num_steps must be positive, Got: {num_steps}")
  
  z = init_target_noise(z0, cond_mask, rng=rng)

  dt = 1.0 / num_steps
  mask = cond_mask.bool()[:, :, None, None, None]

  for step in range(num_steps):
    t_b = 1.0 - step * dt

    t = torch.full(
      (z.shape[0],),
      t_b,
      device=z.device,
      dtype=z.dtype,
    )

    velocity = model(z, t, cond_mask)

    if velocity.shape != z.shape:
      raise RuntimeError(
        f"Model returned velocity with shape "
        f"{tuple(velocity.shape)}, expected "
        f"{tuple(z.shape)}"
      )

    z = euler_step(z, velocity, dt)

    z = torch.where(mask, z0, z)

  return z
