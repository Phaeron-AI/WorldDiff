from __future__ import annotations

# Third Party Import(s)

import torch

from torch import Tensor

# Local Import(s)

from src.model.flow.interpolant import sample_noise
from src.model.flow.scheduler import RectifiedFlowScheduler


def euler_step(
  z: Tensor,
  velocity: Tensor,
  dt: Tensor | float
) -> Tensor:
  """
  Euler Step Method:

    zt-del_t = zt - del_t * v(zt, t)
  """
  if z.ndim != 5:
    raise ValueError(
      f"Expected: [B, V, C, h, w]; Got: {tuple(z.shape)}"
    )

  if z.shape != velocity.shape:
    raise ValueError(
      f"Velocity shape must match z shape. "
      f"Got z={tuple(z.shape)}, "
      f"velocity={tuple(velocity.shape)}"
    )

  if dt <= 0.0:
    raise ValueError(
      f"dt must be positive, got {dt}"
    )

  return z - dt * velocity


def init_target_noise(
  z0: Tensor,
  cond_mask: Tensor,
  *,
  rng: torch.Generator | None = None
) -> Tensor:
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
    raise ValueError(
      f"num_steps must be positive, Got: {num_steps}"
    )

  scheduler = RectifiedFlowScheduler(
    num_steps,
    device=z0.device,
    dtype=z0.dtype,
  )

  # Initialize target views with Gaussian noise
  # and keep conditioning views at their clean latent.
  z = init_target_noise(
    z0,
    cond_mask,
    rng=rng,
  )

  cond_mask = cond_mask.bool()

  mask = cond_mask[:, :, None, None, None]

  timesteps = scheduler.step_times()
  step_sizes = scheduler.step_size()

  for step in range(num_steps):
    # Scheduler provides the model evaluation time.
    # Expand the scalar timestep to one value per batch item.
    t = timesteps[step].expand(z.shape[0])

    # Scheduler provides the positive step magnitude.
    dt = step_sizes[step]

    velocity = model(
      z,
      t,
      cond_mask=cond_mask,
    )

    if velocity.shape != z.shape:
      raise RuntimeError(
        f"Model returned velocity with shape "
        f"{tuple(velocity.shape)}, expected "
        f"{tuple(z.shape)}"
      )

    z = euler_step(
      z,
      velocity,
      dt,
    )

    # Conditioning views remain exactly equal to z0.
    z = torch.where(
      mask,
      z0,
      z,
    )

  return z