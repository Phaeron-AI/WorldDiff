from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor, Generator

def sample_noise(z0: Tensor, *, rng: Generator | None = None) -> Tensor:
  """
    Objective here is to sample noise from a Gaussian Distribution:
      - eps = N(0, I)
  """
  if z0.ndim != 5:
    raise ValueError(f"Expected: [B, V, C, h, w]; Got: {tuple(z0.shape)}")
  
  return torch.randn(
    z0.shape,
    device=z0.device,
    dtype=z0.dtype,
    generator=rng
  )

def sample_time(batch_size: int, *, device: torch.device, rng: Generator | None = None) -> Tensor:
  """
    Here we sample timestep from a Uniform Distribution:
      - t = U(0, I)
  """
  if batch_size <= 0:
    raise ValueError(f"batch_size cannot be negative, got: {batch_size}")

  return torch.rand(
    batch_size,
    device=device,
    generator=rng
  )

def linear_interpolant(z0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
  """
    here we get the interpolant at time t:
      - zt = (1-t) * z0 + t * eps

    The order is something like this:
      - z0 --> zt --> eps

      where,
        zt at (t=0) = z0
        zt at (t=1) = eps
  """
  if z0.ndim != 5:
    raise ValueError(
      f"z0 must have shape [B, V, C, h, w], "
      f"got {tuple(z0.shape)}"
    )
  
  if z0.shape != eps.shape:
    raise ValueError(
      f"eps and z0 must be of the same shape"
      f"Got: eps: {tuple(eps.shape)}; z0: {tuple(z0.shape)}"
    )

  if t.ndim != 1:
    raise ValueError(f"Expected: [B,]; got: {tuple(t.shape)}")

  if t.shape[0] != z0.shape[0]:
    raise ValueError(
      f"t batch size must match z0. "
      f"Got t={t.shape[0]}, z0={z0.shape[0]}"
    )

  if torch.any(t < 0.0) or torch.any(t > 1.0):
    raise ValueError(
      "t must lie in the interval [0, 1]"
    )

  t = t[:, None, None, None, None]

  return (1.0 - t) * z0 + t * eps


def velocity_target(z0: Tensor, eps: Tensor) -> Tensor:
  """
    On differentiating the Linear Interpolant w.r.t t
    d(zt)/dt = eps - z0 = u
  """
  if z0.ndim != 5:
    raise ValueError(
      f"z0 must have shape [B, V, C, h, w], "
      f"got {tuple(z0.shape)}"
    )
  
  if z0.shape != eps.shape:
    raise ValueError(
      f"eps and z0 must be of the same shape"
      f"Got: eps: {tuple(eps.shape)}; z0: {tuple(z0.shape)}"
    )

  return eps - z0