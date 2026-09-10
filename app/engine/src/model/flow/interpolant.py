from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor, Generator

def sample_noise(z0: Tensor, *, rng: Generator | None = None) -> Tensor:
  """
    Objective here is to sample noise from a Gaussian Distribution:
      - eps = N(0, I)
  """
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
  return (1.0 - t) * z0 + t * eps


def velocity_target(z0: Tensor, eps: Tensor) -> Tensor:
  """
    On differentiating the Linear Interpolant w.r.t t
    d(zt)/dt = eps - z0 = u
  """
  return eps - z0