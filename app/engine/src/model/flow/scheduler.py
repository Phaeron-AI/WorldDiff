from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

class RectifiedFlowScheduler:
  def __init__(
    self, 
    num_steps: int, 
    *, 
    device: torch.device, 
    dtype: torch.dtype = torch.float32
  ) -> None:
    if num_steps <= 0:
      raise ValueError(
        f"num_steps must be positive, got {num_steps}"
      )
    
    self.num_steps = num_steps
    self.device = device
    self.dtype = dtype

    self.timesteps = torch.linspace(
      1.0,
      0.0,
      num_steps + 1,
      device=device,
      dtype=dtype
    )

  def __len__(self) -> int:
    return self.num_steps

  def step_times(self) -> Tensor:
    """
      Return the model-evaluation times.

      For N steps:

        [1, 1-dt, ..., dt]

      These are the left endpoints of the reverse-time
      Euler integration intervals.
    """
    return self.timesteps[:-1]

  def step_size(self) -> Tensor:
    """
      Return the positive magnitude of each integration step
    """
    return self.timesteps[:-1] - self.timesteps[1]