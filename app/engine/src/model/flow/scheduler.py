from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

class RectifiedFlowScheduler:
  def __init__(self, num_steps: int) -> None:
    self.num_steps = num_steps

  def timesteps(self, *, device: torch.device) -> Tensor:
    t = torch.linspace(
      1.0, 0.0, self.num_steps + 1, device=device
    )

    return t