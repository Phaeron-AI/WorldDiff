from __future__ import annotations

# Third Party Import(s)
import torch

from torch import Tensor

# Local Import(s)
from src.models.dit.model import MultiViewDiT
from src.models.flow.sampler import sample


@torch.no_grad()
def reconstruct(
  model: MultiViewDiT,
  z0: Tensor,
  rays: Tensor,
  cond_mask: Tensor,
  *,
  num_steps: int = 20,
  rng: torch.Generator | None = None,
) -> Tensor:
  model.eval()

  cond_mask = cond_mask.bool()
  target_mask = ~cond_mask

  reconstructed = sample(
    model=model,
    z0=z0,
    rays=rays,
    cond_mask=cond_mask,
    num_steps=num_steps,
    rng=rng,
  )

  target_indices = target_mask.nonzero(as_tuple=True)

  prediction = reconstructed[target_indices]
  target = z0[target_indices]

  mse = torch.mean(
    (prediction - target) ** 2
  )

  return mse