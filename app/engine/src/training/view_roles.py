from __future__ import annotations

# Thid Party Import(s)
import torch
from torch import Generator, Tensor

def sample_view_roles(B: int, V: int, *, p_drop: float, rng: Generator, device: torch.device | str) -> Tensor:
  if B < 1:
    raise ValueError(f"B must be >=1, got {B}")
  
  if V < 2:
    raise ValueError(f"V must be >=2, got {V}")

  if not 0.0 <= p_drop < 1.0:
    raise ValueError(
      f"p_drop must satisfy 0 <= p_drop < 1, got {p_drop}"
    )

  cond_mask_cpu = torch.zeros(
    B, V, dtype=torch.bool, device="cpu"
  )

  drop_draw = torch.rand(
    B, generator=rng, device="cpu"
  )

  drop_mask = drop_draw < p_drop
  non_drop_indices = torch.where(~drop_mask)[0]

  if non_drop_indices.numel() > 0:
    k_draw = torch.randint(
      low=1,
      high=V,
      size=(non_drop_indices.numel(),),
      generator=rng,
      device="cpu",
    )

    for i, batch_index in enumerate(non_drop_indices):
      k = int(k_draw[i].item())
      permutation = torch.randperm(
        V,
        generator=rng,
        device="cpu",
      )
      cond_mask_cpu[batch_index, permutation[:k]] = True

  return cond_mask_cpu.to(device=device)
