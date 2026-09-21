from __future__ import annotations

import torch
from torch import Tensor

from src.models.dit.model import MultiViewDiT
from src.training.ema import EMA
from src.training.lr_schedule import learning_rate
from src.training.step import train_step


def train_overfit(
  model: MultiViewDiT,
  z0: Tensor,
  rays: Tensor,
  cond_mask: Tensor,
  *,
  num_steps: int = 5000,
  lr: float = 1e-4,
  G_noise: torch.Generator | None = None,
  G_role: torch.Generator | None = None,
  ema: EMA | None = None,
  use_lr_schedule: bool = False,
  warmup_steps: int = 1000,
  max_grad_norm: float | None = 1.0,
) -> list[float]:
  if num_steps < 1:
    raise ValueError(
      f"num_steps must be >= 1, got {num_steps}"
    )

  if G_noise is None:
    G_noise = torch.Generator(
      device=z0.device
    )

  if G_role is None:
    G_role = torch.Generator(
      device="cpu"
    )

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
  )

  losses: list[float] = []

  successful_steps = 0

  for step in range(num_steps):
    if use_lr_schedule:
      current_lr = learning_rate(
        step=successful_steps,
        max_lr=lr,
        warmup_steps=warmup_steps,
        total_steps=num_steps,
      )
    else:
      current_lr = lr

    result = train_step(
      model=model,
      optimizer=optimizer,
      z0=z0,
      rays=rays,
      G_noise=G_noise,
      G_role=G_role,
      cond_mask=cond_mask,
      ema=ema,
      max_grad_norm=max_grad_norm,
      learning_rate=current_lr,
    )

    if result.skipped:
      continue

    losses.append(result.loss)
    successful_steps += 1

    if step % 100 == 0:
      print(
        f"step={step:05d} "
        f"loss={result.loss:.6e}"
      )

  return losses