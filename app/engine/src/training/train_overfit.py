from __future__ import annotations

# Third Party Import(s)
import torch
from torch import Tensor

# Local Import(s)
from src.models.dit.model import MultiViewDiT
from src.models.flow import (
  linear_interpolant,
  apply_conditioning,
  masked_flow_matching_loss,
)


def train_overfit(
  model: MultiViewDiT,
  z0: Tensor,
  rays: Tensor,
  cond_mask: Tensor,
  num_steps: int = 5000,
  lr: float = 1e-4,
) -> None:
  model.train()

  # ------------------------------------------------------------
  # Input invariants
  # ------------------------------------------------------------
  assert z0.ndim == 5, "z0 must have shape [B, V, C, H, W]"
  assert rays.ndim == 5, "rays must have shape [B, V, 6, H, W]"
  assert cond_mask.ndim == 2, "cond_mask must have shape [B, V]"
  assert cond_mask.dtype == torch.bool, "cond_mask must be bool"

  B, V, C, H, W = z0.shape

  assert B == 1, "P2 overfit gate expects B=1"
  assert V == 2, "P2 overfit gate expects exactly 2 views"
  assert rays.shape == (B, V, 6, H, W)
  assert cond_mask.shape == (B, V)

  # Exactly one conditioning view and one target view.
  assert cond_mask.sum().item() == 1, (
    "cond_mask must contain exactly one conditioning view"
  )

  target_mask = ~cond_mask

  assert target_mask.sum().item() == 1, (
    "target_mask must contain exactly one target view"
  )

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
  )

  for step in range(num_steps):
    optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------
    # 1. Sample rectified-flow endpoint and time
    # ------------------------------------------------------------
    epsilon = torch.randn_like(z0)

    t = torch.rand(
      B,
      device=z0.device,
      dtype=z0.dtype,
    )

    # ------------------------------------------------------------
    # 2. Construct point on the flow trajectory
    #
    # z_t = (1 - t) * z0 + t * epsilon
    # ------------------------------------------------------------
    z_t = linear_interpolant(
      z0=z0,
      eps=epsilon,
      t=t,
    )

    # ------------------------------------------------------------
    # 3. Pin the conditioning view to the clean latent
    #
    # cond view:
    #   z_t = z0
    #
    # target view:
    #   z_t = (1 - t) * z0 + t * epsilon
    # ------------------------------------------------------------
    z_t = apply_conditioning(
      zt=z_t,
      z0=z0,
      cond_mask=cond_mask,
    )

    # ------------------------------------------------------------
    # 4. Predict velocity
    # ------------------------------------------------------------
    velocity = model(
      x=z_t,
      rays=rays,
      t=t,
      cond_mask=cond_mask,
    )

    # ------------------------------------------------------------
    # 5. Rectified-flow velocity target
    #
    # z_t = (1 - t) * z0 + t * epsilon
    #
    # Therefore:
    #
    #   dz_t / dt = epsilon - z0
    #
    # The conditioning view is NOT scored.
    # Only the target view contributes to the loss.
    # ------------------------------------------------------------
    target = epsilon - z0

    loss = masked_flow_matching_loss(
      v_pred=velocity,
      target=target,
      target_mask=target_mask,
    )

    # ------------------------------------------------------------
    # 6. Optimize
    # ------------------------------------------------------------
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
      print(
        f"step={step:05d} "
        f"loss={loss.item():.6e}"
      )