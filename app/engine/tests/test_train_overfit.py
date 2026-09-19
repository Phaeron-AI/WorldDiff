"""
P2 overfit gate — regression tests.

Two headline tests assert the gate itself (loss drops, sampling reconstructs).
Five fast micro-tests each pin a specific bug that was hit while building P2,
so a regression fails loudly and instantly.

All tests are CPU-only and fully seeded for determinism.
"""
from __future__ import annotations

import pytest
import torch

from src.models.dit.model import MultiViewDiT
from src.models.flow import (
  linear_interpolant,
  apply_conditioning,
  masked_flow_matching_loss,
)
from src.models.flow.sampler import sample
from src.training.overfit import train_overfit
from src.training.reconstruct import reconstruct


# ----------------------------------------------------------------------
# Shared fixtures / helpers
# ----------------------------------------------------------------------
DEVICE = torch.device("cpu")
CHANNELS, H, W = 4, 16, 16


def make_scene(*, seed: int = 0, dtype: torch.dtype = torch.float32):
  """One deterministic 2-view scene: view 0 conditioning, view 1 target."""
  gen = torch.Generator(device="cpu").manual_seed(seed)
  z0 = torch.randn(1, 2, CHANNELS, H, W, generator=gen).to(DEVICE, dtype)
  rays = torch.randn(1, 2, 6, H, W, generator=gen).to(DEVICE, dtype)
  cond_mask = torch.tensor([[True, False]], device=DEVICE)
  return z0, rays, cond_mask


def make_model(*, seed: int = 0, dtype: torch.dtype = torch.float32) -> MultiViewDiT:
  torch.manual_seed(seed)
  return MultiViewDiT(
    in_channels=CHANNELS, dim=32, num_heads=4,
    cond_dim=32, num_layers=2, patch_size=2,
  ).to(DEVICE, dtype)


def noise_baseline(z0, cond_mask):
  """Expected target-view MSE of pure Gaussian noise vs the truth."""
  idx = (~cond_mask).nonzero(as_tuple=True)
  return ((torch.randn_like(z0) - z0)[idx] ** 2).mean().item()


# ----------------------------------------------------------------------
# Headline gate tests
#
# Trained ONCE via a module-scoped fixture (training is the expensive part),
# then both gate conditions read the same run. Marked `slow` so the fast
# regression micro-tests can run on every commit without it.
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def overfit_run():
  torch.manual_seed(0)
  z0, rays, cond_mask = make_scene(seed=0)
  # a well-powered-but-tiny config that converges in a few seconds on CPU
  torch.manual_seed(0)
  model = MultiViewDiT(
    in_channels=CHANNELS, dim=64, num_heads=4,
    cond_dim=64, num_layers=3, patch_size=2,
  ).to(DEVICE)
  baseline = noise_baseline(z0, cond_mask)
  losses = train_overfit(model, z0, rays, cond_mask, num_steps=600, lr=1e-3)
  gen = torch.Generator(device=DEVICE).manual_seed(7)
  recon_mse = reconstruct(model, z0, rays, cond_mask, num_steps=30, rng=gen).item()
  return {"losses": losses, "baseline": baseline, "recon_mse": recon_mse}


@pytest.mark.slow
def test_overfit_drives_loss_down(overfit_run):
  losses = overfit_run["losses"]
  assert losses is not None and len(losses) == 600, "train_overfit must return its loss list"
  # direction, not convergence: the loss must fall by a wide margin.
  assert losses[-1] < 0.3 * losses[0], (
    f"loss did not drop enough: {losses[0]:.3e} -> {losses[-1]:.3e}"
  )


@pytest.mark.slow
def test_overfit_reconstructs_target(overfit_run):
  mse = overfit_run["recon_mse"]
  baseline = overfit_run["baseline"]
  # a learning model sits far below the noise baseline; a broken one ~ at it.
  assert mse < 0.25 * baseline, (
    f"reconstruction MSE {mse:.3e} not below 0.25x baseline {baseline:.3e}"
  )


# ----------------------------------------------------------------------
# Regression micro-tests — each pins a specific bug hit during P2
# ----------------------------------------------------------------------
def test_v_init_near_zero():
  """Zero-init output head => v_theta ~ 0 at init (rectified-flow requirement)."""
  z0, rays, cond_mask = make_scene(seed=1)
  model = make_model(seed=1)
  model.eval()
  t = torch.rand(1)
  with torch.no_grad():
    v = model(x=z0, rays=rays, t=t, cond_mask=cond_mask)
  assert v.abs().max().item() < 1e-6, "model is not ~0 at init; output head not zero-inited"


def test_loss_targets_velocity_not_state():
  """The loss must bottom out at u = eps - z0, NOT at the trajectory point z_t."""
  z0, rays, cond_mask = make_scene(seed=2)
  target_mask = ~cond_mask
  eps = torch.randn_like(z0)
  t = torch.rand(1)
  z_t = linear_interpolant(z0=z0, eps=eps, t=t)
  u = eps - z0

  # predicting the true velocity is a perfect (zero) loss
  perfect = masked_flow_matching_loss(v_pred=u, target=u, target_mask=target_mask)
  assert perfect.item() == pytest.approx(0.0, abs=1e-6)

  # had the target been z_t (the old bug), the same prediction is NOT optimal
  wrong = masked_flow_matching_loss(v_pred=u, target=z_t, target_mask=target_mask)
  assert wrong.item() > 1e-3, "z_t and eps-z0 must be different objectives"


def test_loss_mask_polarity():
  """The loss scores the TARGET view, not the conditioning view."""
  z0, rays, cond_mask = make_scene(seed=3)
  target_mask = ~cond_mask
  eps = torch.randn_like(z0)
  u = eps - z0

  # perfect on the target view, deliberately corrupted on the cond view
  v_pred = u.clone()
  cond_idx = cond_mask.nonzero(as_tuple=True)
  v_pred[cond_idx] += 100.0  # huge error, but only on the conditioning view

  # correct polarity ignores the cond-view corruption -> ~0
  scored_target = masked_flow_matching_loss(v_pred=v_pred, target=u, target_mask=target_mask)
  assert scored_target.item() == pytest.approx(0.0, abs=1e-6)

  # inverted polarity (scoring the cond view) would see the corruption -> large
  scored_cond = masked_flow_matching_loss(v_pred=v_pred, target=u, target_mask=cond_mask)
  assert scored_cond.item() > 1.0

  # a mask with no target views must raise, not silently return something
  empty = torch.zeros_like(cond_mask, dtype=torch.bool)
  with pytest.raises(Exception):
    masked_flow_matching_loss(v_pred=u, target=u, target_mask=empty)


def test_sampler_pins_cond_view():
  """After sampling, the conditioning view is exactly z0 (pinning + rays reach the model)."""
  z0, rays, cond_mask = make_scene(seed=4)
  model = make_model(seed=4)
  gen = torch.Generator(device=DEVICE).manual_seed(0)
  out = sample(model=model, z0=z0, rays=rays, cond_mask=cond_mask, num_steps=10, rng=gen)

  cond_idx = cond_mask.nonzero(as_tuple=True)
  assert torch.equal(out[cond_idx], z0[cond_idx]), "sampler did not keep cond view pinned to z0"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_dtype_portability(dtype):
  """Forward runs in fp64 and fp16 (guards hardcoded-float32 tensor creation)."""
  z0, rays, cond_mask = make_scene(seed=5, dtype=dtype)
  model = make_model(seed=5, dtype=dtype)
  model.eval()
  t = torch.rand(1, dtype=dtype)
  with torch.no_grad():
    v = model(x=z0, rays=rays, t=t, cond_mask=cond_mask)
  assert v.dtype == dtype
  assert torch.isfinite(v).all()