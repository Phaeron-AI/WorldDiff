"""
Driver for the P2 overfit gate.

Builds a single fixed 2-view scene, trains MultiViewDiT to memorize it,
then reconstructs the target view from noise and reports the MSE.

Run from the engine root (where `src/` lives):

  python -m src.training.run_overfit

  python -m src.training.run_overfit --steps 4000 --dim 128 --num-layers 6
"""

from __future__ import annotations

import argparse

import torch

from src.models.dit.model import MultiViewDiT
from src.training.overfit import train_overfit
from src.training.reconstruct import reconstruct


def build_fixed_scene(
  *,
  channels: int,
  height: int,
  width: int,
  device: torch.device,
  dtype: torch.dtype,
  seed: int,
):
  """One deterministic 2-view scene: view 0 conditioning, view 1 target."""
  gen = torch.Generator(device="cpu").manual_seed(seed)

  z0 = torch.randn(
    1,
    2,
    channels,
    height,
    width,
    generator=gen,
  )

  rays = torch.randn(
    1,
    2,
    6,
    height,
    width,
    generator=gen,
  )

  z0 = z0.to(
    device=device,
    dtype=dtype,
  )

  rays = rays.to(
    device=device,
    dtype=dtype,
  )

  cond_mask = torch.tensor(
    [[True, False]],
    device=device,
  )

  return z0, rays, cond_mask


def main() -> None:
  p = argparse.ArgumentParser(
    description="WorldDiff P2 overfit gate"
  )

  # Data / scene
  p.add_argument("--channels", type=int, default=4)
  p.add_argument("--height", type=int, default=16)
  p.add_argument("--width", type=int, default=16)

  # Model
  p.add_argument("--dim", type=int, default=128)
  p.add_argument("--num-heads", type=int, default=4)
  p.add_argument("--cond-dim", type=int, default=128)
  p.add_argument("--num-layers", type=int, default=4)
  p.add_argument("--patch-size", type=int, default=2)

  # Optimizer
  p.add_argument("--steps", type=int, default=3000)
  p.add_argument("--lr", type=float, default=3e-4)

  # Sampling
  p.add_argument("--recon-steps", type=int, default=50)

  # Misc
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--recon-seed", type=int, default=7)
  p.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
  )

  args = p.parse_args()

  device = torch.device(args.device)
  dtype = torch.float32

  torch.manual_seed(args.seed)

  # ------------------------------------------------------------
  # 1. Build the fixed 2-view scene
  #
  # View 0 = conditioning
  # View 1 = target
  # ------------------------------------------------------------
  z0, rays, cond_mask = build_fixed_scene(
    channels=args.channels,
    height=args.height,
    width=args.width,
    device=device,
    dtype=dtype,
    seed=args.seed,
  )

  assert z0.shape == (
    1,
    2,
    args.channels,
    args.height,
    args.width,
  )

  assert rays.shape == (
    1,
    2,
    6,
    args.height,
    args.width,
  )

  assert cond_mask.shape == (1, 2)
  assert cond_mask.dtype == torch.bool
  assert cond_mask.tolist() == [[True, False]]

  # ------------------------------------------------------------
  # 2. Construct MultiViewDiT
  # ------------------------------------------------------------
  model = MultiViewDiT(
    in_channels=args.channels,
    dim=args.dim,
    num_heads=args.num_heads,
    cond_dim=args.cond_dim,
    num_layers=args.num_layers,
    patch_size=args.patch_size,
  ).to(
    device=device,
    dtype=dtype,
  )

  n_params = sum(
    p.numel()
    for p in model.parameters()
  )

  print(
    f"device={device} "
    f"params={n_params:,} "
    f"scene=[1,2,{args.channels},{args.height},{args.width}]"
  )

  # ------------------------------------------------------------
  # 3. Fixed reconstruction noise
  #
  # Using the same RNG seed before and after training makes the
  # reconstruction comparison deterministic.
  # ------------------------------------------------------------
  def recon_mse() -> float:
    gen = torch.Generator(
      device=device
    ).manual_seed(args.recon_seed)

    return reconstruct(
      model,
      z0,
      rays,
      cond_mask,
      num_steps=args.recon_steps,
      rng=gen,
    ).item()

  # ------------------------------------------------------------
  # 4. Noise-vs-target baseline
  #
  # This is the expected target-view MSE of pure Gaussian noise.
  # ------------------------------------------------------------
  target_mask = ~cond_mask
  target_indices = target_mask.nonzero(as_tuple=True)

  baseline_noise = torch.randn_like(z0)

  baseline = torch.mean(
    (
      baseline_noise[target_indices]
      - z0[target_indices]
    ) ** 2
  ).item()

  print(
    f"\n[before] recon_MSE = {recon_mse():.4e} "
    f" (noise baseline ~ {baseline:.3e})"
  )

  # ------------------------------------------------------------
  # 5. Overfit the fixed scene
  # ------------------------------------------------------------
  print(
    f"\ntraining for {args.steps} steps @ lr={args.lr} ..."
  )

  losses = train_overfit(
    model,
    z0,
    rays,
    cond_mask,
    num_steps=args.steps,
    lr=args.lr,
  )

  if len(losses) == 0:
    raise RuntimeError(
      "train_overfit returned no recorded losses"
    )

  # ------------------------------------------------------------
  # 6. Reconstruction gate
  # ------------------------------------------------------------
  final_recon = recon_mse()

  print(
    f"\n[after]  train_loss "
    f"{losses[0]:.4e} -> {losses[-1]:.4e}"
  )

  print(
    f"[after]  recon_MSE = {final_recon:.4e}"
  )

  # ------------------------------------------------------------
  # 7. P2 gate
  #
  # Both conditions must pass:
  #
  #   loss dropped substantially
  #   reconstruction is substantially below noise
  # ------------------------------------------------------------
  loss_pass = (
    losses[-1]
    < 0.5 * losses[0]
  )

  recon_pass = (
    final_recon
    < 0.1 * baseline
  )

  ok = loss_pass and recon_pass

  loss_ratio = (
    losses[0]
    / max(losses[-1], 1e-9)
  )

  recon_ratio = (
    baseline
    / max(final_recon, 1e-9)
  )

  print(
    f"\nP2 gate: {'PASS' if ok else 'FAIL'} "
    f"(loss dropped {loss_ratio:.0f}x, "
    f"recon {recon_ratio:.0f}x below noise baseline)"
  )


if __name__ == "__main__":
  main()