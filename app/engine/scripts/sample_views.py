"""Decode predictions from a checkpoint and write PNG grids.

  python -m scripts.sample_views --num-scenes 4
  python -m scripts.sample_views --step 1000 --w 1.5
"""

from __future__ import annotations

# Native Import(s)
import argparse
import json
from pathlib import Path

# Third Party Import(s)
import numpy as np
import torch
from PIL import Image

# Local Import(s)
from src.data.loader import SceneLatentData
from src.eval.validate import (
  K_ROUTINE,
  _generator,
  _substream,
  _validation_seed,
  guided_sample,
)
from src.models.dit.model import MultiViewDiT
from src.models.flow import sample
from src.models.vae.vae_adapter import VAEAdapter
from src.training.state.checkpoint import CheckpointManager

VAL_SEED = 2026  # must match run_train.py


def to_uint8(img: torch.Tensor) -> np.ndarray:
  """[3,H,W] in [0,1] -> HxWx3 uint8."""
  arr = (img.clamp(0.0, 1.0) * 255.0).round().byte()
  return arr.permute(1, 2, 0).cpu().numpy()


def make_grid(rows, *, pad: int = 4, scale: int = 4) -> Image.Image:
  n_rows, n_cols = len(rows), len(rows[0])
  h, w = rows[0][0].shape[-2:]

  canvas = np.full(
    (n_rows * h + (n_rows + 1) * pad, n_cols * w + (n_cols + 1) * pad, 3),
    255,
    dtype=np.uint8,
  )

  for r, row in enumerate(rows):
    for c, img in enumerate(row):
      y = pad + r * (h + pad)
      x = pad + c * (w + pad)
      canvas[y:y + h, x:x + w] = to_uint8(img)

  if scale > 1:
    canvas = canvas.repeat(scale, axis=0).repeat(scale, axis=1)

  return Image.fromarray(canvas)


def main() -> None:
  p = argparse.ArgumentParser(description="Sample and decode views")

  p.add_argument("--run-dir", type=Path, default=Path("runs/dev"))
  p.add_argument("--step", type=int, default=None,
                 help="omit to load the latest checkpoint")
  p.add_argument("--cache-dir", type=Path, default=Path("cache/dev"))
  p.add_argument("--out-dir", type=Path, default=Path("outputs/samples"))

  # Must match the architecture the checkpoint was trained with --
  # state_dict() records the TrainingConfig but not the model shape.
  p.add_argument("--dim", type=int, default=128)
  p.add_argument("--num-heads", type=int, default=4)
  p.add_argument("--cond-dim", type=int, default=128)
  p.add_argument("--num-layers", type=int, default=4)
  p.add_argument("--patch-size", type=int, default=2)

  p.add_argument("--num-scenes", type=int, default=4)
  p.add_argument("--val-scenes", type=int, default=32)
  p.add_argument("--steps", type=int, default=K_ROUTINE)
  p.add_argument("--w", type=float, default=1.0,
                 help="w != 1 routes through guided_sample (two-state CFG)")
  p.add_argument("--scale", type=int, default=4)
  p.add_argument("--device", type=str,
                 default="cuda" if torch.cuda.is_available() else "cpu")

  args = p.parse_args()
  device = torch.device(args.device)

  # ------------------------------------------------------------------
  # Data -- the same held-out slice run_train.py validates on.
  # ------------------------------------------------------------------
  dataset = SceneLatentData(args.cache_dir)
  manifest = json.loads((args.cache_dir / "manifest.json").read_text())

  n_train = len(dataset) - args.val_scenes
  val_set = [dataset[i] for i in range(n_train, len(dataset))]
  val_set = val_set[:args.num_scenes]

  probe = val_set[0]
  _, latent_c, _, _ = probe["latents"].shape

  # ------------------------------------------------------------------
  # Model + checkpoint
  # ------------------------------------------------------------------
  model = MultiViewDiT(
    in_channels=latent_c,
    dim=args.dim,
    num_heads=args.num_heads,
    cond_dim=args.cond_dim,
    num_layers=args.num_layers,
    patch_size=args.patch_size,
  ).to(device)

  manager = CheckpointManager(args.run_dir)

  if args.step is None:
    ckpt, ckpt_path = manager.load_latest()
  else:
    # .resolve() because load() joins any relative path onto checkpoint_dir,
    # and step_path() has already done that join.
    ckpt_path = manager.step_path(args.step).resolve()
    ckpt = manager.load(ckpt_path)

  model.load_state_dict(ckpt["model"])
  model.eval()

  step = int(ckpt["counters"]["optimizer_step"])

  vae = VAEAdapter(
    manifest["config"]["vae_model"],
    dtype=torch.float32,
    device=device,
  )

  args.out_dir.mkdir(parents=True, exist_ok=True)

  print(f"checkpoint   {ckpt_path.name}  (step {step})")
  print(f"sampling     K={args.steps}  w={args.w}")
  print(f"scenes       {len(val_set)}")
  print()

  # ------------------------------------------------------------------
  # Sample
  # ------------------------------------------------------------------
  with torch.no_grad():
    for scene in val_set:
      scene_id = scene["scene_id"]

      z0 = scene["latents"].to(device).unsqueeze(0)
      rays = scene["rays"].to(device).unsqueeze(0)
      cond_mask = scene["cond_mask"].to(device, dtype=torch.bool).unsqueeze(0)

      k = int(cond_mask.sum().item())

      # Same seed derivation as validate_routine, so these images are the
      # ones behind the reported metric rather than a different draw.
      base = _validation_seed(VAL_SEED, scene_id, k)
      g_cond = _generator(_substream(base, "c"), device)

      if args.w == 1.0:
        pred = sample(
          model, z0, rays, cond_mask, num_steps=args.steps, rng=g_cond
        )
      else:
        g_uncond = _generator(_substream(base, "u"), device)
        pred = guided_sample(
          model, z0, rays, cond_mask,
          w=args.w, num_steps=args.steps,
          g_uncond=g_uncond, g_cond=g_cond,
        )

      target_mask = ~cond_mask
      latent_mse = (
        (pred[target_mask] - z0[target_mask]) ** 2
      ).mean().item()

      truth = vae.decode(z0)[0]    # [V,3,H,W]
      recon = vae.decode(pred)[0]

      grid = make_grid(
        [list(truth), list(recon)],
        scale=args.scale,
      )

      out = args.out_dir / f"{scene_id}_step{step:06d}_w{args.w}.png"
      grid.save(out)

      roles = "".join("C" if m else "T" for m in cond_mask[0].tolist())
      print(f"{scene_id}  views {roles}  latent_mse {latent_mse:.4f}  -> {out.name}")

  print(f"\ntop row = D(z0) ground truth, bottom row = D(z_hat) prediction")
  print(f"C columns are pinned and should match exactly; T columns are the test")


if __name__ == "__main__":
  main()