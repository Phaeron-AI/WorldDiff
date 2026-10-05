"""Build a latent cache for training.

  python -m scripts.build_cache --cache-dir cache/dev --num-scenes 256
"""

from __future__ import annotations

# Native Import(s)
import argparse
from pathlib import Path

# Third Party Import(s)
import torch

# Local Import(s)
from src.data.latent_cache import CacheConfig, build_latent_cache
from src.models.vae.vae_adapter import VAEAdapter


def main() -> None:
  p = argparse.ArgumentParser(description="Build the WorldDiff latent cache")

  p.add_argument("--cache-dir", type=Path, default=Path("cache/dev"))
  p.add_argument("--num-scenes", type=int, default=256)
  p.add_argument("--num-views", type=int, default=4)
  p.add_argument("--num-input", type=int, default=2)
  p.add_argument("--height", type=int, default=64)
  p.add_argument("--width", type=int, default=64)
  p.add_argument("--num-points", type=int, default=20_000)
  p.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
  p.add_argument("--base-seed", type=int, default=0)
  p.add_argument("--latent-dtype", type=str, default="float32")
  p.add_argument("--no-ray-normalization", action="store_true")
  p.add_argument("--overwrite", action="store_true")

  # CPU on purpose -- see the note below.
  p.add_argument("--device", type=str, default="cpu")

  args = p.parse_args()

  if not (1 <= args.num_input < args.num_views):
    raise SystemExit(
      f"num_input must satisfy 1 <= num_input < num_views, "
      f"got num_input={args.num_input}, num_views={args.num_views}"
    )

  device = torch.device(args.device)

  # The adapter supplies scaling_factor and downsample, both of which are
  # part of the config that identifies the cache -- so it is built first.
  adapter = VAEAdapter(args.vae_model, dtype=torch.float32, device=device)

  if args.height % adapter.downsample or args.width % adapter.downsample:
    raise SystemExit(
      f"height and width must be divisible by the VAE downsample "
      f"({adapter.downsample}), got {args.height}x{args.width}"
    )

  cfg = CacheConfig(
    num_scenes=args.num_scenes,
    num_views=args.num_views,
    num_input=args.num_input,
    height=args.height,
    width=args.width,
    num_points=args.num_points,
    vae_model=args.vae_model,
    scaling_factor=adapter.scaling_factor,
    downsample=adapter.downsample,
    base_seed=args.base_seed,
    ray_normalization=not args.no_ray_normalization,
    latent_dtype=args.latent_dtype,
  )

  lat_h = args.height // adapter.downsample
  lat_w = args.width // adapter.downsample

  print(f"device       {device}")
  print(f"scenes       {cfg.num_scenes}  (V={cfg.num_views}, num_input={cfg.num_input})")
  print(f"images       {cfg.height}x{cfg.width}")
  print(f"latents      {adapter.latent_channels}x{lat_h}x{lat_w}")
  print(f"config hash  {cfg.hash()}")
  print(f"writing to   {args.cache_dir}")
  print()

  out = build_latent_cache(
    args.cache_dir,
    adapter,
    cfg,
    overwrite=args.overwrite,
  )

  files = sorted(out.glob("scene_*.pt"))
  total_mb = sum(f.stat().st_size for f in files) / 1e6

  print(f"done: {len(files)} scenes, {total_mb:.1f} MB")
  print(f"manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
  main()