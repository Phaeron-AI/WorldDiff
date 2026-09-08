from __future__ import annotations

# Native Import(s)
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

# Third Party Import(s)
import torch

@dataclass(frozen=True)
class CacheConfig:
  num_scenes: int
  num_views: int
  num_input: int
  height: int
  width: int
  num_points: int
  vae_model: str
  scaling_factor: float
  downsample: int
  base_seed: int = 0
  ray_normalization: bool = True
  latent_dtype: str = "float32"

  def hash(self) -> str:
    blob = json.dumps(asdict(self), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]

def build_latent_cache(
  cache_dir: str | Path,
  adapter,
  cfg: CacheConfig,
  *,
  overwrite: bool = False
) -> Path:
  from src.data.synth_blender import synth_scene
  from src.model.camera import rescale_intrinsics

  cache_dir = Path(cache_dir)
  dt = getattr(torch, cfg.latent_dtype)

  for i in range(cfg.num_scenes):
    path = cache_dir / f"scene_{i:06d}.pt"
    if path.exists() and not overwrite:
      continue

    s = synth_scene(
      seed=cfg.base_seed + i, num_views=cfg.num_views,
      num_input=cfg.num_input, height=cfg.height, width=cfg.width,
      num_points=cfg.num_points
    )

    K_lat = rescale_intrinsics(s.intrinsics, adapter.downsample)

    z = adapter.encode(s.images)

    torch.save({
      "latents": z.to(dt).cpu(),
      "c2w": s.c2w.cpu(),
      "intrinsics": K_lat.cpu(),
      "cond_mask": s.cond_mask.cpu(),
      "scene_id": s.scene_id,
    }, path)

  manifest = {
    "hash": cfg.hash(), 
    "config": asdict(cfg),
    "files": [f"scene_{i:06d}.pt" for i in range(cfg.num_scenes)]
  }

  (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  return cache_dir
