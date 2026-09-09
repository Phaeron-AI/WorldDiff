from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.latent_cache import CacheConfig, build_latent_cache


class _StubAdapter:
  """Stand-in for VAEAdapter: 8x downsample, [V,3,H,W] -> [V,4,H/8,W/8]."""
  downsample = 8

  def encode(self, images: torch.Tensor) -> torch.Tensor:
    z = F.avg_pool2d(2.0 * images - 1.0, 8)
    return torch.cat([z, z[:, :1]], dim=1)


def _cfg(**over) -> CacheConfig:
  base = dict(
    num_scenes=4, num_views=4, num_input=2, height=64, width=64,
    num_points=3000, vae_model="stub", scaling_factor=0.18215, downsample=8,
  )
  base.update(over)
  return CacheConfig(**base)  # type: ignore


def test_build_creates_manifest_and_files(tmp_path) -> None:
  cfg = _cfg()
  build_latent_cache(tmp_path, _StubAdapter(), cfg)
  assert (tmp_path / "manifest.json").exists()
  for i in range(cfg.num_scenes):
    assert (tmp_path / f"scene_{i:06d}.pt").exists()


def test_build_creates_missing_dir(tmp_path) -> None:
  nested = tmp_path / "does" / "not" / "exist"      # mkdir must handle this
  build_latent_cache(nested, _StubAdapter(), _cfg(num_scenes=1))
  assert (nested / "manifest.json").exists()


def test_manifest_hash_matches_config(tmp_path) -> None:
  import json
  cfg = _cfg()
  build_latent_cache(tmp_path, _StubAdapter(), cfg)
  manifest = json.loads((tmp_path / "manifest.json").read_text())
  assert manifest["hash"] == cfg.hash()
  assert len(manifest["files"]) == cfg.num_scenes


def test_config_hash_changes_with_any_field() -> None:
  assert _cfg().hash() != _cfg(height=128).hash()
  assert _cfg().hash() != _cfg(latent_dtype="float16").hash()
  assert _cfg().hash() == _cfg().hash()              # stable


def test_record_shapes_and_cpu(tmp_path) -> None:
  cfg = _cfg()
  build_latent_cache(tmp_path, _StubAdapter(), cfg)
  rec = torch.load(tmp_path / "scene_000000.pt", map_location="cpu")
  assert rec["latents"].shape == (4, 4, 8, 8)
  assert rec["c2w"].shape == (4, 4, 4)
  assert rec["intrinsics"].shape == (4, 3, 3)
  assert rec["cond_mask"].shape == (4,)
  assert rec["scene_id"] == "synth-000000"
  for k in ("latents", "c2w", "intrinsics", "cond_mask"):
    assert rec[k].device.type == "cpu"


def test_latent_dtype_respected(tmp_path) -> None:
  build_latent_cache(tmp_path, _StubAdapter(), _cfg(latent_dtype="float16"))
  rec = torch.load(tmp_path / "scene_000000.pt", map_location="cpu")
  assert rec["latents"].dtype == torch.float16


def test_intrinsics_are_latent_res(tmp_path) -> None:
  # K rescaled by 1/downsample at build time; principal point ~ latent centre
  cfg = _cfg()
  build_latent_cache(tmp_path, _StubAdapter(), cfg)
  rec = torch.load(tmp_path / "scene_000000.pt", map_location="cpu")
  assert torch.allclose(rec["intrinsics"][:, 0, 2], torch.full((4,), 64 / 8 / 2), atol=1e-4)


def test_skips_existing_without_overwrite(tmp_path) -> None:
  tmp_path.mkdir(parents=True, exist_ok=True)
  sentinel = tmp_path / "scene_000000.pt"
  sentinel.write_bytes(b"SENTINEL")                  # pre-existing, not a real record
  build_latent_cache(tmp_path, _StubAdapter(), _cfg(num_scenes=2), overwrite=False)
  assert sentinel.read_bytes() == b"SENTINEL"        # untouched -> resume, not restart
  assert (tmp_path / "scene_000001.pt").exists()     # the missing one was built


def test_overwrite_rebuilds(tmp_path) -> None:
  tmp_path.mkdir(parents=True, exist_ok=True)
  sentinel = tmp_path / "scene_000000.pt"
  sentinel.write_bytes(b"SENTINEL")
  build_latent_cache(tmp_path, _StubAdapter(), _cfg(num_scenes=1), overwrite=True)
  assert sentinel.read_bytes() != b"SENTINEL"        # replaced with a real record
  rec = torch.load(sentinel, map_location="cpu")
  assert rec["latents"].shape == (4, 4, 8, 8)