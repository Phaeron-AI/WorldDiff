from __future__ import annotations

import json

import pytest
import torch

from src.data.latent_cache import CacheConfig
from src.data.loader import SceneLatentData, collate_scenes, make_loader
from src.data.rays import RayEncoder


def _cfg(**over) -> CacheConfig:
  base = dict(
    num_scenes=3, num_views=4, num_input=2, height=64, width=64,
    num_points=3000, vae_model="stub", scaling_factor=0.18215, downsample=8,
  )
  base.update(over)
  return CacheConfig(**base)  # type: ignore


def _asdict(cfg: CacheConfig) -> dict:
  from dataclasses import asdict
  return asdict(cfg)


def _fake_cache(tmp_path, cfg: CacheConfig, *, C=4, h=8, w=8):
  """Write a cache directly (no VAE) so the read path is tested in isolation."""
  g = torch.Generator().manual_seed(0)
  for i in range(cfg.num_scenes):
    c2w = torch.eye(4).expand(cfg.num_views, 4, 4).clone()
    c2w[:, :3, 3] = torch.randn(cfg.num_views, 3, generator=g)
    K = torch.zeros(cfg.num_views, 3, 3)
    K[:, 0, 0] = K[:, 1, 1] = 4.0
    K[:, 0, 2] = K[:, 1, 2] = w / 2
    K[:, 2, 2] = 1.0
    cond = torch.zeros(cfg.num_views, dtype=torch.bool)
    cond[: cfg.num_input] = True
    torch.save({
      "latents": torch.randn(cfg.num_views, C, h, w, generator=g),
      "c2w": c2w,
      "intrinsics": K,
      "cond_mask": cond,
      "scene_id": f"synth-{i:06d}",
    }, tmp_path / f"scene_{i:06d}.pt")
  manifest = {"hash": cfg.hash(), "config": _asdict(cfg),
              "files": [f"scene_{i:06d}.pt" for i in range(cfg.num_scenes)]}
  (tmp_path / "manifest.json").write_text(json.dumps(manifest))
  return tmp_path


def test_len(tmp_path) -> None:
  cfg = _cfg()
  _fake_cache(tmp_path, cfg)
  assert len(SceneLatentData(tmp_path)) == cfg.num_scenes    # guards __len__ -> None


def test_getitem_shapes(tmp_path) -> None:
  _fake_cache(tmp_path, _cfg())
  item = SceneLatentData(tmp_path)[0]
  assert item["latents"].shape == (4, 4, 8, 8)
  assert item["rays"].shape == (4, 6, 8, 8)
  assert item["c2w"].shape == (4, 4, 4)
  assert item["intrinsics"].shape == (4, 3, 3)
  assert item["cond_mask"].shape == (4,) and item["cond_mask"].dtype == torch.bool
  assert item["scene_id"] == "synth-000000"
  assert item["latents"].dtype == torch.float32


def test_rays_match_rayencoder(tmp_path) -> None:
  cfg = _cfg()
  _fake_cache(tmp_path, cfg)
  ds = SceneLatentData(tmp_path)
  rec = torch.load(ds._files[0], map_location="cpu")
  ref = RayEncoder(normalize=cfg.ray_normalization)(rec["intrinsics"], rec["c2w"], 8, 8)
  assert torch.equal(ref, ds[0]["rays"])


def test_collate_shapes_and_scene_id(tmp_path) -> None:
  _fake_cache(tmp_path, _cfg())
  ds = SceneLatentData(tmp_path)
  batch = collate_scenes([ds[0], ds[1], ds[2]])
  assert batch["latents"].shape == (3, 4, 4, 8, 8)
  assert batch["rays"].shape == (3, 4, 6, 8, 8)
  assert batch["c2w"].shape == (3, 4, 4, 4)
  assert batch["cond_mask"].shape == (3, 4)
  assert isinstance(batch["scene_id"], list)
  assert batch["scene_id"] == ["synth-000000", "synth-000001", "synth-000002"]


def test_make_loader_yields_batch(tmp_path) -> None:
  _fake_cache(tmp_path, _cfg())
  dl = make_loader(SceneLatentData(tmp_path), batch_size=2, shuffle=False, num_workers=0)
  batch = next(iter(dl))
  assert batch["latents"].shape == (2, 4, 4, 8, 8)
  assert len(batch["scene_id"]) == 2


def test_loader_covers_all(tmp_path) -> None:
  cfg = _cfg()
  _fake_cache(tmp_path, cfg)
  dl = make_loader(SceneLatentData(tmp_path), batch_size=2, shuffle=False)  # 2 + 1
  seen = sum(b["latents"].shape[0] for b in dl)
  assert seen == cfg.num_scenes


def test_stale_cache_raises(tmp_path) -> None:
  _fake_cache(tmp_path, _cfg())
  with pytest.raises(RuntimeError):
    SceneLatentData(tmp_path, expected_cfg=_cfg(height=128))   # different hash


def test_matching_cfg_ok(tmp_path) -> None:
  cfg = _cfg()
  _fake_cache(tmp_path, cfg)
  assert len(SceneLatentData(tmp_path, expected_cfg=cfg)) == cfg.num_scenes


def test_no_expected_cfg_skips_guard(tmp_path) -> None:
  _fake_cache(tmp_path, _cfg())
  assert len(SceneLatentData(tmp_path, expected_cfg=None)) == 3