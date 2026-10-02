from __future__ import annotations

import pytest
import torch

from src.data.synth_blender import (
  look_at,
  make_point_cloud,
  render,
  sample_cameras,
  synth_scene,
)
from src.data.rays import RayEncoder


H = W = 64


def _scene(seed: int = 0, num_views: int = 4, num_input: int = 2):
  return synth_scene(
    seed=seed,
    num_views=num_views,
    num_input=num_input,
    height=H,
    width=W,
    num_points=8000,
  )


# ----------------------------------------------------------------------
# Contract & shapes
# ----------------------------------------------------------------------

def test_valid_scenesample_and_masks() -> None:
  s = _scene()
  assert s.images.shape == (4, 3, H, W)
  assert s.num_views == 4
  assert s.input_idx.tolist() == [0, 1]
  assert s.target_idx.tolist() == [2, 3]
  assert s.scene_id == "synth-000000"
  # images are valid pixels
  assert s.images.min() >= 0.0 and s.images.max() <= 1.0


def test_rejects_bad_num_input() -> None:
  with pytest.raises(ValueError):
    _scene(num_input=0)               # no conditioning view
  with pytest.raises(ValueError):
    _scene(num_views=3, num_input=3)  # no target view


# ----------------------------------------------------------------------
# Determinism  (the reproducibility contract)
# ----------------------------------------------------------------------

def test_deterministic_same_seed() -> None:
  a = _scene(seed=7)
  b = _scene(seed=7)
  assert torch.equal(a.images, b.images)
  assert torch.equal(a.c2w, b.c2w)
  assert torch.equal(a.intrinsics, b.intrinsics)


def test_different_seed_differs() -> None:
  a = _scene(seed=0)
  b = _scene(seed=1)
  assert not torch.equal(a.c2w, b.c2w)
  assert not torch.equal(a.images, b.images)


def test_render_deterministic_identical_inputs() -> None:
  # the core bug that was fixed: repeated render of identical inputs must match
  g = torch.Generator().manual_seed(0)
  K, c2w = sample_cameras(4, rng=g, height=H, width=W)
  pts, col = make_point_cloud(rng=g, num_points=8000)
  i1 = render(pts, col, K, c2w, H, W)
  i2 = render(pts, col, K, c2w, H, W)
  assert torch.equal(i1, i2)


# ----------------------------------------------------------------------
# Camera geometry
# ----------------------------------------------------------------------

def test_cameras_orthonormal_det1() -> None:
  s = _scene()
  R = s.c2w[:, :3, :3]
  eye3 = torch.eye(3).expand(4, 3, 3)
  assert torch.allclose(R.transpose(-1, -2) @ R, eye3, atol=1e-5)
  assert torch.allclose(torch.linalg.det(R), torch.ones(4), atol=1e-5)


def test_cameras_look_at_center() -> None:
  s = _scene()
  o = s.c2w[:, :3, 3]
  forward = s.c2w[:, :3, 2]                       # +z column
  want = torch.nn.functional.normalize(-o, dim=-1)  # target is the origin
  assert torch.allclose(forward, want, atol=1e-4)


def test_look_at_matches_sample_cameras() -> None:
  # standalone look_at must reproduce the vectorised c2w that sample_cameras
  # builds for the same eye (target = origin).
  g = torch.Generator().manual_seed(11)
  _, c2w = sample_cameras(4, rng=g, height=H, width=W)
  target = torch.zeros(3)
  for v in range(4):
    eye = c2w[v, :3, 3]
    m = look_at(eye, target)
    assert m.shape == (4, 4)
    assert torch.allclose(m, c2w[v], atol=1e-5)


# ----------------------------------------------------------------------
# Multi-view signal
# ----------------------------------------------------------------------

def test_parallax_exists() -> None:
  # a fixed world point projects to different pixels across views
  s = _scene()
  K = s.intrinsics
  X = torch.tensor([0.3, -0.2, 0.4])
  px = []
  for v in range(4):
    R = s.c2w[v, :3, :3]
    o = s.c2w[v, :3, 3]
    uv = K[v] @ (R.T @ (X - o))
    px.append((uv[0] / uv[2]).item())
  assert max(px) - min(px) > 1.0


def test_convention_matches_rayencoder() -> None:
  # STAR: render one bright point; the pixel it lands in, RayEncoder's ray there
  # must point back at it. This is the synth <-> rays coordinate contract.
  enc = RayEncoder(normalize=True)
  g = torch.Generator().manual_seed(3)
  K, c2w = sample_cameras(4, rng=g, height=H, width=W)
  X = torch.tensor([[0.2, 0.1, -0.3]])
  white = torch.tensor([[1.0, 1.0, 1.0]])
  imgs = render(X, white, K, c2w, H, W, point_radius=0)
  rays = enc(K, c2w, H, W)

  checked = 0
  for v in range(4):
    lit = (imgs[v, 0] > 0.5).nonzero()
    if lit.numel() == 0:
      continue
    row, col = lit[0].tolist()
    d = rays[v, 0:3, row, col]
    want = torch.nn.functional.normalize(X[0] - c2w[v, :3, 3], dim=0)
    assert torch.dot(d, want).item() > 0.99
    checked += 1
  assert checked >= 1


# ----------------------------------------------------------------------
# Rasterizer correctness
# ----------------------------------------------------------------------

def test_render_occlusion_nearest_wins() -> None:
  # two points on the same viewing ray: the nearer colour must survive.
  K = torch.tensor([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1.0]]).unsqueeze(0)
  c2w = torch.eye(4).unsqueeze(0)
  c2w[0, :3, 3] = torch.tensor([0.0, 0.0, -3.0])  # camera behind origin, +z forward
  pts = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])  # first is nearer
  cols = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # near=red, far=blue
  img = render(pts, cols, K, c2w, H, W, point_radius=0)
  center = img[0, :, H // 2, W // 2]
  assert torch.allclose(center, torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_render_background_when_empty() -> None:
  # all points behind the camera -> pure background
  K = torch.tensor([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1.0]]).unsqueeze(0)
  c2w = torch.eye(4).unsqueeze(0)  # at origin, looking +z
  pts = torch.tensor([[0.0, 0.0, -1.0]])  # behind
  cols = torch.tensor([[1.0, 1.0, 1.0]])
  img = render(pts, cols, K, c2w, H, W, bg=0.0)
  assert float(img.max()) == 0.0