from __future__ import annotations

import pytest
import torch

from src.models.camera import rescale_intrinsics
from src.data.rays import RayEncoder


def _K(fx=30.0, fy=30.0, cx=32.0, cy=32.0) -> torch.Tensor:
  return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_rescale_values() -> None:
  K = _K().unsqueeze(0)                 # [1,3,3]
  out = rescale_intrinsics(K, 8)
  s = 1.0 / 8
  assert out[0, 0, 0] == pytest.approx(30.0 * s)   # fx
  assert out[0, 1, 1] == pytest.approx(30.0 * s)   # fy
  assert out[0, 0, 2] == pytest.approx(32.0 * s)   # cx
  assert out[0, 1, 2] == pytest.approx(32.0 * s)   # cy
  assert out[0, 2, 2] == pytest.approx(1.0)        # row 2 untouched
  assert out[0, 0, 1] == 0.0 and out[0, 1, 0] == 0.0


def test_does_not_mutate_input() -> None:
  K = _K().unsqueeze(0)
  before = K.clone()
  _ = rescale_intrinsics(K, 8)
  assert torch.equal(K, before)          # clone(), not in-place on the arg


def test_rejects_bad_factor() -> None:
  with pytest.raises(ValueError):
    rescale_intrinsics(_K(), 0)
  with pytest.raises(ValueError):
    rescale_intrinsics(_K(), -4)


def test_leading_dims_preserved() -> None:
  K = _K().expand(2, 5, 3, 3).contiguous()   # [B,V,3,3]
  out = rescale_intrinsics(K, 4)
  assert out.shape == (2, 5, 3, 3)
  assert torch.allclose(out[..., 0, 0], torch.full((2, 5), 30.0 / 4))


def test_rescale_matches_rayencoder() -> None:
  f = 8
  Hf = Wf = 64
  Hl, Wl = Hf // f, Wf // f

  K_full = _K(cx=Wf / 2, cy=Hf / 2).unsqueeze(0)
  K_lat = rescale_intrinsics(K_full, f)

  c2w = torch.eye(4).unsqueeze(0)
  c2w[0, :3, 3] = torch.tensor([0.3, -0.2, -2.0])
  th = torch.tensor(0.2)
  c2w[0, :3, :3] = torch.tensor(
    [[torch.cos(th), 0.0, torch.sin(th)],
     [0.0, 1.0, 0.0],
     [-torch.sin(th), 0.0, torch.cos(th)]]
  )

  enc = RayEncoder(normalize=True)
  rays_full = enc(K_full, c2w, Hf, Wf)   # [1,6,Hf,Wf]
  rays_lat = enc(K_lat, c2w, Hl, Wl)     # [1,6,Hl,Wl]

  # latent pixel (rl,cl) centre maps to full continuous coord (c+0.5)*f - 0.5
  max_err = 0.0
  for rl in range(Hl):
    for cl in range(Wl):
      cf = int(round((cl + 0.5) * f - 0.5))
      rf = int(round((rl + 0.5) * f - 0.5))
      if not (0 <= cf < Wf and 0 <= rf < Hf):
        continue
      d_lat = rays_lat[0, 0:3, rl, cl]
      d_full = rays_full[0, 0:3, rf, cf]
      max_err = max(max_err, float(1.0 - torch.dot(d_lat, d_full)))
  assert max_err < 1e-3