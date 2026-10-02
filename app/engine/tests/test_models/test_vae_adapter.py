from __future__ import annotations

import math

import pytest
import torch

from src.models.vae_adapter import VAEAdapter


# A frozen VAE is expensive to load; build it once for the whole module.
@pytest.fixture(scope="module")
def adapter() -> VAEAdapter:
  return VAEAdapter(dtype=torch.float32)


def _smooth_image(v: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
  """Low-frequency RGB in [0,1] -- a fair reconstruction target (random noise
  is the VAE's adversarial worst case and not representative)."""
  ys = torch.linspace(0, 3.14159, h).view(1, 1, h, 1)
  xs = torch.linspace(0, 3.14159, w).view(1, 1, 1, w)
  base = 0.5 + 0.5 * torch.sin(xs + ys)              # [1,1,h,w]
  img = base.expand(v, 3, h, w).clone()
  img[:, 1] = 0.5 + 0.5 * torch.cos(2 * xs)          # vary channels a bit
  return img.clamp(0.0, 1.0)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
  mse = torch.mean((a - b) ** 2).item()
  if mse == 0:
    return math.inf
  return 10.0 * math.log10(1.0 / mse)


def test_properties(adapter: VAEAdapter) -> None:
  assert adapter.downsample == 8
  assert adapter.latent_channels == 4
  assert adapter.scaling_factor > 0.0


def test_latent_shape(adapter: VAEAdapter) -> None:
  imgs = _smooth_image(v=4, h=256, w=256)
  z = adapter.encode(imgs)
  assert z.shape == (4, adapter.latent_channels, 256 // 8, 256 // 8)
  back = adapter.decode(z)
  assert back.shape == (4, 3, 256, 256)


def test_leading_dims_preserved(adapter: VAEAdapter) -> None:
  imgs = _smooth_image(v=1, h=64, w=64).unsqueeze(0).expand(2, 3, 3, 64, 64).clone()
  z = adapter.encode(imgs)
  assert z.shape == (2, 3, adapter.latent_channels, 8, 8)
  assert adapter.decode(z).shape == (2, 3, 3, 64, 64)


def test_roundtrip_psnr(adapter: VAEAdapter) -> None:
  imgs = _smooth_image(v=2, h=256, w=256)
  back = adapter.decode(adapter.encode(imgs))
  # a correct range map (+/-1) and scaling factor -> faithful reconstruction.
  # a range/scaling bug tanks this well below 20 dB.
  assert _psnr(imgs, back) > 22.0


def test_output_range(adapter: VAEAdapter) -> None:
  back = adapter.decode(adapter.encode(_smooth_image(v=2, h=64, w=64)))
  assert back.min() >= 0.0 and back.max() <= 1.0


def test_encode_deterministic(adapter: VAEAdapter) -> None:
  imgs = _smooth_image(v=2, h=64, w=64)
  assert torch.equal(adapter.encode(imgs), adapter.encode(imgs))  # mode(), not sample


def test_frozen(adapter: VAEAdapter) -> None:
  assert all(not p.requires_grad for p in adapter.parameters())
  assert sum(p.requires_grad for p in adapter.parameters()) == 0