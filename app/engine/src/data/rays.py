from __future__ import annotations

# Native Imports

# Third Party Imports
import torch
from torch import Tensor

class RayEncoder:
  def __init__(self, *, normalize: bool = True, cache_grid: bool = True) -> None:
    self.normalize = normalize
    self.cache_grid = cache_grid
    self._grid_cache: dict[tuple[int, int, str, int | None, torch.dtype], Tensor] = {}

  def _pixel_grid(self, height: int, width: int, device, dtype) -> Tensor:
    key = (height, width, device.type, device.index, dtype)

    if self.cache_grid and key in self._grid_cache:
      return self._grid_cache[key]

    u = torch.arange(width, device=device, dtype=dtype) + 0.5   # pixel centers, cols
    v = torch.arange(height, device=device, dtype=dtype) + 0.5  # pixel centers, rows

    vv, uu = torch.meshgrid(v, u, indexing="ij")
    ones = torch.ones_like(uu)

    grid = torch.stack([uu, vv, ones], dim=0)  # [3, H, W]  (u, v, 1)

    if self.cache_grid:
      self._grid_cache[key] = grid

    return grid

  def __call__(self, intrinsics: Tensor, c2w: Tensor, height: int, width: int) -> Tensor:
    device = intrinsics.device
    dtype = intrinsics.dtype
    lead = intrinsics.shape[:-2]  # arbitrary leading dims (V, or B,V, ...)

    grid = self._pixel_grid(height, width, device, dtype)  # [3, H, W]
    grid_flat = grid.reshape(3, height * width)            # [3, HW]

    k_inv = torch.linalg.inv(intrinsics)                   # [..., 3, 3]
    d_cam = k_inv @ grid_flat                              # [..., 3, HW]

    rot = c2w[..., :3, :3]                                 # [..., 3, 3]
    d = rot @ d_cam                                        # [..., 3, HW]
    if self.normalize:
      d = d / torch.linalg.vector_norm(d, dim=-2, keepdim=True)

    o = c2w[..., :3, 3].unsqueeze(-1).expand_as(d)         # [..., 3, HW]
    m = torch.linalg.cross(o, d, dim=-2)                   # [..., 3, HW]

    rays = torch.cat([d, m], dim=-2)                       # [..., 6, HW]
    return rays.reshape(*lead, 6, height, width)