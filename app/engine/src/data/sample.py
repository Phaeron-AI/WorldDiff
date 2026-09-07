from __future__ import annotations

# Native Imports
from dataclasses import dataclass, replace

# Third Party Imports
import torch
from torch import Tensor

# Scene Sample Data Class
@dataclass
class SceneSample:
  images: Tensor  # (V, 3, H, W)
  intrinsics: Tensor  # (V, 3, 3)
  c2w: Tensor  # (V, 4, 4)
  cond_mask: Tensor  # (V,)
  scene_id: str

  @property
  def num_views(self) -> int: 
    return self.images.shape[0]

  @property
  def input_idx(self) -> Tensor: 
    return torch.where(self.cond_mask)[0]

  @property
  def target_idx(self) -> Tensor: 
    return torch.where(-self.cond_mask)[0]

  def to(self, device: torch.device | str) -> "SceneSample": 
    return replace(
      self,
      images=self.images.to(device),
      intrinsics=self.intrinsics.to(device),
      c2w=self.c2w.to(device),
      cond_mask=self.cond_mask.to(device),
    )

  def __post_init__(self) -> None:
    """
    Initialize Scene Sample object and validate the following properties:
    - images: 4D Tensor of shape (V, 3, H, W)
    - intrinsics: 3D Tensor of shape (V, 3, 3)
    - c2w: 3D Tensor of shape (V, 4, 4) with bottom-right element = 1, i.e [0, 0, 0, 1]
    - cond_mask: 1D Tensor of shape (V,) with dtype = torch.bool
    - All Tensors (except cond_mask) should be on the same device and have the same floating type (torch.float32)
    """

    v = self.images.shape[0]

    assert self.images.ndim == 4 and self.images.shape[1] == 3, (
      f"Expected images of shape (V, 3, H, W); got {tuple(self.images.shape)}"
    )
    assert self.images.is_floating_point(), (
      f"Expected floating-point images; got dtype {self.images.dtype}"
    )
    assert self.intrinsics.shape == (v, 3, 3), (
      f"Expected intrinsics of shape ({v}, 3, 3); got {tuple(self.intrinsics.shape)}"
    )
    assert self.c2w.shape == (v, 4, 4), (
      f"Expected c2w of shape ({v}, 4, 4); got {tuple(self.c2w.shape)}"
    )
    assert self.cond_mask.shape == (v,), (
      f"Expected cond_mask of shape ({v},); got {tuple(self.cond_mask.shape)}"
    )

    bottom = self.c2w[:, 3, :]
    expected = torch.tensor(
      [0.0, 0.0, 0.0, 1.0], device=self.c2w.device, dtype=self.c2w.dtype
    ).expand(v, 4)
    assert torch.allclose(bottom, expected), (
      f"Expected c2w bottom row [0,0,0,1] per view; got {bottom.tolist()}"
    )

    assert self.cond_mask.dtype == torch.bool, (
      f"Expected cond_mask dtype torch.bool; got {self.cond_mask.dtype}"
    )
    assert self.cond_mask.any() and (~self.cond_mask).any(), (
      "cond_mask must have at least one input (True) and one target (False) view"
    )

    devices = {
      self.images.device,
      self.intrinsics.device,
      self.c2w.device,
      self.cond_mask.device,
    }
    assert len(devices) == 1, f"All tensors must share a device; got {devices}"
    assert self.images.dtype == self.intrinsics.dtype == self.c2w.dtype, (
      f"images/intrinsics/c2w must share dtype; got "
      f"{self.images.dtype}, {self.intrinsics.dtype}, {self.c2w.dtype}"
    )
