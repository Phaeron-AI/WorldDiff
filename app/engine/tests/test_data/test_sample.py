from __future__ import annotations

import pytest
import torch

from src.data.sample import SceneSample


def _sample(
  V: int = 2,
  H: int = 8,
  W: int = 8,
  mask: torch.Tensor | None = None,
  dtype: torch.dtype = torch.float32,
) -> SceneSample:
  return SceneSample(
    images=torch.rand(V, 3, H, W, dtype=dtype),
    intrinsics=torch.eye(3, dtype=dtype).expand(V, 3, 3).contiguous(),
    c2w=torch.eye(4, dtype=dtype).expand(V, 4, 4).contiguous(),
    cond_mask=torch.tensor([True, False]) if mask is None else mask,
    scene_id="s0",
  )


def test_valid_sample_and_properties() -> None:
  s = _sample()
  assert s.num_views == 2
  assert s.input_idx.tolist() == [0]
  assert s.target_idx.tolist() == [1]


def test_to_moves_all_tensors() -> None:
  s = _sample().to("cpu")
  assert s.num_views == 2
  assert s.images.device.type == "cpu"
  assert s.cond_mask.device.type == "cpu"


def test_rejects_bad_intrinsics_shape() -> None:
  s = _sample()
  with pytest.raises(AssertionError):
    SceneSample(
      images=s.images,
      intrinsics=s.intrinsics[:, :, :2],  # (V, 3, 2)
      c2w=s.c2w,
      cond_mask=s.cond_mask,
      scene_id=s.scene_id,
    )


def test_rejects_nonhomogeneous_c2w() -> None:
  s = _sample()
  bad = s.c2w.clone()
  bad[:, 3, :] = torch.tensor([1.0, 0.0, 0.0, 2.0])
  with pytest.raises(AssertionError):
    SceneSample(
      images=s.images, intrinsics=s.intrinsics, c2w=bad,
      cond_mask=s.cond_mask, scene_id=s.scene_id,
    )


def test_rejects_all_input_mask() -> None:
  with pytest.raises(AssertionError):
    _sample(mask=torch.tensor([True, True]))


def test_rejects_all_target_mask() -> None:
  with pytest.raises(AssertionError):
    _sample(mask=torch.tensor([False, False]))


def test_rejects_channel_mismatch() -> None:
  s = _sample()
  with pytest.raises(AssertionError):
    SceneSample(
      images=torch.rand(2, 1, 8, 8), intrinsics=s.intrinsics, c2w=s.c2w,
      cond_mask=s.cond_mask, scene_id=s.scene_id,
    )


def test_rejects_cond_mask_wrong_length() -> None:
  s = _sample()
  with pytest.raises(AssertionError):
    SceneSample(
      images=s.images, intrinsics=s.intrinsics, c2w=s.c2w,
      cond_mask=torch.tensor([True, False, True]), scene_id=s.scene_id,
    )


def test_rejects_non_float_images() -> None:
  s = _sample()
  with pytest.raises(AssertionError):
    SceneSample(
      images=(torch.rand(2, 3, 8, 8) * 255).to(torch.uint8),
      intrinsics=s.intrinsics, c2w=s.c2w, cond_mask=s.cond_mask, scene_id=s.scene_id,
    )


def test_rejects_dtype_mismatch() -> None:
  s = _sample()
  with pytest.raises(AssertionError):
    SceneSample(
      images=s.images.to(torch.float64), intrinsics=s.intrinsics, c2w=s.c2w,
      cond_mask=s.cond_mask, scene_id=s.scene_id,
    )