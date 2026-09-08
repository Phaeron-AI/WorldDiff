from __future__ import annotations

# Third Party Import(s)
import torch

# Local Import(s)
from src.data.rays import RayEncoder


def make_intrinsics(n: int, height: int, width: int, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
  """Create simple pinhole camera intrinsics."""
  fx = 100.0
  fy = 100.0

  # Pixel-center convention:
  # u = col + 0.5
  # v = row + 0.5
  cx = width / 2.0 + 0.5
  cy = height / 2.0 + 0.5

  K = torch.zeros(
    n,
    3,
    3,
    device=device,
    dtype=dtype,
  )

  K[:, 0, 0] = fx
  K[:, 1, 1] = fy
  K[:, 0, 2] = cx
  K[:, 1, 2] = cy
  K[:, 2, 2] = 1.0

  return K


def make_identity_c2w(
    n: int,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Create n identity camera-to-world transforms."""

  return torch.eye(
    4,
    device=device,
    dtype=dtype,
  ).expand(n, -1, -1).clone()


def test_output_shape():
  """Output should preserve arbitrary leading dimensions."""

  H, W = 8, 10

  encoder = RayEncoder()

  # [V, 3, 3]
  K = make_intrinsics(4, H, W)

  # [V, 4, 4]
  c2w = make_identity_c2w(4)

  rays = encoder(K, c2w, H, W)

  assert rays.shape == (4, 6, H, W)

  # Also test [B, V, 3, 3]
  B, V = 2, 3

  K = make_intrinsics(B * V, H, W).reshape(B, V, 3, 3)
  c2w = make_identity_c2w(B * V).reshape(B, V, 4, 4)

  rays = encoder(K, c2w, H, W)

  assert rays.shape == (B, V, 6, H, W)


def test_direction_unit_norm():
  """Directions should have unit norm when normalize=True."""
  H, W = 8, 10

  encoder = RayEncoder(normalize=True)

  K = make_intrinsics(2, H, W)
  c2w = make_identity_c2w(2)

  rays = encoder(K, c2w, H, W)

  direction = rays[:, :3]

  norms = torch.linalg.vector_norm(
    direction,
    dim=1,
  )

  assert torch.allclose(
    norms,
    torch.ones_like(norms),
    atol=1e-5,
    rtol=1e-5,
  )


def test_plucker_constraint():
  """
  For Plücker coordinates:

    m = o × d

  therefore:

    d · m = 0
  """
  H, W = 8, 10

  encoder = RayEncoder()

  K = make_intrinsics(2, H, W)

  c2w = make_identity_c2w(2)

  # Give the cameras non-zero translations so that
  # the moment is actually non-zero.
  c2w[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
  c2w[1, :3, 3] = torch.tensor([-2.0, 1.0, 4.0])

  rays = encoder(K, c2w, H, W)

  direction = rays[:, :3]
  moment = rays[:, 3:]

  dot = (direction * moment).sum(dim=1)

  assert torch.allclose(
    dot,
    torch.zeros_like(dot),
    atol=1e-5,
    rtol=1e-5,
  )


def test_identity_camera():
  """
  With c2w = I:

    o = [0, 0, 0]

  so:

    m = o × d = 0.

  The principal-point ray should point along +Z.
  """
  H, W = 9, 11

  encoder = RayEncoder(normalize=True)

  K = make_intrinsics(1, H, W)
  c2w = make_identity_c2w(1)

  rays = encoder(K, c2w, H, W)

  direction = rays[0, :3]
  moment = rays[0, 3:]

  # Camera origin is zero -> all moments should be zero.
  assert torch.allclose(
    moment,
    torch.zeros_like(moment),
    atol=1e-6,
    rtol=1e-6,
  )

  # The principal point is placed at the center pixel.
  # For H=9, W=11:
  #
  #   cx = 11/2 + 0.5 = 6.0
  #   cy =  9/2 + 0.5 = 5.0
  #
  # Pixel center:
  #
  #   u = 5 + 0.5 = 5.5
  #   v = 4 + 0.5 = 4.5
  #
  # So instead use even dimensions where the principal point
  # lands exactly on a pixel center.
  H, W = 8, 10

  K = make_intrinsics(1, H, W)
  c2w = make_identity_c2w(1)

  rays = encoder(K, c2w, H, W)

  direction = rays[0, :3]

  # cx = 10/2 + 0.5 = 5.5
  # cy =  8/2 + 0.5 = 4.5
  #
  # Pixel center at:
  #   col=5 -> u=5.5
  #   row=4 -> v=4.5
  principal_ray = direction[:, 4, 5]

  assert torch.allclose(
    principal_ray,
    torch.tensor([0.0, 0.0, 1.0]),
    atol=1e-5,
    rtol=1e-5,
  )


def test_translation_changes_moment_not_direction():
  """
  Translating a camera should not change ray directions if
  rotation remains unchanged, but it should change moments.
  """
  H, W = 8, 10

  encoder = RayEncoder(normalize=True)

  K = make_intrinsics(2, H, W)

  c2w = make_identity_c2w(2)

  # Same rotation, different translations.
  c2w[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
  c2w[1, :3, 3] = torch.tensor([2.0, 3.0, 4.0])

  rays = encoder(K, c2w, H, W)

  direction_0 = rays[0, :3]
  direction_1 = rays[1, :3]

  moment_0 = rays[0, 3:]
  moment_1 = rays[1, 3:]

  # Translation alone must not affect direction.
  assert torch.allclose(
    direction_0,
    direction_1,
    atol=1e-6,
    rtol=1e-6,
  )

  # But moment must change.
  assert not torch.allclose(
    moment_0,
    moment_1,
    atol=1e-6,
    rtol=1e-6,
  )


def test_roundtrip_projection():
  """
  End-to-end geometric correctness.

  Construct a known world-space point X that lies on a particular
  pixel ray, project it back to the image, and verify that the
  encoder produces the same ray direction.
  """
  H, W = 8, 10

  encoder = RayEncoder(normalize=True)

  K = make_intrinsics(1, H, W)

  # Non-trivial camera translation.
  c2w = make_identity_c2w(1)
  c2w[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])

  # Pick a pixel center.
  row = 2
  col = 7

  u = col + 0.5
  v = row + 0.5

  pixel = torch.tensor(
    [u, v, 1.0],
    dtype=torch.float32,
  )

  # Independently construct the camera-space ray.
  K_single = K[0]

  d_cam = torch.linalg.solve(
    K_single,
    pixel,
  )

  d_cam = d_cam / torch.linalg.vector_norm(d_cam)

  # Convert it to world space.
  R = c2w[0, :3, :3]
  o = c2w[0, :3, 3]

  d_world = R @ d_cam

  # Pick a point three units along the ray.
  X = o + 3.0 * d_world

  # Project X back into camera coordinates.
  X_cam = R.T @ (X - o)

  projected = K_single @ X_cam

  projected_uv = projected[:2] / projected[2]

  # It should project back to exactly our selected pixel center.
  assert torch.allclose(
    projected_uv,
    torch.tensor([u, v]),
    atol=1e-5,
    rtol=1e-5,
  )


  # Encode all rays and inspect the selected pixel.
  rays = encoder(K, c2w, H, W)

  encoded_direction = rays[0, :3, row, col]

  assert torch.allclose(
    encoded_direction,
    d_world,
    atol=1e-5,
    rtol=1e-5,
  )