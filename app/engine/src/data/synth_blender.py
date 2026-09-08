from __future__ import annotations

# Native Import(s)
import math

# Third Party Imports(s)
import torch
from torch import Tensor
import torch.nn.functional as F

# Local Import(s)
from src.data.sample import SceneSample


# ----------------------------------------------------------------------
# Camera frame
# ----------------------------------------------------------------------

def look_at(
  eye: Tensor,                     # [3] world position of camera
  target: Tensor,                  # [3] world point to look at
  world_up: Tensor | None = None,  # [3] world up-hint, default [0,1,0]
) -> Tensor:                       # [4,4] c2w, OpenCV (+x right, +y down, +z fwd)

  if world_up is None:
    world_up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)

  f = F.normalize(target - eye, dim=-1)  # +z forward
  r = F.normalize(torch.linalg.cross(f, world_up, dim=-1), dim=-1)  # +x right
  u = torch.linalg.cross(f, r, dim=-1)  # +y down (unit)

  R = torch.stack([r, u, f], dim=1) # columns = axes

  c2w = torch.eye(4, device=eye.device, dtype=eye.dtype)
  c2w[:3, :3] = R
  c2w[:3, 3] = eye
  return c2w


def sample_cameras(
  num_views: int,
  *,
  rng: torch.Generator,
  radius_range: tuple[float, float] = (2.5, 4.0),
  elev_range: tuple[float, float] = (-0.3, 0.6),
  target: Tensor | None = None,
  fov_deg: float = 50.0,
  height: int,
  width: int,
  dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
  """Sample random cameras around a target point.

  Camera convention:
    +x = right, +y = down, +z = forward  (OpenCV)

  Returns:
    intrinsics: [V, 3, 3]
    c2w:        [V, 4, 4]
  """
  if num_views <= 0:
    raise ValueError(f"num_views must be positive, got {num_views}")

  if height <= 0 or width <= 0:
    raise ValueError(f"height and width must be positive, got {height} x {width}")

  if radius_range[0] <= 0 or radius_range[1] <= 0:
    raise ValueError("radius_range must contain positive values")

  if radius_range[0] > radius_range[1]:
    raise ValueError("radius_range must be ordered (min, max)")

  if elev_range[0] > elev_range[1]:
    raise ValueError("elev_range must be ordered (min, max)")

  if not (0.0 < fov_deg < 180.0):
    raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

  device = rng.device

  if target is None:
    target = torch.zeros(3, device=device, dtype=dtype)
  else:
    target = torch.as_tensor(target, device=device, dtype=dtype)
    if target.shape != (3,):
      raise ValueError(f"target must have shape [3], got {tuple(target.shape)}")

  # 1. spherical sampling -----------------------------------------------------
  #   x = r cos(elev) cos(azim),  y = r sin(elev),  z = r cos(elev) sin(azim)
  radius = torch.empty(num_views, device=device, dtype=dtype).uniform_(
    radius_range[0], radius_range[1], generator=rng
  )
  azimuth = torch.empty(num_views, device=device, dtype=dtype).uniform_(
    0.0, 2.0 * math.pi, generator=rng
  )
  elevation = torch.empty(num_views, device=device, dtype=dtype).uniform_(
    elev_range[0], elev_range[1], generator=rng
  )

  cos_elev = torch.cos(elevation)
  sin_elev = torch.sin(elevation)

  eye_offset = torch.stack(
    [
      radius * cos_elev * torch.cos(azimuth),
      radius * sin_elev,
      radius * cos_elev * torch.sin(azimuth),
    ],
    dim=-1,
  )  # [V, 3]
  eye = eye_offset + target  # [V, 3]

  # 2. look-at rotation (vectorised; same math as look_at) --------------------
  world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

  forward = F.normalize(target.unsqueeze(0) - eye, dim=-1)  # +z
  right = F.normalize(
    torch.linalg.cross(forward, world_up.expand_as(forward), dim=-1), dim=-1
  )  # +x
  down = torch.linalg.cross(forward, right, dim=-1)  # +y (unit)

  R = torch.stack([right, down, forward], dim=-1)  # [V, 3, 3], columns = axes

  c2w = torch.eye(4, device=device, dtype=dtype).expand(num_views, -1, -1).clone()
  c2w[:, :3, :3] = R
  c2w[:, :3, 3] = eye

  # 3. shared pinhole intrinsics ----------------------------------------------
  fov_rad = math.radians(fov_deg)
  fx = 0.5 * width / math.tan(0.5 * fov_rad)
  fy = fx
  cx = width / 2.0
  cy = height / 2.0

  K = torch.zeros(num_views, 3, 3, device=device, dtype=dtype)
  K[:, 0, 0] = fx
  K[:, 1, 1] = fy
  K[:, 0, 2] = cx
  K[:, 1, 2] = cy
  K[:, 2, 2] = 1.0

  return K, c2w


# ----------------------------------------------------------------------
# Scene content
# ----------------------------------------------------------------------

def make_point_cloud(
  *,
  rng: torch.Generator,
  num_points: int = 20_000,
  extent: float = 1.2,
  dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
  """Colored point cloud; colours are a smooth function of position.

  Returns:
    points: [N, 3] world-space
    colors: [N, 3] in [0, 1]
  """
  if num_points <= 0:
    raise ValueError(f"num_points must be positive, got {num_points}")
  if extent <= 0:
    raise ValueError(f"extent must be positive, got {extent}")

  points = torch.empty(num_points, 3, device=rng.device, dtype=dtype).uniform_(
    -extent, extent, generator=rng
  )

  freq = 2.5
  phase = torch.tensor([0.0, 2.0, 4.0], device=rng.device, dtype=dtype)
  colors = (0.5 + 0.5 * torch.sin(freq * points + phase)).clamp(0.0, 1.0)

  return points, colors


# ----------------------------------------------------------------------
# Rasterizer  (deterministic z-buffer)
# ----------------------------------------------------------------------

def render(
  points: Tensor,      # [N,3]
  colors: Tensor,      # [N,3]
  intrinsics: Tensor,  # [V,3,3]
  c2w: Tensor,         # [V,4,4]
  height: int,
  width: int,
  *,
  point_radius: int = 1,
  bg: float = 0.0,
  near: float = 1e-3,
) -> Tensor:           # [V,3,H,W]
  """Project a colored point cloud into each camera with a true z-buffer.

  Per view: world -> camera -> pixels; splat each point as a small disk; then
  keep, per pixel, the colour of the NEAREST sample. Selection is a stable sort
  (group-by-pixel, depth-ordered) + first-occurrence mask, so it is fully
  deterministic on CPU and CUDA -- unlike duplicate-index scatter, whose write
  order is undefined.
  """
  if points.ndim != 2 or points.shape[-1] != 3:
    raise ValueError(f"points must have shape [N,3], got {tuple(points.shape)}")
  if colors.shape != points.shape:
    raise ValueError(
      f"colors must have shape {tuple(points.shape)}, got {tuple(colors.shape)}"
    )
  if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
    raise ValueError(f"intrinsics must have shape [V,3,3], got {tuple(intrinsics.shape)}")
  if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
    raise ValueError(f"c2w must have shape [V,4,4], got {tuple(c2w.shape)}")
  if intrinsics.shape[0] != c2w.shape[0]:
    raise ValueError("intrinsics and c2w must have the same number of views")
  if height <= 0 or width <= 0:
    raise ValueError(f"height and width must be positive, got {height} x {width}")
  if point_radius < 0:
    raise ValueError(f"point_radius must be >= 0, got {point_radius}")
  if not (0.0 <= bg <= 1.0):
    raise ValueError(f"bg must be in [0,1], got {bg}")

  num_views = intrinsics.shape[0]
  images = torch.full(
    (num_views, 3, height, width),
    fill_value=bg,
    device=points.device,
    dtype=colors.dtype,
  )

  # disk offsets, computed once
  offsets = [
    (dx, dy)
    for dx in range(-point_radius, point_radius + 1)
    for dy in range(-point_radius, point_radius + 1)
    if dx * dx + dy * dy <= point_radius * point_radius
  ]

  for v in range(num_views):
    K = intrinsics[v]
    R = c2w[v, :3, :3]
    o = c2w[v, :3, 3]

    # world -> camera:  X_cam = R^T (X - o)  ==  (X - o) @ R
    Xc = (points - o) @ R
    z = Xc[:, 2]

    front = z > near  # in front of the near plane
    if not front.any():
      continue
    Xc = Xc[front]
    z = z[front]
    col = colors[front]

    proj = Xc @ K.T  # [M,3]
    px = proj[:, 0] / proj[:, 2]
    py = proj[:, 1] / proj[:, 2]

    # pixel index of the disk centre (pixel centre = idx + 0.5)
    cx = torch.round(px - 0.5).long()
    cy = torch.round(py - 0.5).long()

    # splat: expand each point over the disk, carrying its depth
    xs, ys, cs, zs = [], [], [], []
    for dx, dy in offsets:
      xs.append(cx + dx)
      ys.append(cy + dy)
      cs.append(col)
      zs.append(z)
    x = torch.cat(xs)
    y = torch.cat(ys)
    c = torch.cat(cs)
    zc = torch.cat(zs)

    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not inside.any():
      continue
    x, y, c, zc = x[inside], y[inside], c[inside], zc[inside]

    # deterministic z-buffer: nearest sample per pixel wins
    flat = y * width + x                          # [P] pixel id
    order_z = torch.argsort(zc)                   # ascending depth
    flat_z, col_z = flat[order_z], c[order_z]
    order_f = torch.argsort(flat_z, stable=True)  # group by pixel, depth preserved
    flat_s, col_s = flat_z[order_f], col_z[order_f]

    first = torch.ones_like(flat_s, dtype=torch.bool)
    first[1:] = flat_s[1:] != flat_s[:-1]         # first per pixel == nearest

    images[v].view(3, -1)[:, flat_s[first]] = col_s[first].T

  return images


# ----------------------------------------------------------------------
# Top-level scene generator
# ----------------------------------------------------------------------

def synth_scene(
  *,
  seed: int,
  num_views: int = 4,
  num_input: int = 2,
  height: int = 256,
  width: int = 256,
  num_points: int = 20_000,
  dtype: torch.dtype = torch.float32,
) -> SceneSample:
  """Generate one deterministic synthetic multi-view scene."""
  if num_views <= 0:
    raise ValueError(f"num_views must be positive, got {num_views}")
  if not (1 <= num_input < num_views):
    raise ValueError(
      f"num_input must satisfy 1 <= num_input < num_views, "
      f"got num_input={num_input}, num_views={num_views}"
    )

  # one local RNG -- no global torch.rand()
  rng = torch.Generator()
  rng.manual_seed(seed)

  K, c2w = sample_cameras(
    num_views=num_views, rng=rng, height=height, width=width, dtype=dtype
  )
  points, colors = make_point_cloud(rng=rng, num_points=num_points, dtype=dtype)
  images = render(
    points=points, colors=colors, intrinsics=K, c2w=c2w, height=height, width=width
  )

  cond_mask = torch.zeros(num_views, dtype=torch.bool)
  cond_mask[:num_input] = True

  scene_id = f"synth-{seed:06d}"

  return SceneSample(images, K, c2w, cond_mask, scene_id)