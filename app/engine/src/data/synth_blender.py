from __future__ import annotations

# Native Import(s)
import math

# Third Party Imports(s)
import torch
from torch import Tensor, Generator
import torch.nn.functional as F

# Local Import(s)
from sample import SceneSample

def look_at(eye: Tensor, target: Tensor, world_up: Tensor | None = None) -> Tensor:
  f  = F.normalize(target - eye)
  if world_up is None:
    world_up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)
  r = F.normalize(torch.cross(f, world_up))
  u = F.normalize(torch.cross(f, r))
  R = torch.stack([r, u, f], dim=1)

  return torch.tensor([R, eye])

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
  """
  Sample random cameras around a target point.

  Camera convention:
    +x = right
    +y = down
    +z = forward

  Returns:
    intrinsics: [V, 3, 3]
    c2w:       [V, 4, 4]
  """

  if num_views <= 0:
    raise ValueError(f"num_views must be positive, got {num_views}")

  if height <= 0 or width <= 0:
    raise ValueError(
      f"height and width must be positive, got {height} x {width}"
    )

  if radius_range[0] <= 0 or radius_range[1] <= 0:
    raise ValueError("radius_range must contain positive values")

  if radius_range[0] > radius_range[1]:
    raise ValueError("radius_range must be ordered (min, max)")

  if elev_range[0] > elev_range[1]:
    raise ValueError("elev_range must be ordered (min, max)")

  if not (0.0 < fov_deg < 180.0):
    raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

  # ------------------------------------------------------------
  # Device
  # ------------------------------------------------------------

  device = rng.device

  # ------------------------------------------------------------
  # Target
  # ------------------------------------------------------------

  if target is None:
    target = torch.zeros(
      3,
      device=device,
      dtype=dtype,
    )
  else:
    target = torch.as_tensor(
      target,
      device=device,
      dtype=dtype,
    )

    if target.shape != (3,):
      raise ValueError(
        f"target must have shape [3], got {tuple(target.shape)}"
      )

  # ------------------------------------------------------------
  # 1. Sample spherical coordinates
  #
  # radius    ∈ [r_min, r_max]
  # azimuth   ∈ [0, 2π)
  # elevation ∈ [e_min, e_max]
  #
  # We use:
  #
  # x = r cos(elev) cos(azimuth)
  # y = r sin(elev)
  # z = r cos(elev) sin(azimuth)
  # ------------------------------------------------------------

  radius = torch.empty(
    num_views,
    device=device,
    dtype=dtype,
  ).uniform_(
    radius_range[0],
    radius_range[1],
    generator=rng,
  )

  azimuth = torch.empty(
    num_views,
    device=device,
    dtype=dtype,
  ).uniform_(
    0.0,
    2.0 * math.pi,
    generator=rng,
  )

  elevation = torch.empty(
    num_views,
    device=device,
    dtype=dtype,
  ).uniform_(
    elev_range[0],
    elev_range[1],
    generator=rng,
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
  )
  # [V, 3]

  eye = eye_offset + target
  # [V, 3]

  # ------------------------------------------------------------
  # 2. Build look-at rotation
  #
  # OpenCV convention:
  #
  #   forward = +Z
  #   right   = +X
  #   down    = +Y
  #
  # We first construct the usual world-space "up" vector,
  # then convert it to camera-space down.
  # ------------------------------------------------------------

  world_up = torch.tensor(
    [0.0, 1.0, 0.0],
    device=device,
    dtype=dtype,
  )

  forward = target.unsqueeze(0) - eye
  forward = torch.nn.functional.normalize(
    forward,
    dim=-1,
  )
  # [V, 3]

  # right = forward × world_up
  right = torch.cross(
    forward,
    world_up.expand_as(forward),
    dim=-1,
  )

  right = torch.nn.functional.normalize(
    right,
    dim=-1,
  )
  # [V, 3]

  # Camera +Y is DOWN, whereas world_up points UP.
  #
  # up = right × forward
  # down = -up
  #
  # Therefore:
  #
  # down = forward × right
  down = torch.cross(
    forward,
    right,
    dim=-1,
  )

  down = torch.nn.functional.normalize(
    down,
    dim=-1,
  )
  # [V, 3]

  # ------------------------------------------------------------
  # 3. Construct camera-to-world rotation
  #
  # Columns are the camera basis vectors expressed in world space:
  #
  #   column 0 = +X = right
  #   column 1 = +Y = down
  #   column 2 = +Z = forward
  # ------------------------------------------------------------

  R = torch.stack(
    [right, down, forward],
    dim=-1,
  )
  # [V, 3, 3]

  # ------------------------------------------------------------
  # 4. Construct homogeneous camera-to-world matrices
  # ------------------------------------------------------------

  c2w = torch.eye(
    4,
    device=device,
    dtype=dtype,
  ).expand(
    num_views,
    -1,
    -1,
  ).clone()

  c2w[:, :3, :3] = R
  c2w[:, :3, 3] = eye

  # ------------------------------------------------------------
  # 5. Shared pinhole intrinsics
  #
  # fx = fy = 0.5 * width / tan(0.5 * fov)
  # cx = width / 2
  # cy = height / 2
  # ------------------------------------------------------------

  fov_rad = math.radians(fov_deg)

  fx = 0.5 * width / math.tan(0.5 * fov_rad)
  fy = fx

  cx = width / 2.0
  cy = height / 2.0

  K = torch.zeros(
    num_views,
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

  return K, c2w


def make_point_cloud(
  *,
  rng: torch.Generator,
  num_points: int = 20_000,
  extent: float = 1.2,
  dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
  """
  Create a synthetic colored point cloud.

  Returns:
    points: [N, 3] world-space coordinates
    colors: [N, 3] RGB values in [0, 1]
  """

  if num_points <= 0:
    raise ValueError(
      f"num_points must be positive, got {num_points}"
    )

  if extent <= 0:
    raise ValueError(
      f"extent must be positive, got {extent}"
    )

  # ------------------------------------------------------------
  # 1. Uniformly sample points inside the cube
  #
  # [-extent, extent]^3
  # ------------------------------------------------------------

  points = torch.empty(
    num_points,
    3,
    device=rng.device,
    dtype=dtype,
  ).uniform_(
    -extent,
    extent,
    generator=rng,
  )

  # ------------------------------------------------------------
  # 2. Smooth position-dependent colors
  #
  # Nearby points receive similar colors.
  #
  # color = 0.5 + 0.5 * sin(freq * position + phase)
  # ------------------------------------------------------------

  freq = 2.5

  # Different phase per channel makes RGB less correlated.
  phase = torch.tensor(
    [0.0, 2.0, 4.0],
    device=rng.device,
    dtype=dtype,
  )

  colors = 0.5 + 0.5 * torch.sin(
    freq * points + phase
  )

  colors = colors.clamp(0.0, 1.0)

  return points, colors


# ----------------------------------------------------------------------
# Rasterizer
# ----------------------------------------------------------------------

def render(
  points: Tensor,                  # [N,3]
  colors: Tensor,                  # [N,3]
  intrinsics: Tensor,              # [V,3,3]
  c2w: Tensor,                     # [V,4,4]
  height: int,
  width: int,
  *,
  point_radius: int = 1,
  bg: float = 0.0,
) -> Tensor:                       # [V,3,H,W]
  """
  Project a colored point cloud into each camera.

  Painter's algorithm:
    1. transform world -> camera
    2. project to pixels
    3. sort visible points far -> near
    4. splat colors
    5. near points overwrite far points
  """

  # ------------------------------------------------------------
  # Validation
  # ------------------------------------------------------------

  if points.ndim != 2 or points.shape[-1] != 3:
    raise ValueError(
      f"points must have shape [N,3], got {tuple(points.shape)}"
    )

  if colors.shape != points.shape:
    raise ValueError(
      f"colors must have shape {tuple(points.shape)}, "
      f"got {tuple(colors.shape)}"
    )

  if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
    raise ValueError(
      "intrinsics must have shape [V,3,3], "
      f"got {tuple(intrinsics.shape)}"
    )

  if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
    raise ValueError(
      "c2w must have shape [V,4,4], "
      f"got {tuple(c2w.shape)}"
    )

  if intrinsics.shape[0] != c2w.shape[0]:
    raise ValueError(
      "intrinsics and c2w must have the same number of views"
    )

  if height <= 0 or width <= 0:
    raise ValueError(
      f"height and width must be positive, got {height} x {width}"
    )

  if point_radius < 0:
    raise ValueError(
      f"point_radius must be >= 0, got {point_radius}"
    )

  if not (0.0 <= bg <= 1.0):
    raise ValueError(
      f"bg must be in [0,1], got {bg}"
    )

  # ------------------------------------------------------------
  # Setup
  # ------------------------------------------------------------

  num_views = intrinsics.shape[0]
  num_points = points.shape[0]

  images = torch.full(
    (
      num_views,
      3,
      height,
      width,
    ),
    fill_value=bg,
    device=points.device,
    dtype=colors.dtype,
  )

  eps = torch.finfo(points.dtype).eps

  # ------------------------------------------------------------
  # Render each view
  # ------------------------------------------------------------

  for v in range(num_views):

    K = intrinsics[v]
    R = c2w[v, :3, :3]
    o = c2w[v, :3, 3]

    # ----------------------------------------------------------
    # World -> camera
    #
    # X_cam = R^T (X_world - o)
    #
    # Since X is [N,3], the equivalent row-vector expression is:
    #
    # X_cam = (X - o) @ R
    # ----------------------------------------------------------

    Xc = (points - o) @ R
    # [N,3]

    z = Xc[:, 2]

    # ----------------------------------------------------------
    # Keep points in front of camera
    # ----------------------------------------------------------

    visible = z > eps

    if not visible.any():
      continue

    Xc = Xc[visible]
    z = z[visible]
    col = colors[visible]

    # ----------------------------------------------------------
    # Perspective projection
    #
    # [u',v',w']^T = K X_cam
    #
    # px = u'/w'
    # py = v'/w'
    # ----------------------------------------------------------

    proj = Xc @ K.T
    # [M,3]

    px = proj[:, 0] / proj[:, 2]
    py = proj[:, 1] / proj[:, 2]

    # ----------------------------------------------------------
    # Cull points outside image
    #
    # Pixel coordinates use the continuous image domain:
    #
    #   x ∈ [0, W)
    #   y ∈ [0, H)
    # ----------------------------------------------------------

    inside = (
      (px >= 0.0)
      & (px < width)
      & (py >= 0.0)
      & (py < height)
    )

    if not inside.any():
      continue

    px = px[inside]
    py = py[inside]
    z = z[inside]
    col = col[inside]

    # ----------------------------------------------------------
    # Painter's algorithm
    #
    # Far -> near.
    #
    # Later writes overwrite earlier writes.
    # ----------------------------------------------------------

    order = torch.argsort(
      z,
      descending=True,
    )

    px = px[order]
    py = py[order]
    col = col[order]

    # ----------------------------------------------------------
    # Convert continuous pixel coordinates to pixel centers.
    #
    # Renderer convention:
    #
    #   pixel center = integer + 0.5
    #
    # Therefore:
    #
    #   pixel index = round(coord - 0.5)
    # ----------------------------------------------------------

    center_x = torch.round(px - 0.5).long()
    center_y = torch.round(py - 0.5).long()

    # ----------------------------------------------------------
    # Splat each point as a small disk.
    # ----------------------------------------------------------

    for dx in range(-point_radius, point_radius + 1):
      for dy in range(-point_radius, point_radius + 1):

        # Disk rather than square.
        if dx * dx + dy * dy > point_radius * point_radius:
          continue

        x = center_x + dx
        y = center_y + dy

        valid = (
          (x >= 0)
          & (x < width)
          & (y >= 0)
          & (y < height)
        )

        if not valid.any():
          continue

        x = x[valid]
        y = y[valid]
        c = col[valid]

        # Because points were sorted far -> near,
        # later assignments correspond to nearer points.
        #
        # Advanced indexing assignment gives us the desired
        # painter's behavior for this simple P1 renderer.
        images[v, :, y, x] = c.T

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
  """
  Generate one deterministic synthetic multi-view scene.
  """

  if num_views <= 0:
    raise ValueError(
      f"num_views must be positive, got {num_views}"
    )

  if not (1 <= num_input < num_views):
    raise ValueError(
      f"num_input must satisfy 1 <= num_input < num_views, "
      f"got num_input={num_input}, num_views={num_views}"
    )

  # ------------------------------------------------------------
  # One local RNG.
  #
  # No global torch.rand().
  # ------------------------------------------------------------

  rng = torch.Generator()

  rng.manual_seed(seed)

  # ------------------------------------------------------------
  # Camera sampling
  # ------------------------------------------------------------

  K, c2w = sample_cameras(
    num_views=num_views,
    rng=rng,
    height=height,
    width=width,
    dtype=dtype,
  )

  # ------------------------------------------------------------
  # Point cloud
  # ------------------------------------------------------------

  points, colors = make_point_cloud(
    rng=rng,
    num_points=num_points,
    dtype=dtype,
  )

  # ------------------------------------------------------------
  # Render
  # ------------------------------------------------------------

  images = render(
    points=points,
    colors=colors,
    intrinsics=K,
    c2w=c2w,
    height=height,
    width=width,
  )

  # ------------------------------------------------------------
  # Conditioning mask
  #
  # First num_input views = conditioning views.
  # Remaining views = target/query views.
  # ------------------------------------------------------------

  cond_mask = torch.zeros(
    num_views,
    dtype=torch.bool,
  )

  cond_mask[:num_input] = True

  # ------------------------------------------------------------
  # Scene identifier
  # ------------------------------------------------------------

  scene_id = f"synth-{seed:06d}"

  # ------------------------------------------------------------
  # SceneSample
  #
  # Its own asserts are the final contract guard.
  # ------------------------------------------------------------

  return SceneSample(
    images,
    K,
    c2w,
    cond_mask,
    scene_id,
  )
