from __future__ import annotations

import hashlib

from collections.abc import Sequence
from typing import Any, TypedDict

import torch

from src.eval.metrics import pooled_mse, pooled_psnr
from src.models.flow import sample
from src.models.flow.conditioning import apply_conditioning
from src.models.flow.reverse.guidance import cfg_velocity
from src.models.flow.forward.interpolant import sample_noise
from src.models.flow.reverse.sampler import euler_step, init_target_noise
from src.models.flow.reverse.scheduler import RectifiedFlowScheduler


K_ROUTINE = 8
W_ROUTINE = 1.0


class SceneItem(TypedDict):
  latents: torch.Tensor
  rays: torch.Tensor
  cond_mask: torch.Tensor
  scene_id: str


def _validation_seed(
  val_seed: int,
  scene_id: str,
  k: int,
) -> int:
  payload = f"{val_seed}\0{scene_id}\0{k}".encode("utf-8")
  digest = hashlib.sha256(payload).digest()
  return int.from_bytes(
    digest[:8],
    "little",
    signed=False,
  )


def _substream(
  base: int,
  tag: str,
) -> int:
  if tag not in {"u", "c", "c2"}:
    raise ValueError(
      f"unknown validation RNG substream: {tag!r}"
    )

  digest = hashlib.sha256(
    f"{base}\0{tag}".encode("utf-8")
  ).digest()

  return int.from_bytes(
    digest[:8],
    "little",
    signed=False,
  )


def _generator(
  seed: int,
  device: torch.device,
) -> torch.Generator:
  generator = torch.Generator(device=device)
  generator.manual_seed(seed)
  return generator


def _validate_scene_shapes(
  z0: torch.Tensor,
  rays: torch.Tensor,
  cond_mask: torch.Tensor,
) -> None:
  if z0.ndim != 5:
    raise ValueError(
      "latents must have shape [B,V,C,H,W]"
    )

  if (
    rays.ndim != 5
    or rays.shape[:2] != z0.shape[:2]
    or rays.shape[2] != 6
    or rays.shape[3:] != z0.shape[3:]
  ):
    raise ValueError(
      "rays must have shape [B,V,6,H,W] "
      "matching latents"
    )

  if (
    cond_mask.ndim != 2
    or cond_mask.shape != z0.shape[:2]
  ):
    raise ValueError(
      "cond_mask must have shape [B,V]"
    )

  if cond_mask.dtype != torch.bool:
    raise ValueError(
      "cond_mask must be bool"
    )


def guided_sample(
  model: torch.nn.Module,
  z0: torch.Tensor,
  rays: torch.Tensor,
  cond_mask: torch.Tensor,
  *,
  w: float,
  num_steps: int,
  g_uncond: torch.Generator,
  g_cond: torch.Generator,
) -> torch.Tensor:
  """Reference two-state CFG sampler; returns final conditional state."""
  if num_steps <= 0:
    raise ValueError("K must be positive")

  _validate_scene_shapes(
    z0,
    rays,
    cond_mask,
  )

  cond_mask = cond_mask.to(
    device=z0.device,
    dtype=torch.bool,
  )

  uncond_mask = torch.zeros_like(
    cond_mask,
    dtype=torch.bool,
  )

  batch = z0.shape[0]

  # Independent initial states.
  z_u = sample_noise(
    z0,
    rng=g_uncond,
  )

  z_c = init_target_noise(
    z0,
    cond_mask,
    rng=g_cond,
  )

  scheduler = RectifiedFlowScheduler(
    num_steps=num_steps,
    device=z0.device,
    dtype=z0.dtype,
  )

  times = scheduler.step_times()
  step_sizes = scheduler.step_size()

  with torch.inference_mode():
    for i in range(num_steps):
      t = times[i].expand(batch)

      # Separate unconditional and conditional forwards.
      v_u = model(
        z_u,
        rays,
        t,
        cond_mask=uncond_mask,
      )

      v_c = model(
        z_c,
        rays,
        t,
        cond_mask=cond_mask,
      )

      if (
        v_u.shape != z_u.shape
        or v_c.shape != z_c.shape
      ):
        raise ValueError(
          "model velocity shape must match "
          "latent state shape"
        )

      # Caller owns argument order.
      v_cfg = cfg_velocity(
        v_u,
        v_c,
        float(w),
      )

      dt = step_sizes[i]

      # Unconditional state is driven only by v_u.
      # It is never re-pinned.
      z_u = euler_step(
        z_u,
        v_u,
        dt,
      )

      # Conditional state is driven by CFG velocity
      # and then re-pinned on observed views.
      z_c = euler_step(
        z_c,
        v_cfg,
        dt,
      )

      z_c = apply_conditioning(
        z_c,
        z0,
        cond_mask,
      )

  return z_c


def _conditional_sample(
  model: torch.nn.Module,
  z0: torch.Tensor,
  rays: torch.Tensor,
  cond_mask: torch.Tensor,
  *,
  num_steps: int,
  rng: torch.Generator,
) -> torch.Tensor:
  """Routine path using the existing conditional sampler."""
  return sample(
    model,
    z0,
    rays,
    cond_mask,
    num_steps=num_steps,
    rng=rng,
  )


def validate_routine(
  model: torch.nn.Module,
  val_set: Sequence[SceneItem],
  *,
  vae: Any,
  val_seed: int,
  device: torch.device,
  num_input: int,
) -> dict[str, float]:
  """Run fixed routine validation and return tracker-compatible metrics."""
  if K_ROUTINE <= 0:
    raise ValueError(
      "K_ROUTINE must be positive"
    )

  if num_input <= 0:
    raise ValueError(
      "num_input must be positive"
    )

  if not val_set:
    raise ValueError(
      "validation set is empty"
    )

  was_training = model.training
  model.eval()

  latent_sse = torch.zeros(
    (),
    device=device,
    dtype=torch.float64,
  )

  latent_count = torch.zeros(
    (),
    device=device,
    dtype=torch.float64,
  )

  pixel_sse = torch.zeros(
    (),
    device=device,
    dtype=torch.float64,
  )

  pixel_count = torch.zeros(
    (),
    device=device,
    dtype=torch.float64,
  )

  # Second conditional draw, for the dispersion estimate.
  pair_sse = torch.zeros(
    (),
    device=device,
    dtype=torch.float64,
  )

  draw2_sse = torch.zeros(
    (),
    device=device,
    dtype=torch.float64,
  )

  try:
    with torch.inference_mode():
      for scene in val_set:
        scene_id = scene["scene_id"]

        # Loader item:
        # latents:   [V,C,H,W]
        # rays:      [V,6,H,W]
        # cond_mask: [V]
        #
        # Model API:
        # latents:   [1,V,C,H,W]
        # rays:      [1,V,6,H,W]
        # cond_mask: [1,V]

        z0 = scene["latents"].to(
          device=device,
        ).unsqueeze(0)

        rays = scene["rays"].to(
          device=device,
        ).unsqueeze(0)

        cond_mask = scene["cond_mask"].to(
          device=device,
          dtype=torch.bool,
        ).unsqueeze(0)

        _validate_scene_shapes(
          z0,
          rays,
          cond_mask,
        )

        target_mask = ~cond_mask

        if not torch.any(target_mask):
          raise ValueError(
            f"scene {scene_id!r} has no target views"
          )

        k = int(
          cond_mask.sum().item()
        )

        if k <= 0:
          raise ValueError(
            f"scene {scene_id!r} has K <= 0 "
            "conditioning views"
          )

        # Derivation 75:
        # routine validation must use exactly num_input
        # observed views.
        if k != num_input:
          raise ValueError(
            f"scene {scene_id!r} has k={k} "
            f"conditioning views, expected "
            f"num_input={num_input}"
          )

        # Per-scene deterministic seed.
        base = _validation_seed(
          val_seed,
          scene_id,
          k,
        )

        g_cond = _generator(
          _substream(base, "c"),
          z0.device,
        )

        prediction = _conditional_sample(
          model,
          z0,
          rays,
          cond_mask,
          num_steps=K_ROUTINE,
          rng=g_cond,
        )

        # Check once after the trajectory rather than
        # synchronizing the device at every step.
        if not torch.isfinite(
          prediction
        ).all():
          raise FloatingPointError(
            f"non-finite final validation state "
            f"in scene {scene_id!r}"
          )

        # ------------------------------------------------------------
        # Latent metric
        # ------------------------------------------------------------
        #
        # Deliberately NOT using pooled_mse here.
        # Latents are not image values and must remain unclamped.
        #
        pred_target = prediction[target_mask]
        ref_target = z0[target_mask]

        latent_diff = (
          pred_target - ref_target
        )

        latent_sse += latent_diff.square().sum(
          dtype=torch.float64,
        )

        latent_count += latent_diff.numel()

        # ------------------------------------------------------------
        # Pixel metric
        # ------------------------------------------------------------
        #
        # Decode only target views.
        #
        decoded_prediction = vae.decode(
          pred_target.unsqueeze(0),
        )

        decoded_reference = vae.decode(
          ref_target.unsqueeze(0),
        )

        # pooled_mse owns the image-domain clamp internally.
        scene_sse, scene_count = pooled_mse(
          decoded_prediction,
          decoded_reference,
        )

        pixel_sse += scene_sse
        pixel_count += scene_count

        # ------------------------------------------------------------
        # Dispersion
        # ------------------------------------------------------------
        #
        # A second independent draw on the same conditioning problem.
        # It uses its own RNG substream, so the "c" draw above is
        # untouched and val/latent_mse stays comparable with every run
        # recorded before this metric existed.
        #
        g_disp = _generator(
          _substream(base, "c2"),
          z0.device,
        )

        draw2 = _conditional_sample(
          model,
          z0,
          rays,
          cond_mask,
          num_steps=K_ROUTINE,
          rng=g_disp,
        )

        if not torch.isfinite(
          draw2
        ).all():
          raise FloatingPointError(
            f"non-finite second validation draw "
            f"in scene {scene_id!r}"
          )

        draw2_target = draw2[target_mask]

        pair_sse += (
          pred_target - draw2_target
        ).square().sum(
          dtype=torch.float64,
        )

        draw2_sse += (
          draw2_target - ref_target
        ).square().sum(
          dtype=torch.float64,
        )

    if (
      latent_count.item() == 0
      or pixel_count.item() == 0
    ):
      raise ValueError(
        "validation set contains no target elements"
      )

    latent_mse = (
      latent_sse / latent_count
    )

    mse = (
      pixel_sse / pixel_count
    )

    # pooled_psnr expects:
    #   squared-error SUM, number of elements
    #
    # Do NOT pass mse here. It would divide by count
    # a second time and inflate PSNR by 10*log10(count).
    psnr = pooled_psnr(
      float(pixel_sse.item()),
      int(pixel_count.item()),
    )

    # Symmetric sample-to-truth distance. Using both draws keeps the
    # ratio from being biased by whichever draw was scored first, and
    # costs nothing: draw2_sse is already accumulated.
    d_truth = (
      latent_sse + draw2_sse
    ) / (2.0 * latent_count)

    d_sample = pair_sse / latent_count

    # R = 2*V_model / (V_model + V_truth)
    #
    #   0 -> collapsed to the conditional mean
    #   1 -> model spread matches the data's
    #
    # R is NOT capped at 1: it tends to 2 as the targets become
    # deterministic, so R > 1 means over-dispersion.
    #
    # A bit-exact model would make this 0/0 and yield nan, which the
    # gate rejects. That cannot happen on a stochastic task.
    r = d_sample / d_truth

    v_model = d_sample / 2.0
    v_truth = d_truth - v_model

    return {
      "val/latent_mse": float(
        latent_mse.cpu()
      ),
      "val/mse": float(
        mse.cpu()
      ),
      "val/psnr": psnr,
      "val/R": float(
        r.cpu()
      ),
      "val/d_truth": float(
        d_truth.cpu()
      ),
      "val/v_model": float(
        v_model.cpu()
      ),
      "val/v_truth": float(
        v_truth.cpu()
      ),
    }

  finally:
    if was_training:
      model.train()