from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.flow.interpolant import linear_interpolant
from src.training.ema import EMA
from src.models.flow.loss import (
  _masked_flow_matching_loss_per_sample,
  masked_flow_matching_loss,
)
from src.models.flow.conditioning import apply_conditioning
from src.training.view_roles import sample_view_roles


@dataclass
class TrainStepResult:
  loss: float
  grad_norm: float
  clipped: bool
  per_k_loss: dict[int, float]
  skipped: bool
  cond_mask: Tensor


def _set_learning_rate(
  optimizer: torch.optim.Optimizer,
  learning_rate: float,
) -> None:
  for param_group in optimizer.param_groups:
    param_group["lr"] = learning_rate


def _sample_noise_and_time(
  z0: Tensor,
  *,
  generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
  epsilon = torch.randn(
    z0.shape,
    dtype=z0.dtype,
    device=z0.device,
    generator=generator,
  )

  t = torch.rand(
    z0.shape[0],
    dtype=torch.float32,
    device=z0.device,
    generator=generator,
  )

  return epsilon, t


def _compute_per_k_loss(
  per_sample_loss: Tensor,
  cond_mask: Tensor,
) -> dict[int, float]:
  result: dict[int, list[float]] = {}

  for batch_index in range(
    cond_mask.shape[0]
  ):
    k = int(
      cond_mask[batch_index].sum().item()
    )

    result.setdefault(k, []).append(
      float(
        per_sample_loss[
          batch_index
        ].detach().item()
      )
    )

  return {
    k: sum(values) / len(values)
    for k, values in result.items()
  }


def _global_grad_norm(
  parameters: list[nn.Parameter],
) -> Tensor:
  gradients = [
    param.grad
    for param in parameters
    if param.grad is not None
  ]

  if not gradients:
    return torch.zeros(
      (),
      dtype=torch.float32,
    )

  squared_norm = torch.zeros(
    (),
    dtype=torch.float32,
    device=gradients[0].device,
  )

  for grad in gradients:
    squared_norm += torch.sum(
      grad.detach().float() ** 2
    )

  return torch.sqrt(squared_norm)


def _clear_gradients(
  optimizer: torch.optim.Optimizer,
) -> None:
  optimizer.zero_grad(
    set_to_none=True
  )


def _result_for_skip(
  *,
  loss: float,
  grad_norm: float,
  clipped: bool,
  per_k_loss: dict[int, float],
  cond_mask: Tensor,
) -> TrainStepResult:
  return TrainStepResult(
    loss=loss,
    grad_norm=grad_norm,
    clipped=clipped,
    per_k_loss=per_k_loss,
    skipped=True,
    cond_mask=cond_mask.detach().clone(),
  )


def train_step(
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  z0: Tensor,
  rays: Tensor,
  *,
  G_noise: torch.Generator,
  G_role: torch.Generator,
  p_drop: float = 0.1,
  cond_mask: Tensor | None = None,
  ema: EMA | None = None,
  scaler: torch.amp.GradScaler | None = None,
  max_grad_norm: float | None = 1.0,
  learning_rate: float | None = None,
  autocast_dtype: torch.dtype | None = None,
  epsilon: Tensor | None = None,
  t: Tensor | None = None,
) -> TrainStepResult:
  if z0.ndim != 5:
    raise ValueError(
      "z0 must have shape [B, V, C, h, w], "
      f"got shape {tuple(z0.shape)}"
    )

  if rays.ndim != 5:
    raise ValueError(
      "rays must have shape [B, V, 6, h, w], "
      f"got shape {tuple(rays.shape)}"
    )

  B, V, C, h, w = z0.shape

  if rays.shape != (B, V, 6, h, w):
    raise ValueError(
      "rays must have shape "
      f"({B}, {V}, 6, {h}, {w}), "
      f"got {tuple(rays.shape)}"
    )

  # ------------------------------------------------------------------
  # View-role sampling.
  # A supplied mask must not consume G_role.
  # ------------------------------------------------------------------
  if cond_mask is None:
    cond_mask = sample_view_roles(
      B=B,
      V=V,
      p_drop=p_drop,
      rng=G_role,
      device=z0.device,
    )
  else:
    if cond_mask.shape != (B, V):
      raise ValueError(
        f"cond_mask must have shape ({B}, {V}), "
        f"got {tuple(cond_mask.shape)}"
      )

    cond_mask = cond_mask.bool()

    if cond_mask.device != z0.device:
      cond_mask = cond_mask.to(
        device=z0.device
      )

  target_mask = ~cond_mask

  if torch.any(
    target_mask.sum(dim=1) == 0
  ):
    bad = torch.where(
      target_mask.sum(dim=1) == 0
    )[0].tolist()

    raise ValueError(
      "Every sample must have at least one target "
      f"view; no target views for samples {bad}"
    )

  # ------------------------------------------------------------------
  # Learning rate belongs to the successful-step scheduler.
  # The caller supplies the LR for this attempted step.
  # ------------------------------------------------------------------
  if learning_rate is not None:
    _set_learning_rate(
      optimizer,
      learning_rate,
    )

  _clear_gradients(optimizer)

  # ------------------------------------------------------------------
  # Noise/time.
  # Explicit epsilon/t are primarily for deterministic parity tests.
  # They do not consume G_noise.
  # ------------------------------------------------------------------
  if epsilon is None or t is None:
    sampled_epsilon, sampled_t = (
      _sample_noise_and_time(
        z0,
        generator=G_noise,
      )
    )

    if epsilon is None:
      epsilon = sampled_epsilon

    if t is None:
      t = sampled_t

  if epsilon.shape != z0.shape:
    raise ValueError(
      f"epsilon must have shape {tuple(z0.shape)}, "
      f"got {tuple(epsilon.shape)}"
    )

  if t.shape != (B,):
    raise ValueError(
      f"t must have shape ({B},), "
      f"got {tuple(t.shape)}"
    )

  # ------------------------------------------------------------------
  # Rectified-flow interpolation.
  # ------------------------------------------------------------------
  z_t = linear_interpolant(
    z0,
    epsilon,
    t,
  )

  # ------------------------------------------------------------------
  # Use the verified P2 conditioning implementation.
  # ------------------------------------------------------------------
  z_t = apply_conditioning(
    z_t,
    z0,
    cond_mask,
  )

  device_type = z0.device.type

  autocast_enabled = (
    autocast_dtype is not None
    and device_type in ("cuda", "cpu")
  )

  with torch.autocast(
    device_type=device_type,
    dtype=autocast_dtype,
    enabled=autocast_enabled,
  ):
    v_pred = model(
      z_t,
      rays,
      t,
      cond_mask,
    )

    target = epsilon - z0

    loss = masked_flow_matching_loss(
      v_pred,
      target,
      target_mask,
    )

  # Compute per-sample diagnostics through the same
  # underlying loss implementation.
  per_sample_loss = (
    _masked_flow_matching_loss_per_sample(
      v_pred,
      target,
      target_mask,
    )
  )

  per_k_loss = _compute_per_k_loss(
    per_sample_loss,
    cond_mask,
  )

  # ------------------------------------------------------------------
  # Non-finite loss.
  #
  # With a scaler, backward/unscale must occur before scaler.update().
  # We deliberately do NOT call optimizer.step().
  # ------------------------------------------------------------------
  loss_finite = bool(
    torch.isfinite(loss.detach()).item()
  )

  if not loss_finite:
    if scaler is not None:
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)

      # scaler.update() is now legal because the current
      # scaled loss has participated in the scaler protocol.
      scaler.update()

    _clear_gradients(optimizer)

    return _result_for_skip(
      loss=float(loss.detach().item()),
      grad_norm=float("nan"),
      clipped=False,
      per_k_loss=per_k_loss,
      cond_mask=cond_mask,
    )

  # ------------------------------------------------------------------
  # Backward.
  # ------------------------------------------------------------------
  if scaler is not None:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
  else:
    loss.backward()

  parameters = [
    param
    for param in model.parameters()
    if param.requires_grad
  ]

  grad_norm_tensor = _global_grad_norm(
    parameters
  )

  # ------------------------------------------------------------------
  # Non-finite gradients.
  # ------------------------------------------------------------------
  if not bool(
    torch.isfinite(
      grad_norm_tensor
    ).item()
  ):
    if scaler is not None:
      scaler.update()

    _clear_gradients(optimizer)

    return _result_for_skip(
      loss=float(loss.detach().item()),
      grad_norm=float("nan"),
      clipped=False,
      per_k_loss=per_k_loss,
      cond_mask=cond_mask,
    )

  # ------------------------------------------------------------------
  # Global L2 clipping.
  #
  # grad_norm_tensor is deliberately captured BEFORE clipping.
  # ------------------------------------------------------------------
  clipped = False

  if max_grad_norm is not None:
    if max_grad_norm <= 0.0:
      raise ValueError(
        "max_grad_norm must be > 0 or None, "
        f"got {max_grad_norm}"
      )

    if (
      grad_norm_tensor.item()
      > max_grad_norm
    ):
      torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_grad_norm,
      )
      clipped = True

  # ------------------------------------------------------------------
  # Optimizer step.
  # ------------------------------------------------------------------
  if scaler is not None:
    scaler.step(optimizer)
    scaler.update()
  else:
    optimizer.step()

  # ------------------------------------------------------------------
  # EMA advances ONLY after a successful optimizer step.
  # ------------------------------------------------------------------
  if ema is not None:
    ema.update(model)

  return TrainStepResult(
    loss=float(loss.detach().item()),
    grad_norm=float(
      grad_norm_tensor.detach().item()
    ),
    clipped=clipped,
    per_k_loss=per_k_loss,
    skipped=False,
    cond_mask=cond_mask.detach().clone(),
  )