"""The collapsed model scores BETTER on the old gate. The new gate rejects it.

This is the regression test for a design bug, not a code bug. Measured on the
v2 checkpoint, averaging the model's own four draws scored 0.2274 against
0.3353 for a single draw -- 32% "better" by reconstruction error alone, for a
strictly worse generative model. Any gate that looks only at val/latent_mse
prefers blur, and `best.pt = argmax PSNR` would have selected the blurriest
checkpoint in the run.

The two fixtures below bracket the failure:

  CollapseToZero  velocity K*x drives the state to 0 in a single Euler step
                  and holds it there, so every draw is identical. R = 0.
  ZeroVelocity    velocity 0 leaves the state at its initial noise, so each
                  draw is an independent sample. R = 1, and the score is the
                  ~2.0 noise baseline of derivation 36.

CollapseToZero wins on val/latent_mse and loses on the gate. That is the
whole point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.eval.gate import gate
from src.eval.validate import (
  K_ROUTINE,
  _conditional_sample,
  _generator,
  _substream,
  _validation_seed,
  validate_routine,
)

VAL_SEED = 4242
NUM_INPUT = 2


class CollapseToZero(nn.Module):
  """v = K*x. One Euler step lands on 0 and v(0) = 0 holds it there.

  The initial noise is annihilated, so every draw is bit-identical:
  a perfect mode collapse, and a very good val/latent_mse.
  """

  def forward(self, x, rays, t, cond_mask):
    return K_ROUTINE * x


class ZeroVelocity(nn.Module):
  """v = 0. The state never moves, so each draw IS its initial noise.

  Maximally dispersed and maximally inaccurate: the noise baseline.
  """

  def forward(self, x, rays, t, cond_mask):
    return torch.zeros_like(x)


class StubVAE:
  """Minimal stand-in: [1,n,C,h,w] -> [1,n,3,h,w]. The pixel path is not
  what this file tests; pooled_mse owns the image-domain clamp anyway."""

  def decode(self, z):
    if z.shape[2] >= 3:
      return z[:, :, :3].clamp(0.0, 1.0)
    return z[:, :, :1].expand(-1, -1, 3, -1, -1).clamp(0.0, 1.0)


def _scenes(n=4, v=4, c=4, h=8, w=8):
  out = []
  for i in range(n):
    g = torch.Generator().manual_seed(1000 + i)
    cond = torch.zeros(v, dtype=torch.bool)
    cond[:NUM_INPUT] = True
    out.append({
      "scene_id": f"gate-scene-{i:03d}",
      "latents": torch.randn(v, c, h, w, generator=g),
      "rays": torch.randn(v, 6, h, w, generator=g),
      "cond_mask": cond,
    })
  return out


def _run(model):
  return validate_routine(
    model,
    _scenes(),
    vae=StubVAE(),
    val_seed=VAL_SEED,
    device=torch.device("cpu"),
    num_input=NUM_INPUT,
  )


def test_collapse_scores_better_on_latent_mse_but_fails_the_gate():
  collapsed = _run(CollapseToZero())
  dispersed = _run(ZeroVelocity())

  # The old, one-sided gate PREFERS the collapsed model.
  assert collapsed["val/latent_mse"] < dispersed["val/latent_mse"], (
    "fixture broken: collapse is supposed to win on reconstruction error "
    f"({collapsed['val/latent_mse']:.4f} vs {dispersed['val/latent_mse']:.4f})"
  )

  # R tells them apart.
  assert collapsed["val/R"] < 0.01, collapsed["val/R"]
  assert dispersed["val/R"] > 0.90, dispersed["val/R"]

  # The two-sided gate rejects the collapsed model for the right reason.
  verdict = gate(collapsed)
  assert not verdict.passed
  assert any("dispersion" in r for r in verdict.reasons), verdict.reasons
  assert not any("accuracy" in r for r in verdict.reasons), (
    "collapse should fail on dispersion alone -- its accuracy is fine, "
    f"which is exactly the trap: {verdict.reasons}"
  )


def test_zero_velocity_sits_at_the_noise_baseline_and_is_fully_dispersed():
  m = _run(ZeroVelocity())
  # Each draw is its own standard normal against unit-variance targets:
  # E[(eps - z0)^2] = 2, and E[(eps_a - eps_b)^2] = 2, so R -> 1.
  assert 1.6 < m["val/latent_mse"] < 2.4, m["val/latent_mse"]
  assert 0.90 < m["val/R"] < 1.10, m["val/R"]


def test_collapse_has_zero_model_variance():
  m = _run(CollapseToZero())
  assert m["val/v_model"] < 1e-12, m["val/v_model"]
  # With no model variance, all the distance to truth is the data's own.
  assert abs(m["val/v_truth"] - m["val/d_truth"]) < 1e-9


def test_dispersion_draw_leaves_latent_mse_bit_identical():
  """The second draw must not perturb the first.

  val/latent_mse has to stay comparable with every run recorded before this
  metric existed, so the dispersion draw uses its own RNG substream. This
  recomputes the single "c" draw by hand and demands an exact match.
  """
  model = ZeroVelocity()
  scenes = _scenes()
  device = torch.device("cpu")

  # Guard the guard. Without this, the test is vacuously true whenever the
  # second draw is missing entirely -- which is exactly the state it is
  # supposed to be protecting against.
  produced = _run(model)
  for key in ("val/R", "val/d_truth", "val/v_model", "val/v_truth"):
    assert key in produced, (
      f"validate_routine did not return {key!r}: the dispersion patch is "
      "not applied, so this test proves nothing"
    )

  sse = torch.zeros((), dtype=torch.float64)
  count = 0
  for scene in scenes:
    z0 = scene["latents"].unsqueeze(0)
    rays = scene["rays"].unsqueeze(0)
    cm = scene["cond_mask"].unsqueeze(0)
    k = int(cm.sum().item())
    base = _validation_seed(VAL_SEED, scene["scene_id"], k)
    pred = _conditional_sample(
      model, z0, rays, cm,
      num_steps=K_ROUTINE,
      rng=_generator(_substream(base, "c"), device),
    )
    t = ~cm
    diff = pred[t] - z0[t]
    sse += diff.square().sum(dtype=torch.float64)
    count += diff.numel()

  expected = float((sse / count).cpu())
  assert produced["val/latent_mse"] == expected, (
    "the dispersion draw changed the 'c' stream; every historical "
    "val/latent_mse is now incomparable"
  )