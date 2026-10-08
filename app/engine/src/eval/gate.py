"""The validation gate.

Derivation 36 gave a one-sided gate: reconstruction error must fall far below
the noise baseline. That is necessary but not sufficient once the task is
stochastic, because the cheapest way to lower reconstruction error against a
single sample of a distribution is to stop sampling and predict the mean.

Measured on the v2 checkpoint: averaging the model's own four draws scores
0.2274 against 0.3353 for a single draw -- 32% "better" by a one-sided gate,
for a strictly worse generative model. So the gate is two-sided: the model
must be accurate AND it must still be sampling.

This module is pure: it takes the metrics dict that validate_routine returns
and decides. No torch, no I/O, so it is cheap to test exhaustively.
"""

from __future__ import annotations

from typing import NamedTuple

# Score of a model that predicts pure noise. Derivation 36.
NOISE_BASELINE = 2.0

# Accuracy: how far below the noise baseline is "far below".
# The v2 run reached 6.0x; 2.0x is a floor, not a target.
MIN_NOISE_MARGIN = 2.0

# Dispersion band on R = d_sample_sample / d_sample_truth.
#
#   R = 2*V_model / (V_model + V_truth)
#
# so R = 1 means the model's spread matches the data's, R = 0 is total
# collapse, and R -> 2 as the targets become deterministic. Solving the
# identity, the band below is exactly:
#
#   R_MIN = 0.70  <->  V_model >= 0.54 * V_truth   (not collapsed)
#   R_MAX = 1.30  <->  V_model <= 1.86 * V_truth   (not noise-pumping)
#
# v2 measured R = 0.858 pooled, per-scene range 0.785 - 0.944, so it clears
# R_MIN with room on every scene. Widen or tighten deliberately: this is a
# judgement about how much under-dispersion is acceptable, not a constant
# derivable from the data.
R_MIN = 0.70
R_MAX = 1.30


class GateResult(NamedTuple):
  passed: bool
  reasons: tuple[str, ...]   # empty iff passed

  def __bool__(self) -> bool:
    return self.passed


def gate(
  metrics: dict,
  *,
  noise_baseline: float = NOISE_BASELINE,
  min_noise_margin: float = MIN_NOISE_MARGIN,
  r_min: float = R_MIN,
  r_max: float = R_MAX,
) -> GateResult:
  """Two-sided validation gate: accurate AND still sampling.

  Raises KeyError if the dispersion metrics are missing, rather than
  silently degrading to the one-sided gate this module exists to replace.
  """
  if min_noise_margin <= 0:
    raise ValueError("min_noise_margin must be positive")
  if not r_min < r_max:
    raise ValueError(f"empty dispersion band: [{r_min}, {r_max}]")

  for key in ("val/latent_mse", "val/R"):
    if key not in metrics:
      raise KeyError(
        f"gate() requires {key!r}; validate_routine must be run with "
        "dispersion estimation enabled"
      )

  latent_mse = float(metrics["val/latent_mse"])
  r = float(metrics["val/R"])
  reasons: list[str] = []

  # NaN fails every comparison, so test the predicate that must HOLD and
  # invert it. `not (x < y)` is True for NaN; `x >= y` would also be, but
  # spelling it this way keeps the intent readable.
  ceiling = noise_baseline / min_noise_margin
  if not latent_mse < ceiling:
    reasons.append(
      f"accuracy: val/latent_mse {latent_mse:.4f} is not below "
      f"{ceiling:.4f} (= {noise_baseline} / {min_noise_margin})"
    )

  if not r_min <= r <= r_max:
    if r < r_min:
      how = "collapsing toward the conditional mean"
    elif r > r_max:
      how = "over-dispersed; adding noise the data does not have"
    else:
      how = "not a number"
    reasons.append(
      f"dispersion: val/R {r:.4f} outside [{r_min}, {r_max}] -- {how}"
    )

  return GateResult(not reasons, tuple(reasons))


def describe(metrics: dict) -> str:
  """One line for a training log."""
  result = gate(metrics)
  mark = "PASS" if result.passed else "FAIL"
  return (
    f"gate {mark}  latent_mse {metrics['val/latent_mse']:.4f}  "
    f"R {metrics['val/R']:.3f}"
    + ("" if result.passed else "  | " + "; ".join(result.reasons))
  )