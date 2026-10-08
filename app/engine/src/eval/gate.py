from __future__ import annotations

from typing import NamedTuple

NOISE_BASELINE = 2.0
MIN_NOISE_MARGIN = 2.0

R_MIN = 0.7
R_MAX = 1.3

class GateResult(NamedTuple):
  passed: bool
  reasons: tuple[str, ...]

  def __bool__(self) -> bool:
    return self.passed


def gate(
  metrics: dict, 
  *, 
  noise_baseline: float = NOISE_BASELINE, 
  min_noise_margin: float = MIN_NOISE_MARGIN, 
  r_min: float = R_MIN, 
  r_max: float = R_MAX
) -> GateResult:
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
  result = gate(metrics)
  mark = "PASS" if result.passed else "FAIL"
  return (
    f"gate {mark}  latent_mse {metrics['val/latent_mse']:.4f}  "
    f"R {metrics['val/R']:.3f}"
    + ("" if result.passed else "  | " + "; ".join(result.reasons))
  )