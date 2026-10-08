from __future__ import annotations

import pytest
import torch

from src.eval.validate import validate_routine
from src.models.dit.model import MultiViewDiT


# Four scenes x 2 target views x 4 channels x 8 x 8 = 2048 target elements.
# v == 0 exactly at init, so val/latent_mse is the sample mean of (eps - z0)^2:
# mean 2.0, SD = sqrt(8 / 2048) = 0.0625. The 0.15 tolerance below is 2.4 sigma.
NUM_SCENES = 4
NUM_VIEWS = 4
NUM_INPUT = 2
LATENT_CHANNELS = 4
LATENT_HW = 8


class StubVAE:
  """Decoder stand-in. Only val/latent_mse is asserted; pixels are not meaningful."""

  def decode(self, latents: torch.Tensor) -> torch.Tensor:
    b, v, _, h, w = latents.shape
    return torch.sigmoid(latents[:, :, :3]).reshape(b, v, 3, h, w)


def _make_model() -> MultiViewDiT:
  # in_channels is the LATENT channel count. The model adds the 6 Plucker
  # channels internally (C+6 in, C out), so do not pre-add them here.
  # There is no out_channels parameter.
  return MultiViewDiT(
    in_channels=LATENT_CHANNELS,
    dim=32,
    num_heads=2,
    cond_dim=32,
    num_layers=2,
    patch_size=2,
  )


def _make_scene(
  *,
  scene_id: str,
  generator: torch.Generator,
  num_input: int = NUM_INPUT,
) -> dict:
  latents = torch.randn(
    NUM_VIEWS, LATENT_CHANNELS, LATENT_HW, LATENT_HW, generator=generator
  )
  rays = torch.randn(
    NUM_VIEWS, 6, LATENT_HW, LATENT_HW, generator=generator
  )

  cond_mask = torch.zeros(NUM_VIEWS, dtype=torch.bool)
  cond_mask[:num_input] = True

  return {
    "latents": latents,
    "rays": rays,
    "cond_mask": cond_mask,
    "scene_id": scene_id,
  }


def _make_val_set(seed: int = 1234, num_input: int = NUM_INPUT) -> list[dict]:
  generator = torch.Generator().manual_seed(seed)
  return [
    _make_scene(
      scene_id=f"scene_{i:03d}",
      generator=generator,
      num_input=num_input,
    )
    for i in range(NUM_SCENES)
  ]


def test_validate_routine_zero_init_latent_mse() -> None:
  """An untrained model leaves target views at their initial noise."""
  torch.manual_seed(0)
  device = torch.device("cpu")

  model = _make_model().to(device)
  assert model.training, "model should start in train mode for the restore check"

  metrics = validate_routine(
    model,
    _make_val_set(),  # type: ignore
    vae=StubVAE(),
    val_seed=2026,
    device=device,
    num_input=NUM_INPUT,
  )

  assert set(metrics) == {
    "val/latent_mse",
    "val/mse",
    "val/psnr",
    "val/R",
    "val/d_truth",
    "val/v_model",
    "val/v_truth",
  }
  assert all(isinstance(value, float) for value in metrics.values())

  # E[(eps - z0)^2] = Var(eps) + Var(z0) = 2, at 2.4 sigma for this sample size.
  assert abs(metrics["val/latent_mse"] - 2.0) < 0.15

  # The finally block must put the model back the way it found it.
  assert model.training


def test_validate_routine_restores_eval_mode() -> None:
  """A model handed over in eval mode must stay in eval mode."""
  device = torch.device("cpu")
  model = _make_model().to(device)
  model.eval()

  validate_routine(
    model,
    _make_val_set(),  # type: ignore
    vae=StubVAE(),
    val_seed=2026,
    device=device,
    num_input=NUM_INPUT,
  )

  assert not model.training


def test_validate_routine_is_deterministic() -> None:
  """Derivation 75: the same checkpoint evaluated twice sees identical noise."""
  torch.manual_seed(0)
  device = torch.device("cpu")
  model = _make_model().to(device)

  kwargs = dict(
    vae=StubVAE(),
    val_seed=2026,
    device=device,
    num_input=NUM_INPUT,
  )

  first = validate_routine(model, _make_val_set(), **kwargs)  # type: ignore

  # Burn global RNG between runs: validation must not depend on it.
  torch.rand(1024)

  second = validate_routine(model, _make_val_set(), **kwargs) # type: ignore

  assert first == second


def test_validate_routine_rejects_wrong_num_input() -> None:
  """Derivation 75 fixes routine validation at k = num_input."""
  device = torch.device("cpu")
  model = _make_model().to(device)

  with pytest.raises(ValueError, match=r"expected num_input=1"):
    validate_routine(
      model,
      _make_val_set(),  # type: ignore
      vae=StubVAE(),
      val_seed=2026,
      device=device,
      num_input=1,
    ) 