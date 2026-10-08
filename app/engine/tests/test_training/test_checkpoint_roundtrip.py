"""A real Trainer state_dict must survive save -> load_latest.

The 12 existing checkpoint tests round-trip a synthetic
{"step": int, "value": tensor}, which is why they stayed green while
torch.__version__ (a TorchVersion, not a str) made every real checkpoint
unloadable under weights_only=True -- breaking resume, silently.
"""

from __future__ import annotations

import torch

from src.models.dit.model import MultiViewDiT
from src.training.state.checkpoint import CheckpointManager
from src.training.train import Trainer, TrainingConfig, numerical_flags

LATENT_C, HW, NUM_VIEWS = 4, 8, 4


def _scene(seed: int) -> dict:
  g = torch.Generator().manual_seed(seed)
  return {
    "latents": torch.randn(NUM_VIEWS, LATENT_C, HW, HW, generator=g),
    "rays": torch.randn(NUM_VIEWS, 6, HW, HW, generator=g),
  }


def _trainer() -> Trainer:
  model = MultiViewDiT(
    in_channels=LATENT_C, dim=32, num_heads=2,
    cond_dim=32, num_layers=2, patch_size=2,
  )
  cfg = TrainingConfig(total_steps=2, batch_size=2, cache_hash="deadbeef")
  return Trainer(
    cfg, model, [_scene(i) for i in range(4)], device=torch.device("cpu")
  )


def test_numerical_flags_are_plain_primitives() -> None:
  """weights_only=True rejects any non-allowlisted class.

  Exact type, not isinstance: TorchVersion subclasses str, so an
  isinstance check passes while torch.load still refuses the file.
  """
  allowed = (str, bool, int, float, type(None))

  for key, value in numerical_flags().items():
    assert type(value) in allowed, (
      f"numerical_flags()[{key!r}] is {type(value).__name__}, "
      f"not a plain primitive -- weights_only=True will reject it"
    )


def test_real_trainer_state_round_trips(tmp_path) -> None:
  trainer = _trainer()
  manager = CheckpointManager(tmp_path)

  manager.save(step=0, state=trainer.state_dict())

  # This is the call that was failing: weights_only=True on a real payload.
  loaded, path = manager.load_latest()

  assert path.name == "step_000000.pt"
  assert loaded["counters"]["optimizer_step"] == 0
  assert loaded["cache_hash"] == "deadbeef"
  assert "model" in loaded and "optimizer" in loaded and "generators" in loaded


def test_real_trainer_state_reloads_into_a_trainer(tmp_path) -> None:
  """Round-tripping the bytes isn't enough -- resume must accept them."""
  saved = _trainer()
  manager = CheckpointManager(tmp_path)
  manager.save(step=0, state=saved.state_dict())

  loaded, _ = manager.load_latest()

  fresh = _trainer()
  fresh.load_state_dict(loaded)

  assert fresh.optimizer_step == saved.optimizer_step