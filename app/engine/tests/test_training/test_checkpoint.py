from pathlib import Path

import pytest
import torch

from src.training.checkpoint import CheckpointError, CheckpointManager


def make_state(step: int) -> dict:
  return {
    "step": step,
    "value": torch.tensor([float(step)]),
  }


def test_c1_round_trip_and_pointer_is_filename(tmp_path):
  manager = CheckpointManager(tmp_path)

  state = make_state(10)
  path = manager.save(
    step=10,
    state=state,
  )

  assert path == manager.step_path(10)
  assert path.exists()

  pointer = manager.current_path.read_text(
    encoding="utf-8"
  ).strip()

  assert pointer == path.name
  assert not Path(pointer).is_absolute()

  loaded, loaded_path = manager.load_latest()

  assert loaded_path == path
  assert loaded["step"] == 10
  assert torch.equal(
    loaded["value"],
    state["value"],
  )


def test_c2_milestones_get_distinct_names(tmp_path):
  manager = CheckpointManager(tmp_path)

  path_10 = manager.save_milestone(
    step=10,
    state=make_state(10),
  )

  path_20 = manager.save_milestone(
    step=20,
    state=make_state(20),
  )

  assert path_10 != path_20
  assert path_10.name == "milestone_000010.pt"
  assert path_20.name == "milestone_000020.pt"

  assert path_10.exists()
  assert path_20.exists()

  loaded_10 = manager.load(path_10)
  loaded_20 = manager.load(path_20)

  assert loaded_10["step"] == 10
  assert loaded_20["step"] == 20


def test_c3_failed_replace_preserves_old_checkpoint_and_cleans_tmp(
  tmp_path,
  monkeypatch,
):
  manager = CheckpointManager(
    tmp_path,
    replace_retries=0,
  )

  path = manager.save(
    step=10,
    state=make_state(10),
  )

  original = manager.load(path)

  def fail_replace(source, destination):
    raise OSError("simulated replace failure")

  monkeypatch.setattr(
    "os.replace",
    fail_replace,
  )

  with pytest.raises(CheckpointError):
    manager.save(
      step=10,
      state=make_state(999),
    )

  assert path.exists()

  restored = manager.load(path)

  assert restored["step"] == original["step"]
  assert torch.equal(
    restored["value"],
    original["value"],
  )

  assert not path.with_name(
    path.name + ".tmp"
  ).exists()


def test_c4_corrupt_newest_falls_back_to_previous(
  tmp_path,
):
  manager = CheckpointManager(tmp_path)

  old_path = manager.save(
    step=10,
    state=make_state(10),
  )

  newest_path = manager.save(
    step=20,
    state=make_state(20),
  )

  newest_path.write_bytes(
    b"this is a truncated checkpoint"
  )

  loaded, loaded_path = manager.load_latest()

  assert loaded_path == old_path
  assert loaded["step"] == 10
  assert loaded_path != newest_path


def test_c5_stale_pointer_falls_back_to_highest_valid_step(
  tmp_path,
):
  manager = CheckpointManager(tmp_path)

  manager.save(
    step=10,
    state=make_state(10),
  )

  path_20 = manager.save(
    step=20,
    state=make_state(20),
  )

  manager.current_path.write_text(
    "step_999999.pt\n",
    encoding="utf-8",
  )

  loaded, loaded_path = manager.load_latest()

  assert loaded_path == path_20
  assert loaded["step"] == 20


def test_c6_pruning_keeps_last_n_and_protected_checkpoints(
  tmp_path,
):
  manager = CheckpointManager(
    tmp_path,
    keep_last=2,
  )

  manager.save(
    step=10,
    state=make_state(10),
  )

  manager.save(
    step=20,
    state=make_state(20),
  )

  manager.save(
    step=30,
    state=make_state(30),
  )

  manager.save_best(
    step=10,
    state=make_state(10),
  )

  manager.save_milestone(
    step=10,
    state=make_state(10),
  )

  manager.save_emergency(
    state=make_state(10),
  )

  step_paths = sorted(
    manager.checkpoint_dir.glob(
      "step_*.pt"
    )
  )

  assert [
    path.name
    for path in step_paths
  ] == [
    "step_000020.pt",
    "step_000030.pt",
  ]

  assert manager.best_path().exists()
  assert manager.milestone_path(10).exists()

  emergency_paths = list(
    manager.checkpoint_dir.glob(
      "emergency_*.pt"
    )
  )

  assert len(emergency_paths) == 1


def test_c7_emergency_cap_is_three(tmp_path):
  manager = CheckpointManager(
    tmp_path,
    emergency_keep=3,
  )

  for step in range(1, 6):
    manager.save_emergency(
      state=make_state(step),
    )

  emergency_paths = sorted(
    manager.checkpoint_dir.glob(
      "emergency_*.pt"
    ),
    key=manager._emergency_index,
  )

  assert len(emergency_paths) == 3

  assert [
    path.name
    for path in emergency_paths
  ] == [
    "emergency_003.pt",
    "emergency_004.pt",
    "emergency_005.pt",
  ]


def test_c8_load_uses_weights_only(tmp_path, monkeypatch):
  manager = CheckpointManager(tmp_path)

  path = manager.save(
    step=10,
    state=make_state(10),
  )

  original_load = torch.load
  observed = {}

  def wrapped_load(*args, **kwargs):
    observed["weights_only"] = kwargs.get(
      "weights_only"
    )
    return original_load(
      *args,
      **kwargs,
    )

  monkeypatch.setattr(
    torch,
    "load",
    wrapped_load,
  )

  manager.load(path)

  assert observed["weights_only"] is True


class NotATensor:
  """Module-level so torch.save can pickle it (a local class cannot be)."""


def test_c8b_weights_only_refuses_arbitrary_objects(
  tmp_path,
):
  """weights_only=True must actually refuse a pickled instance.

  c8 asserts the flag is passed; this asserts the flag does something.
  """
  manager = CheckpointManager(tmp_path)

  manager.save(
    step=10,
    state={"object": NotATensor()},
  )

  with pytest.raises(CheckpointError):
    manager.load_latest()


def test_c4b_every_candidate_corrupt_raises(
  tmp_path,
):
  """The scan must not loop forever or return junk when nothing loads."""
  manager = CheckpointManager(tmp_path)

  for step in (10, 20):
    path = manager.save(
      step=step,
      state=make_state(step),
    )

    path.write_bytes(
      b"truncated"
    )

  with pytest.raises(
    CheckpointError,
    match="no valid step checkpoint found",
  ):
    manager.load_latest()


def test_c5b_pointer_outside_checkpoint_dir_is_ignored(
  tmp_path,
):
  """A pointer naming anything but a step file is not followed."""
  manager = CheckpointManager(tmp_path)

  manager.save(
    step=10,
    state=make_state(10),
  )

  path_20 = manager.save(
    step=20,
    state=make_state(20),
  )

  for pointer in (
    "../../escape.pt",
    "best.pt",
    "",
    "not_a_checkpoint",
  ):
    manager.current_path.write_text(
      f"{pointer}\n",
      encoding="utf-8",
    )

    loaded, loaded_path = manager.load_latest()

    assert loaded_path == path_20
    assert loaded["step"] == 20


def test_c9_empty_directory_raises(tmp_path):
  manager = CheckpointManager(tmp_path)

  with pytest.raises(
    CheckpointError,
    match="no valid step checkpoint found",
  ):
    manager.load_latest()