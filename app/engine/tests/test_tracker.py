"""Regression tests for src.training.tracker (P3.3b, derivations 53/54/70).

Each test pins a behaviour that was wrong in an earlier revision:
  - windows must reset on log (means and rates were cumulative)
  - throughput must use interval deltas (was lifetime / post-resume time)
  - the backend handle must be None when tensorboard is off (was 0)
  - tracker state must not duplicate Trainer-owned counters

No test needs tensorboard installed: every Tracker is built with
tensorboard=False, which is itself the K0 regression.
"""
from __future__ import annotations

import copy
import math
import time

import pytest

from src.training.tracker import Tracker

# The derivation-70 worked example: five steps, unequal example counts.
#   sum 184 / count 160 = 1.15   (mean-of-means would be 1.4)
I4E_STEPS = [
  (64.0, 32),
  (32.0, 32),
  (32.0, 64),
  (32.0, 16),
  (24.0, 16),
]


def make_tracker(tmp_path, **kwargs) -> Tracker:
  kwargs.setdefault("log_every", 2)
  kwargs.setdefault("tensorboard", False)
  return Tracker(tmp_path, **kwargs)


def run_steps(tracker: Tracker, steps, *, force_last: bool = True):
  """Feed (loss_sum, examples) pairs; return the emitted log records."""
  records = []
  examples = 0
  for step, (loss_sum, n) in enumerate(steps, start=1):
    tracker.update(loss_sum=loss_sum, examples=n)
    examples += n
    record = tracker.maybe_log(
      step=step, epoch=0, batch_cursor=step, examples=examples
    )
    if record:
      records.append(record)
  if force_last:
    record = tracker.maybe_log(
      step=len(steps), epoch=0, batch_cursor=len(steps),
      examples=examples, force=True,
    )
    if record:
      records.append(record)
  return records


# ----------------------------------------------------------------------
# K0 — the writer handle is None, not 0, when tensorboard is disabled
# ----------------------------------------------------------------------
def test_tracker_runs_without_tensorboard(tmp_path):
  tracker = make_tracker(tmp_path)
  tracker.update(loss_sum=4.0, examples=4)
  record = tracker.maybe_log(step=2, epoch=0, batch_cursor=8, examples=8)
  assert record["train/loss"] == pytest.approx(1.0)


def test_tracker_rejects_non_positive_log_every(tmp_path):
  for bad in (0, -1):
    with pytest.raises(ValueError):
      Tracker(tmp_path, log_every=bad, tensorboard=False)


# ----------------------------------------------------------------------
# K1 — each log covers its own window, and the windows tile the run
# ----------------------------------------------------------------------
def test_tracker_window_means_are_per_window(tmp_path):
  records = run_steps(make_tracker(tmp_path), I4E_STEPS)
  means = [r["train/loss"] for r in records]

  assert len(records) == 3
  assert means[0] == pytest.approx(96 / 64)   # steps 1-2
  assert means[1] == pytest.approx(64 / 80)   # steps 3-4  (cumulative: 1.111)
  assert means[2] == pytest.approx(24 / 16)   # step 5


def test_tracker_windows_partition_the_run(tmp_path):
  """Derivation 70: sum/count over all windows is the example-weighted mean."""
  records = run_steps(make_tracker(tmp_path), I4E_STEPS)
  total_sum = sum(r["train/loss_sum"] for r in records)
  total_examples = sum(r["train/loss_examples"] for r in records)

  assert total_sum == pytest.approx(184.0)
  assert total_examples == pytest.approx(160.0)
  assert total_sum / total_examples == pytest.approx(1.15)
  # ... and NOT the mean of the per-window means
  naive = sum(r["train/loss"] for r in records) / len(records)
  assert naive != pytest.approx(1.15)


def test_tracker_window_resets_counts(tmp_path):
  records = run_steps(make_tracker(tmp_path), I4E_STEPS)
  assert [r["train/loss_examples"] for r in records] == [64.0, 80.0, 16.0]


# ----------------------------------------------------------------------
# K2 — clip / skip rates are window rates, and a skip carries no loss
# ----------------------------------------------------------------------
def test_tracker_clip_rate_is_per_window(tmp_path):
  tracker = make_tracker(tmp_path)
  for _ in range(2):
    tracker.update(loss_sum=1.0, examples=1, clipped=True, grad_norm=2.0)
  first = tracker.maybe_log(step=2, epoch=0, batch_cursor=2, examples=2)

  for _ in range(2):
    tracker.update(loss_sum=1.0, examples=1, clipped=False, grad_norm=2.0)
  second = tracker.maybe_log(step=4, epoch=0, batch_cursor=4, examples=4)

  assert first["optim/clip_rate"] == pytest.approx(1.0)
  assert second["optim/clip_rate"] == pytest.approx(0.0)  # lifetime would be 0.5
  assert second["optim/clip_count"] == 0.0


def test_tracker_skip_rate_and_nan_loss(tmp_path):
  tracker = make_tracker(tmp_path)
  # a skipped attempt contributes no loss and no examples
  tracker.update(loss_sum=0.0, examples=0, skipped=True)
  record = tracker.maybe_log(
    step=1, epoch=0, batch_cursor=1, examples=0, force=True
  )
  assert record["optim/skip_rate"] == pytest.approx(1.0)
  assert record["optim/update_count"] == 0.0
  assert math.isnan(record["train/loss"])


def test_tracker_grad_scale_is_absent_when_not_supplied(tmp_path):
  tracker = make_tracker(tmp_path)
  tracker.update(loss_sum=1.0, examples=1, grad_norm=0.5)
  record = tracker.maybe_log(step=2, epoch=0, batch_cursor=2, examples=1, force=True)
  assert "optim/grad_scale" not in record          # bf16: never fabricated
  assert record["optim/grad_norm"] == pytest.approx(0.5)

  tracker.update(loss_sum=1.0, examples=1, grad_scale=256.0)
  record = tracker.maybe_log(step=4, epoch=0, batch_cursor=4, examples=2, force=True)
  assert record["optim/grad_scale"] == pytest.approx(256.0)


# ----------------------------------------------------------------------
# K3 / K4 — throughput uses interval deltas and survives a resume
# ----------------------------------------------------------------------
def test_tracker_throughput_uses_interval_deltas_after_resume(tmp_path):
  tracker = make_tracker(tmp_path / "a", log_every=10)
  examples = 0
  for step in range(1, 101):
    examples += 32
    tracker.update(loss_sum=32.0, examples=32)
    tracker.maybe_log(step=step, epoch=0, batch_cursor=step, examples=examples)

  state = copy.deepcopy(tracker.state_dict())

  resumed = make_tracker(tmp_path / "b", log_every=10)
  resumed.load_state_dict(state)
  time.sleep(0.05)
  for _ in range(10):
    examples += 32
    resumed.update(loss_sum=32.0, examples=32)
  record = resumed.maybe_log(
    step=110, epoch=0, batch_cursor=110, examples=examples, force=True
  )

  # The rate must describe THIS interval: 10 steps / 320 examples. Lifetime
  # counters (110 steps / 3520 examples) over the same elapsed time would be 11x.
  interval = record["perf/interval_time"]
  assert interval > 0.0
  assert record["perf/steps_per_second"] * interval == pytest.approx(10.0, rel=1e-6)
  assert record["perf/examples_per_second"] * interval == pytest.approx(320.0, rel=1e-6)


def test_tracker_zero_elapsed_does_not_divide(tmp_path):
  tracker = make_tracker(tmp_path, log_every=1)
  tracker.update(loss_sum=1.0, examples=1)
  tracker._interval_start_time += 10 ** 6        # force elapsed <= 0
  record = tracker.maybe_log(
    step=1, epoch=0, batch_cursor=1, examples=1, force=True
  )
  assert record["perf/steps_per_second"] == 0.0
  assert record["perf/examples_per_second"] == 0.0
  assert record["perf/interval_time"] == 0.0


# ----------------------------------------------------------------------
# K5 — state is tracker-owned only, and round-trips exactly
# ----------------------------------------------------------------------
def test_tracker_state_excludes_trainer_counters(tmp_path):
  tracker = make_tracker(tmp_path)
  tracker.update(loss_sum=3.0, examples=2, clipped=True, grad_norm=1.5, grad_scale=256.0)
  state = tracker.state_dict()

  assert set(state) == {
    "window", "optimization", "interval_start_step", "interval_start_examples",
  }
  for owned_by_trainer in ("global_step", "epoch", "batch_cursor", "examples"):
    assert owned_by_trainer not in state


def test_tracker_state_roundtrip_is_exact(tmp_path):
  tracker = make_tracker(tmp_path / "a")
  tracker.update(loss_sum=3.0, examples=2, clipped=True, grad_norm=1.5, grad_scale=256.0)
  state = copy.deepcopy(tracker.state_dict())

  resumed = make_tracker(tmp_path / "b")
  resumed.load_state_dict(state)
  assert resumed.state_dict() == state


def test_tracker_window_survives_a_mid_window_resume(tmp_path):
  """A crash mid-window must not lose the carried sums/counts."""
  tracker = make_tracker(tmp_path / "a", log_every=100)
  examples = 0
  for loss_sum, n in I4E_STEPS[:3]:
    tracker.update(loss_sum=loss_sum, examples=n)
    examples += n
  state = copy.deepcopy(tracker.state_dict())

  resumed = make_tracker(tmp_path / "b", log_every=100)
  resumed.load_state_dict(state)
  for loss_sum, n in I4E_STEPS[3:]:
    resumed.update(loss_sum=loss_sum, examples=n)
    examples += n
  record = resumed.maybe_log(
    step=5, epoch=0, batch_cursor=5, examples=examples, force=True
  )

  assert record["train/loss_sum"] == pytest.approx(184.0)
  assert record["train/loss_examples"] == pytest.approx(160.0)
  assert record["train/loss"] == pytest.approx(1.15)


@pytest.mark.parametrize(
  "state",
  [
    {},
    {"window": {}, "optimization": {}, "interval_start_step": 0},
    {
      "window": {"loss_sum": 0.0},
      "optimization": {"grad_norm": None, "grad_scale": None},
      "interval_start_step": 0,
      "interval_start_examples": 0,
    },
  ],
)
def test_tracker_rejects_incomplete_state(tmp_path, state):
  with pytest.raises(ValueError):
    make_tracker(tmp_path).load_state_dict(state)


# ----------------------------------------------------------------------
# K6 / K7 — evaluation isolation and no duplicate logs
# ----------------------------------------------------------------------
def test_tracker_evaluation_does_not_touch_the_window(tmp_path):
  tracker = make_tracker(tmp_path)
  tracker.update(loss_sum=5.0, examples=5)
  before = copy.deepcopy(tracker.state_dict())

  record = tracker.log_evaluation(step=7, metrics={"psnr": 21.5})

  assert tracker.state_dict() == before
  assert record["eval/psnr"] == pytest.approx(21.5)
  assert record["eval/step"] == 7.0


def test_tracker_does_not_log_the_same_step_twice(tmp_path):
  tracker = make_tracker(tmp_path)
  tracker.update(loss_sum=1.0, examples=1)
  assert tracker.maybe_log(step=2, epoch=0, batch_cursor=2, examples=1)
  assert tracker.maybe_log(step=2, epoch=0, batch_cursor=2, examples=1) == {}


def test_tracker_should_log_honours_the_interval(tmp_path):
  tracker = make_tracker(tmp_path, log_every=3)
  assert not tracker.should_log(step=0)
  assert not tracker.should_log(step=2)
  assert tracker.should_log(step=3)
  assert tracker.should_log(step=6)


# ----------------------------------------------------------------------
# Backend wiring — steps handed to wandb never go backwards in a process
# ----------------------------------------------------------------------
class _FakeRun:
  def __init__(self) -> None:
    self.steps: list[int] = []

  def log(self, record, step=None) -> None:
    self.steps.append(step) # type: ignore


def test_tracker_wandb_steps_are_non_decreasing(tmp_path):
  run = _FakeRun()
  tracker = make_tracker(tmp_path, log_every=1, wandb_run=run)
  examples = 0
  for step in (1, 2, 3):
    examples += 1
    tracker.update(loss_sum=1.0, examples=1)
    tracker.maybe_log(step=step, epoch=0, batch_cursor=step, examples=examples)
  tracker.log_evaluation(step=3, metrics={"psnr": 20.0})

  assert run.steps == sorted(run.steps)
  assert run.steps[-1] == 3
