from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any


@dataclass
class MetricWindow:
  loss_sum: float = 0.0
  examples: int = 0
  clip_count: int = 0
  update_count: int = 0
  skip_count: int = 0

  def update(
    self,
    *,
    loss_sum: float,
    examples: int,
    clipped: bool = False,
    skipped: bool = False,
  ) -> None:
    if examples < 0:
      raise ValueError("examples must be non-negative")

    self.loss_sum += float(loss_sum)
    self.examples += int(examples)

    if skipped:
      self.skip_count += 1
    else:
      self.update_count += 1

      if clipped:
        self.clip_count += 1

  @property
  def mean_loss(self) -> float | None:
    if self.examples == 0:
      return None

    return self.loss_sum / self.examples

  @property
  def clip_rate(self) -> float:
    if self.update_count == 0:
      return 0.0

    return self.clip_count / self.update_count

  @property
  def skip_rate(self) -> float:
    attempts = self.update_count + self.skip_count

    if attempts == 0:
      return 0.0

    return self.skip_count / attempts

  def reset(self) -> None:
    self.loss_sum = 0.0
    self.examples = 0
    self.clip_count = 0
    self.update_count = 0
    self.skip_count = 0


@dataclass
class OptimizationWindow:
  grad_norm: float | None = None
  grad_scale: float | None = None

  def update(
    self,
    *,
    grad_norm: float | None = None,
    grad_scale: float | None = None,
  ) -> None:
    if grad_norm is not None:
      self.grad_norm = float(grad_norm)

    if grad_scale is not None:
      self.grad_scale = float(grad_scale)

  def reset(self) -> None:
    self.grad_norm = None
    self.grad_scale = None


@dataclass
class TrackerState:
  window: MetricWindow = field(
    default_factory=MetricWindow
  )
  optimization: OptimizationWindow = field(
    default_factory=OptimizationWindow
  )
  interval_start_step: int = 0
  interval_start_examples: int = 0


class Tracker:
  """
  Tracks windowed training metrics and emits log records.

  Trainer owns:
    - global_step
    - epoch
    - batch_cursor
    - examples

  Tracker owns:
    - loss accumulation
    - clip/skip window statistics
    - latest gradient statistics
    - throughput interval baselines
    - metric emission

  Tracker does not own:
    - model state
    - optimizer state
    - scheduler state
    - EMA state
    - checkpoint serialization
    """

  def __init__(
    self,
    run_dir: str | Path,
    *,
    log_every: int = 10,
    tensorboard: bool = True,
    wandb_run: Any | None = None,
  ) -> None:
    if log_every <= 0:
      raise ValueError(
        "log_every must be positive"
      )

    self.run_dir = Path(run_dir)
    self.run_dir.mkdir(
      parents=True,
      exist_ok=True,
    )

    self.log_every = int(log_every)
    self.wandb_run = wandb_run

    self.state = TrackerState()

    self._writer: Any | None = None

    if tensorboard:
      try:
        from torch.utils.tensorboard import (
          SummaryWriter,
        )

        self._writer = SummaryWriter(
          log_dir=str(
            self.run_dir / "tensorboard"
          )
        )
      except ImportError:
        self._writer = None

    self._interval_start_time = monotonic()

    self._last_log_step = 0

  # ------------------------------------------------------------------
  # Updating training metrics
  # ------------------------------------------------------------------

  def update(
    self,
    *,
    loss_sum: float,
    examples: int,
    grad_norm: float | None = None,
    clipped: bool = False,
    skipped: bool = False,
    grad_scale: float | None = None,
  ) -> None:
    """
    Add one training update to the current logging window.

    loss_sum must be the sum of per-example losses represented
    by this update.
    """

    self.state.window.update(
      loss_sum=loss_sum,
      examples=examples,
      clipped=clipped,
      skipped=skipped,
    )

    self.state.optimization.update(
      grad_norm=grad_norm,
      grad_scale=grad_scale,
    )

  # ------------------------------------------------------------------
  # Logging
  # ------------------------------------------------------------------

  def should_log(
    self,
    *,
    step: int,
  ) -> bool:
    if step <= 0:
      return False

    if step == self._last_log_step:
      return False

    return step % self.log_every == 0

  def maybe_log(
    self,
    *,
    step: int,
    epoch: int,
    batch_cursor: int,
    examples: int,
    learning_rate: float | None = None,
    force: bool = False,
    metrics: dict[str, float] | None = None,
  ) -> dict[str, float]:
    """
    Emit a log record when the logging interval is reached.

    Trainer-owned counters are supplied by the caller and are
    never duplicated inside Tracker.
    """

    if not force and not self.should_log(step=step):
      return {}

    record = self._build_record(
      step=step,
      epoch=epoch,
      batch_cursor=batch_cursor,
      examples=examples,
      learning_rate=learning_rate,
      metrics=metrics,
    )

    self._emit(record, step)

    self._last_log_step = step

    # The current window is complete.
    self.state.window.reset()
    self.state.optimization.reset()

    # The next throughput interval begins here.
    self.state.interval_start_step = step
    self.state.interval_start_examples = examples
    self._interval_start_time = monotonic()

    return record

  def _build_record(
    self,
    *,
    step: int,
    epoch: int,
    batch_cursor: int,
    examples: int,
    learning_rate: float | None,
    metrics: dict[str, float] | None,
  ) -> dict[str, float]:
    window = self.state.window

    now = monotonic()
    elapsed = (
      now - self._interval_start_time
    )

    step_delta = (
      step - self.state.interval_start_step
    )

    example_delta = (
      examples
      - self.state.interval_start_examples
    )

    if elapsed > 0.0:
      steps_per_second = (
        step_delta / elapsed
      )

      examples_per_second = (
        example_delta / elapsed
      )
    else:
      steps_per_second = 0.0
      examples_per_second = 0.0

    loss = window.mean_loss

    record: dict[str, float] = {
      "train/step": float(step),
      "train/epoch": float(epoch),
      "train/batch_cursor": float(
        batch_cursor
      ),
      "train/examples": float(examples),

      "train/loss": (
        float(loss)
        if loss is not None
        else float("nan")
      ),

      "train/loss_sum": float(
        window.loss_sum
      ),

      "train/loss_examples": float(
        window.examples
      ),

      "optim/clip_rate": float(
        window.clip_rate
      ),

      "optim/skip_rate": float(
        window.skip_rate
      ),

      "optim/clip_count": float(
        window.clip_count
      ),

      "optim/update_count": float(
        window.update_count
      ),

      "optim/skip_count": float(
        window.skip_count
      ),

      "perf/steps_per_second": float(
        steps_per_second
      ),

      "perf/examples_per_second": float(
        examples_per_second
      ),

      "perf/interval_time": float(
        max(elapsed, 0.0)
      ),
    }

    if learning_rate is not None:
      record["optim/learning_rate"] = float(
        learning_rate
      )

    if (
      self.state.optimization.grad_norm
      is not None
    ):
      record["optim/grad_norm"] = float(
        self.state.optimization.grad_norm
      )

    if (
      self.state.optimization.grad_scale
      is not None
    ):
      record["optim/grad_scale"] = float(
        self.state.optimization.grad_scale
      )

    if metrics is not None:
      for name, value in metrics.items():
        record[str(name)] = float(value)

    return record

  # ------------------------------------------------------------------
  # Metric emission
  # ------------------------------------------------------------------

  def _emit(
    self,
    record: dict[str, float],
    step: int,
  ) -> None:
    self._write_tensorboard(
      record,
      step,
    )

    self._write_wandb(
      record,
      step,
    )

  def _write_tensorboard(
    self,
    record: dict[str, float],
    step: int,
  ) -> None:
    if self._writer is None:
      return

    for name, value in record.items():
      self._writer.add_scalar(
        name,
        value,
        step,
      )

    self._writer.flush()

  def _write_wandb(
    self,
    record: dict[str, float],
    step: int,
  ) -> None:
    if self.wandb_run is None:
      return

    self.wandb_run.log(
      record,
      step=step,
    )

  # ------------------------------------------------------------------
  # Evaluation
  # ------------------------------------------------------------------

  def log_evaluation(
    self,
    *,
    step: int,
    metrics: dict[str, float],
  ) -> dict[str, float]:
    """
    Emit evaluation metrics without modifying
    the training metric window.
    """

    record = {
      f"eval/{name}": float(value)
      for name, value in metrics.items()
    }

    record["eval/step"] = float(step)

    self._write_tensorboard(
      record,
      step,
    )

    self._write_wandb(
      record,
      step,
    )

    return record

  # ------------------------------------------------------------------
  # Checkpoint state
  # ------------------------------------------------------------------

  def state_dict(self) -> dict[str, Any]:
    """
    Return only Tracker-owned durable state.

    Trainer-owned counters are intentionally excluded.
    """

    return {
      "window": asdict(
        self.state.window
      ),
      "optimization": asdict(
        self.state.optimization
      ),
      "interval_start_step": (
        self.state.interval_start_step
      ),
      "interval_start_examples": (
        self.state.interval_start_examples
      ),
    }

  def load_state_dict(
    self,
    state: dict[str, Any],
  ) -> None:
    required = {
      "window",
      "optimization",
      "interval_start_step",
      "interval_start_examples",
    }

    missing = required - set(state)

    if missing:
      raise ValueError(
        "tracker state missing keys: "
        f"{sorted(missing)}"
      )

    window = state["window"]
    optimization = state["optimization"]

    window_required = {
      "loss_sum",
      "examples",
      "clip_count",
      "update_count",
      "skip_count",
    }

    missing_window = (
      window_required - set(window)
    )

    if missing_window:
      raise ValueError(
        "tracker window state missing keys: "
        f"{sorted(missing_window)}"
      )

    optimization_required = {
      "grad_norm",
      "grad_scale",
    }

    missing_optimization = (
      optimization_required
      - set(optimization)
    )

    if missing_optimization:
      raise ValueError(
        "tracker optimization state "
        "missing keys: "
        f"{sorted(missing_optimization)}"
      )

    self.state = TrackerState(
      window=MetricWindow(
        loss_sum=float(
          window["loss_sum"]
        ),
        examples=int(
          window["examples"]
        ),
        clip_count=int(
          window["clip_count"]
        ),
        update_count=int(
          window["update_count"]
        ),
        skip_count=int(
          window["skip_count"]
        ),
      ),
      optimization=OptimizationWindow(
        grad_norm=(
          None
          if optimization["grad_norm"]
          is None
          else float(
            optimization["grad_norm"]
          )
        ),
        grad_scale=(
          None
          if optimization["grad_scale"]
          is None
          else float(
            optimization["grad_scale"]
          )
        ),
      ),
      interval_start_step=int(
        state["interval_start_step"]
      ),
      interval_start_examples=int(
        state["interval_start_examples"]
      ),
    )

    # A monotonic clock cannot be restored across
    # processes. The interval itself therefore starts
    # fresh after resume.
    self._interval_start_time = monotonic()

    self._last_log_step = (
      self.state.interval_start_step
    )

  # ------------------------------------------------------------------
  # TensorBoard resume
  # ------------------------------------------------------------------

  def prepare_tensorboard_resume(
    self,
    *,
    resume_step: int,
  ) -> None:
    """
    Reopen TensorBoard using purge_step semantics.

    Existing events at or after resume_step are purged
    from the resumed view.
    """

    if self._writer is None:
      return

    self._writer.close()

    try:
      from torch.utils.tensorboard import (
        SummaryWriter,
      )

      self._writer = SummaryWriter(
        log_dir=str(
          self.run_dir / "tensorboard"
        ),
        purge_step=resume_step,
      )
    except ImportError:
      self._writer = None

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def flush(self) -> None:
    if self._writer is not None:
      self._writer.flush()

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()
      self._writer = None