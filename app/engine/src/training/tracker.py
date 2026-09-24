from __future__ import annotations

# Native Import(s)
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from time import monotonic

@dataclass
class MetricWindow:
  loss_sum: float = 0.0
  examples: int = 0

  def update(self, loss_sum: float, examples: int) -> None:
    if examples < 0:
      raise ValueError(f"Must have a non-negative number of examples, got {examples}")

    self.loss_sum += float(loss_sum)
    self.examples += int(examples)

  @property
  def mean(self) -> float | None:
    if self.examples == 0:
      return None

    return self.loss_sum / self.examples

  def reset(self) -> None:
    self.loss_sum = 0.0
    self.examples = 0


@dataclass
class OptimizationStats:
  grad_norm: float | None = None
  clip_count: int = 0
  update_count: int = 0
  skip_count: int = 0
  grad_scale: float | None = None

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

@dataclass
class PerformanceStats:
  elapsed_time: float = 0.0
  examples_per_second: float = 0.0
  steps_per_second: float = 0.0

@dataclass
class TrackerState:
  global_step: int = 0
  epoch: int = 0
  batch_cursor: int = 0
  examples: int = 0

  window: MetricWindow = field(default_factory=MetricWindow)
  optimization: OptimizationStats = field(default_factory=OptimizationStats)
  performance: PerformanceStats = field(default_factory=PerformanceStats)


class Tracker:
  def __init__(
    self, 
    run_dir: str | Path, 
    *, 
    log_every: int = 10, 
    tensorboard: bool = True,
    wandb_run: Any | None = None
  ) -> None:
    if log_every <= 0:
      raise ValueError(f"log_every must be non-negative, got {log_every}")

    self.run_dir = Path(run_dir)
    self.run_dir.mkdir(parents=True, exist_ok=True)

    self.log_every = log_every
    self.wandb_run = wandb_run

    self.state = TrackerState()

    self._start_time = monotonic()
    self._last_log_time = self._start_time
    self._last_log_step = 0

    self._writer = 0

    if tensorboard:
      try:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(
          log_dir=str(self.run_dir / "tensorboard"),
        )
      except ImportError:
        self._writer = None

  @property
  def global_step(self) -> int:
    return self.state.global_step

  @property
  def epoch(self) -> int:
    return self.state.epoch

  @property
  def batch_cursor(self) -> int:
    return self.state.batch_cursor

  def set_position(
    self,
    *,
    global_step: int | None = None,
    epoch: int | None = None,
    batch_cursor: int | None = None
  ) -> None:
    if global_step is not None:
      if global_step < 0:
        raise ValueError(f"Global Step must be non-negative, got {global_step}")
      self.state.global_step = int(global_step)

    if epoch is not None:
      if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
      self.state.epoch = int(epoch)

    if batch_cursor is not None:
      if batch_cursor < 0:
        raise ValueError(f"Batch Cursor must be non-negative, got {batch_cursor}")
      self.state.batch_cursor = int(batch_cursor)

  def advance_step(
    self,
    *,
    epoch: int | None = None,
    batch_cursor: int | None = None
  ) -> None:
    self.state.global_step += 1

    if epoch is not None:
      self.state.epoch = int(epoch)

    if batch_cursor is not None:
      self.state.batch_cursor = int(batch_cursor)

  def add_examples(self, count: int) -> None:
    if count < 0:
      raise ValueError(f"count must be non-negative, got {count}")

    self.state.examples += int(count)

  def update_loss(self, *, loss_sum: float, examples: int) -> None:
    self.state.window.update(
      loss_sum=loss_sum,
      examples=examples
    )

  def record_optimization(
    self, 
    *, 
    grad_norm: float | None = None, 
    clipped: bool = False, 
    skipped: bool = False,
    grad_scale: float | None = None
  ) -> None:
    stats = self.state.optimization

    if grad_norm is not None:
      stats.grad_norm = float(grad_norm)

    if grad_scale is not None:
      stats.grad_scale = float(grad_scale)

    if skipped:
      stats.skip_count += 1
    else:
      stats.update_count += 1

      if clipped:
        stats.clip_count += 1

  def update_performance(self) -> None:
    now = monotonic()

    self.state.performance.elapsed_time = (now - self._start_time)

    elapsed = self.state.performance.elapsed_time

    if elapsed > 0:
      self.state.performance.examples_per_second = (
        self.state.examples / elapsed
      )

    self.state.performance.steps_per_second = (
      self.state.global_step / elapsed
    )

  def should_log(self) -> bool:
    return (
      self.state.global_step > 0
      and self.state.global_step % self.log_every == 0
      and self.state.global_step != self._last_log_step
    )

  def log(
    self,
    *,
    force: bool = False,
    metrics: dict[str, float] | None = None
  ) -> dict[str, float]:
    if not force and not self.should_log():
      return {}

    self.update_performance()

    output: dict[str, float] = {
      "train/step": float(self.state.global_step),
      "train/epoch": float(self.state.epoch),
      "train/batch_cursor": float(self.state.batch_cursor),
      "train/examples": float(self.state.examples),

      "train/loss": (
        float(self.state.window.mean)
        if self.state.window.mean is not None
        else float("nan")
      ),

      "train/loss_sum": float(self.state.window.loss_sum),
      "train/loss_examples": float(self.state.window.examples),

      "optim/grad_norm": (
        float(self.state.optimization.grad_norm)
        if self.state.optimization.grad_norm is not None
        else float("nan")
      ),

      "optim/clip_rate": (
        float(self.state.optimization.clip_rate)
      ),

      "optim/skip_rate": (
        float(self.state.optimization.skip_rate)
      ),

      "optim/update_count": (
        float(self.state.optimization.update_count)
      ),

      "optim/skip_count": (
        float(self.state.optimization.skip_count)
      ),

      "optim/clip_count": (
        float(self.state.optimization.clip_count)
      ),

      "perf/examples_per_second": (
        float(self.state.performance.examples_per_second)
      ),

      "perf/steps_per_second": (
        float(self.state.performance.steps_per_second)
      ),

      "perf/wall_time": (
        float(self.state.performance.elapsed_time)
      ),
    }

    if self.state.optimization.grad_scale is not None:
      output["optim/grad_scale"] = (
        float(self.state.optimization.grad_scale)
      )

    if metrics:
      output.update(
        {str(k): float(v) for k, v in metrics.items()}
      )

    self._write_tensorboard(output)

    if self.wandb_run is not None:
      self._write_wandb(output)

    self._last_log_step = self.state.global_step
    self._last_log_time = monotonic()

    return output

  def _write_tensorboard(
    self, metrics: dict[str, float]
  ) -> None:
    if self._writer is None:
      return

    for name, value in metrics.items():
      self._writer.add_scalar(  # type: ignore
        name,
        value,
        self.state.global_step,
      )

    self._writer.flush()  # type: ignore

  def _write_wandb(self, metrics: dict[str, float]) -> None:
    self.wandb_run.log( # type: ignore
      metrics,
      step=self.state.global_step,
    )

  def log_evaluation(
    self,
    metrics: dict[str, float],
  ) -> dict[str, float]:
    output = {
      f"eval/{name}": float(value)
      for name, value in metrics.items()
    }

    output["eval/step"] = float(self.state.global_step)

    self._write_tensorboard(output)

    if self.wandb_run is not None:
      self._write_wandb(output)

    return output

  def state_dict(self) -> dict[str, Any]:
    self.update_performance()

    return asdict(self.state)

  def load_state_dict(
    self,
    state: dict[str, Any],
  ) -> None:
    required = {
      "global_step",
      "epoch",
      "batch_cursor",
      "examples",
      "window",
      "optimization",
      "performance",
    }

    missing = required - set(state)
    if missing:
      raise ValueError(
        f"tracker state missing keys: {sorted(missing)}"
      )

    self.state = TrackerState(
      global_step=int(state["global_step"]),
      epoch=int(state["epoch"]),
      batch_cursor=int(state["batch_cursor"]),
      examples=int(state["examples"]),
      window=MetricWindow(
        loss_sum=float(state["window"]["loss_sum"]),
        examples=int(state["window"]["examples"]),
      ),
      optimization=OptimizationStats(
        grad_norm=state["optimization"]["grad_norm"],
        clip_count=int(
          state["optimization"]["clip_count"]
        ),
        update_count=int(
          state["optimization"]["update_count"]
        ),
        skip_count=int(
          state["optimization"]["skip_count"]
        ),
        grad_scale=state["optimization"]["grad_scale"],
      ),
      performance=PerformanceStats(
        elapsed_time=float(
          state["performance"]["elapsed_time"]
        ),
        examples_per_second=float(
          state["performance"]["examples_per_second"]
        ),
        steps_per_second=float(
          state["performance"]["steps_per_second"]
        ),
      ),
    )

    self._start_time = monotonic()
    self._last_log_time = self._start_time
    self._last_log_step = self.state.global_step

  def prepare_tensorboard_resume(self) -> None:
    if self._writer is None:
      return

    self._writer.close()  # type: ignore

    try:
      from torch.utils.tensorboard import SummaryWriter

      self._writer = SummaryWriter(
        log_dir=str(self.run_dir / "tensorboard"),
        purge_step=self.state.global_step,
      )
    except ImportError:
      self._writer = None

  def flush(self) -> None:
    if self._writer is not None:
      self._writer.flush()  # type: ignore

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()  # type: ignore
      self._writer = None

    if self.wandb_run is not None:
      finish = getattr(self.wandb_run, "finish", None)
      if finish is not None:
        finish()