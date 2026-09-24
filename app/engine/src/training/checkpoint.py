from __future__ import annotations

# Native Import(s)
import re
from pathlib import Path

# Third Party Import(s)
# Local Import(s)

_STEP_RE = re.compile(
  r"^step_(\d+)\.pt$"
)

class CheckpointError(RuntimeError):
  pass

class CheckpointManager:
  def __init__(
    self, 
    run_dir: str | Path, 
    *, 
    keep_last: int = 3, 
    emergency_keep: int = 3, 
    replace_retries: int = 5
  ) -> None:

    if keep_last < 0:
      raise ValueError(f"keep_last must be non-negative, got {keep_last}")

    if emergency_keep < 0:
      raise ValueError(f"emergency_keep must be non-negative, got {emergency_keep}")

    if replace_retries < 0:
      raise ValueError(f"replace_retries must be non-negative, got {replace_retries}")

    self.run_dir = Path(run_dir)
    self.checkpoint_dir = Path(self.run_dir / "checkpoints")

    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    self.keep_last = int(keep_last)
    self.emergency_keep = int(emergency_keep)
    self.replace_retries = int(replace_retries)

  def step_path(self, step: int) -> Path:
    return Path(self.checkpoint_dir / f"step_{step:06d}.pt")

  @property
  def current_path(self) -> Path:
    return Path(self.checkpoint_dir / "current.pt")

  def best_path(self) -> Path:
    return Path(self.checkpoint_dir / "best.pt")

  def milestone_path(self, step: int) -> Path:
    return Path(self.checkpoint_dir / "milestone_{step:06d}.pt")

  def emergency_path(self, index: int) -> Path:
    if index < 1:
      raise ValueError(f"emergency index must be non-negative, got {index}")

    return Path(self.checkpoint_dir / f"emergency_{index:03d}.pt")\

  