from __future__ import annotations

# Native Import(s)
import re
from pathlib import Path
from typing import Any
import time
import os

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

  def save(self, *, step: int, state: dict[str, Any]) -> Path:
    path = self.step_path(step)

    self._atomic_torch_save(
      state,
      path,
    )

    self._write_current_pointer(
      path.name
    )

    self._apply_retention()

    return path

  def save_best(self, *, step: int, state: dict[str, Any]) -> Path:
    path = self.best_path()

    self._atomic_torch_save(state, path)

    return path

  def save_milestone(self, *, step: int, state: dict[str, Any]) -> Path:
    path = self.milestone_path(step)

    self._atomic_torch_save(state, path)

    return path

  def save_emergency(self, *, state: dict[str, Any]) -> Path:
    if self.emergency_keep == 0:
      raise CheckpointError(
        "emergency checkpoint retention is disabled"
      )

    index = self._next_emergency_index()
    path = self.emergency_path(index)

    self._atomic_torch_save(state, path)
    self._apply_emergency_retention()

    return path

  def load_latest(self, *, map_location: Any = "cpu") -> tuple[dict[str, Any], Path]:
    pointed = self._read_current_pointer()

    if pointed is not None:
      path = (
        self.checkpoint_dir / pointed
      )

      if self._valid_step_path(path):
        state = self._load_checkpoint(
          path,
          map_location=map_location,
        )

        return state, path

    candidates = self._step_paths()

    for path in reversed(candidates):
      try:
        state = self._load_checkpoint(
          path,
          map_location=map_location,
        )

        return state, path
      except Exception:
        continue

    raise CheckpointError(
      "no valid step checkpoint found"
    )

  def load(self, path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    path = Path(path)

    if not path.is_absolute():
      path = (
        self.checkpoint_dir / path
      )

    if not path.exists():
      raise CheckpointError(
        f"checkpoint does not exist: {path}"
      )

    return self._load_checkpoint(
      path,
      map_location=map_location,
    )

  def _atomic_torch_save(
    self,
    state: dict[str, Any],
    destination: Path,
  ) -> None:
    import torch

    destination.parent.mkdir(
      parents=True,
      exist_ok=True,
    )

    tmp = destination.with_name(
      destination.name + ".tmp"
    )

    try:
      with open(tmp, "wb") as f:
        torch.save(state, f)

        f.flush()
        os.fsync(f.fileno())

      self._replace_with_retry(
        tmp,
        destination,
      )

    except Exception as exc:
      try:
        tmp.unlink()
      except FileNotFoundError:
        pass

      raise CheckpointError(
        "failed to save checkpoint "
        f"{destination}: {exc}"
      ) from exc

  def _replace_with_retry(
    self,
    source: Path,
    destination: Path,
  ) -> None:
    last_error: Exception | None = None

    for attempt in range(
      self.replace_retries + 1
    ):
      try:
        os.replace(
          source,
          destination,
        )

        return

      except OSError as exc:
        last_error = exc

        if attempt >= self.replace_retries:
          break

        delay = (
          0.05 * (2 ** attempt)
        )

        time.sleep(delay)

    raise CheckpointError(
      "atomic checkpoint replacement failed "
      f"after {self.replace_retries + 1} "
      f"attempts: {destination}"
    ) from last_error

  def _write_current_pointer(
    self,
    filename: str,
  ) -> None:
    tmp = self.current_path.with_name(
      "current.pt.tmp"
    )

    try:
      with open(
        tmp,
        "w",
        encoding="utf-8",
      ) as f:
        f.write(filename)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

      self._replace_with_retry(
        tmp,
        self.current_path,
      )

    except Exception as exc:
      try:
        tmp.unlink()
      except FileNotFoundError:
        pass

      raise CheckpointError(
        "failed to update current pointer"
      ) from exc

  def _read_current_pointer(self) -> str | None:
    if not self.current_path.exists():
      return None

    try:
      text = self.current_path.read_text(encoding="utf-8").strip()

      if not text:
        return None

      return text

    except OSError:
      return None

  def _step_paths(self) -> list[Path]:
    entries: list[tuple[int, Path]] = []

    for path in self.checkpoint_dir.iterdir():
      match = _STEP_RE.match(path.name)

      if match is None:
        continue

      step = int(match.group(1))

      entries.append((step, path))

    entries.sort(key=lambda item: item[0])

    return [path for _, path in entries]

  def _valid_step_path(self, path: Path) -> bool:
    if not path.is_file():
      return False

    return _STEP_RE.match(path.name) is not None

  def _load_checkpoint(
    self,
    path: Path,
    *,
    map_location: Any,
  ) -> dict[str, Any]:
    import torch

    try:
      state = torch.load(
        path,
        map_location=map_location,
      )
    except Exception as exc:
      raise CheckpointError(
        f"failed to load checkpoint: {path}"
      ) from exc

    if not isinstance(state, dict):
      raise CheckpointError(
        f"checkpoint is not a dictionary: {path}"
      )

    return state

  def _apply_retention(self) -> None:
    paths = self._step_paths()

    if self.keep_last == 0:
      to_remove = paths
    else:
      to_remove = paths[
        : -self.keep_last
      ]

    for path in to_remove:
      self._remove_checkpoint(path)

  def _apply_emergency_retention(
    self,
  ) -> None:
    paths = sorted(
      self.checkpoint_dir.glob(
        "emergency_*.pt"
      ),
      key=self._emergency_index,
    )

    if self.emergency_keep == 0:
      to_remove = paths
    else:
      to_remove = paths[
        : -self.emergency_keep
      ]

    for path in to_remove:
      self._remove_checkpoint(
        path
      )

  def _remove_checkpoint(
    self,
    path: Path,
  ) -> None:
    try:
      path.unlink()
    except FileNotFoundError:
      pass
    except OSError as exc:
      raise CheckpointError(
        f"failed to remove checkpoint: {path}"
      ) from exc

  def _next_emergency_index(
    self,
  ) -> int:
    paths = self.checkpoint_dir.glob(
      "emergency_*.pt"
    )

    indices = [
      self._emergency_index(path)
      for path in paths
    ]

    if not indices:
      return 1

    return max(indices) + 1

  @staticmethod
  def _emergency_index(
    path: Path,
  ) -> int:
    match = re.match(
      r"^emergency_(\d+)\.pt$",
      path.name,
    )

    if match is None:
      return -1

    return int(match.group(1))

  @staticmethod
  def _validate_step(step: int) -> None:
    if step < 0:
      raise ValueError(
        "step must be non-negative"
      )