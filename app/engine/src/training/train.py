from __future__ import annotations

# Native Import(s)
from dataclasses import dataclass

# Third Party Import(s)
import torch

PRECISIONS = ("fp32", "fp16", "bf16")

RESUME_CRITICAL_FIELDS = (
  "batch_size",
  "precision",
  "max_lr",
  "warmup_steps",
  "max_grad_norm",
  "p_drop",
  "ema_beta",
  "data_seed",
  "noise_seed",
  "role_seed",
  "init_scale",
)

@dataclass
class TrainingConfig:
  # Training Budget
  total_steps: int = 1000
  batch_size: int = 8

  # Optimization
  max_lr: float = 1e-4
  warmup_step: int = 100
  betas: tuple[float, float] = (0.9, 0.999)
  weight_decay: float = 0.0
  max_grad_norm: float | None = 1.0

  # View Roles
  p_drop: float = 0.1

  # EMA
  ema_beta: float = 0.9999
  ema_enabled: bool = True

  # Precision
  precision: str = "fp32"
  init_scale: float = 65536.0
  max_consecutive_skips: int | None = None

  # Seeds
  data_seed: int = 0
  noise_seed: int = 1
  role_seed: int = 2
  eval_Seed: int = 3

  # Cadences
  log_every: int = 50
  eval_every: int = 0 # disabled
  checkpoint_every: int = 0 # disabled

  # Safety (Alternating skip/success patterns)
  max_attempted_steps: int | None = None

  # Data
  pin_memory: bool = False
  cache_hash: str | None = None

  def resolved_max_consecutive_skips(self) -> int:
    if self.max_consecutive_skips is not None:
      return self.max_consecutive_skips
    return 32 if self.precision == "fp16" else 8

  def autocast_dtype(self) -> torch.dtype | None:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.precision]

  def validate(self) -> None:
    if self.precision not in PRECISIONS:
      raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision}")
    if self.total_steps < 1:
      raise ValueError(f"total_steps must be >= 1, got {self.total_steps}")
    if self.batch_size < 1:
      raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
    if self.resolved_max_consecutive_skips() < 1:
      raise ValueError("max_consecutive_skips must be >= 1")
    if self.precision == "fp16":
      import math
      floor = math.ceil(math.log2(self.init_scale))
      if self.resolved_max_consecutive_skips() <= floor:
        raise ValueError(
          "fp16 max_consecutive_skips must exceed ceil(log2(init_scale)) = "
          f"{floor} (derivation 50), got {self.resolved_max_consecutive_skips()}"
        )


class TrainingDiverged(RuntimeError):
  """Raised when consecutive skipped steps exceed the configured threshold."""

class ResumeMismatch(RuntimeError):
  """Raised when a checkpoint does not belong to the current experiment."""