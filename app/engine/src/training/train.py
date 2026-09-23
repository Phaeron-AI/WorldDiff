from __future__ import annotations

# Native Import(s)
from dataclasses import dataclass, field
from typing import Any, Protocol, Mapping

# Third Party Import(s)
import torch
from torch import nn, Tensor

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

class EvalHook(Protocol):
  def __call__(
    self, 
    model: nn.Module, 
    *, 
    step: int, 
    generator: torch.Generator
  ) -> Mapping[str, float] | None: ...

class LogHook(Protocol):
  def __call__(self, record: Mapping[str, Any]) -> None: ...

class CheckpointHook(Protocol):
  def __call__(self, state: Mapping[str, Any], *, tag: str) -> None: ...


@dataclass
class Hooks:
  on_log: LogHook | None = None
  on_eval: EvalHook | None = None
  on_checkpoint: CheckpointHook | None = None

@dataclass
class TrainSummary:
  optimizer_steps: int
  attempted_steps: int
  samples_seen: int
  skipped_steps: int
  epochs_completed: int
  final_loss: float
  wall_time_s: float
  aborted: bool = False


def numerical_flags() -> dict[str, Any]:
  flags: dict[str, Any] = {
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
  }

  return flags

@dataclass
class PerKAccumulator:
  sums: dict[int, float] = field(default_factory=dict)
  counts: dict[int, int] = field(default_factory=dict)

  def add(self, per_k_loss: Mapping[int, float], cond_mask: Tensor) -> None:
    k_per_sample = cond_mask.sum(dim=1).tolist()
    for k, mean in per_k_loss.items():
      n = sum(1 for kk in k_per_sample if int(kk) == int(k))
      if n == 0:
        continue
      self.sums[k] = self.sums.get(k, 0.0) + float(mean) * n
      self.counts[k] = self.counts.get(k, 0) + n

  def means(self) -> dict[int, float]:
    return {
      k: self.sums[k] / self.counts[k] for k in sorted(self.sums) 
      if self.counts[k]
    }

  def reset(self) -> None:
    self.sums.clear()
    self.counts.clear()

  def state_dict(self) -> dict:
    return {"sums": dict(self.sums), "counts": dict(self.counts)}

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    self.sums = {int(k): float(v) for k, v in state["sums"].items()}
    self.counts = {int(k): int(v) for k, v in state["counts"].items()}

class Generators:
  def __init__(self, cfg: TrainingConfig, device: torch.device) -> None:
    self.role = torch.Generator(device="cpu").manual_seed(cfg.role_seed)
    self.noise = torch.Generator(device=device).manual_seed(cfg.noise_seed)
 
  def state_dict(self) -> dict:
    return {"role": self.role.get_state(), "noise": self.noise.get_state()}
 
  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    self.role.set_state(state["role"].cpu() if state["role"].is_cuda else state["role"])
    self.noise.set_state(state["noise"])

  def snapshot(self) -> dict:
    """Full RNG snapshot for the evaluation transaction (derivation 65)."""
    snap = {
      "role": self.role.get_state(),
      "noise": self.noise.get_state(),
      "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
      snap["torch_cuda"] = torch.cuda.get_rng_state_all() # type: ignore
    return snap
 
  def restore(self, snap: Mapping[str, Any]) -> None:
    self.role.set_state(snap["role"])
    self.noise.set_state(snap["noise"])
    torch.set_rng_state(snap["torch_cpu"])
    if "torch_cuda" in snap and torch.cuda.is_available():
      torch.cuda.set_rng_state_all(snap["torch_cuda"])