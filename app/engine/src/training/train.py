from __future__ import annotations

# Native Import(s)
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, Mapping, Sequence

# Third Party Import(s)
import torch
from torch import nn, Tensor

# Local Import(s)
from src.training.ema import EMA
from src.training.view_roles import sample_view_roles
from src.training.lr_schedule import learning_rate
from src.training.step import TrainStepResult, train_step

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
  warmup_steps: int = 100
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
  eval_seed: int = 3

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


class Trainer:
  def __init__(
    self, 
    cfg: TrainingConfig, model: nn.Module,
    dataset: Sequence[Mapping[str, Tensor]],
    *,
    device: torch.device | None = None,
    hooks: Hooks | None = None,
    optimizer: torch.optim.Optimizer | None = None
  ) -> None:
    cfg.validate()
    self.cfg = cfg
    if device:
      self.device = torch.device(device)
    self.model = model.to(self.device)
    self.dataset = dataset
    self.hooks = hooks or Hooks()

    if len(dataset) < cfg.batch_size:
      raise ValueError(
        f"dataset has {len(dataset)} scenes but batch_size is {cfg.batch_size}; "
        "with drop_last=True this yields zero training batches"
      )

    self.optimizer = optimizer or torch.optim.Adam(
      self.model.parameters(),
      lr=cfg.max_lr,
      betas=cfg.betas,
      weight_decay=cfg.weight_decay,
    )
    self.ema = EMA(self.model, beta=cfg.ema_beta) if cfg.ema_enabled else None
    self.scaler = (
      torch.amp.GradScaler(self.device.type, init_scale=cfg.init_scale)
      if cfg.precision == "fp16"
      else None
    )
    self.generators = Generators(cfg, self.device)
    self.cursor = EpochCursor(
      data_seed=cfg.data_seed, length=len(dataset), batch_size=cfg.batch_size
    )

    self.optimizer_step = 0
    self.attempted_step = 0
    self.samples_seen = 0
    self.skipped_steps = 0
    self.consecutive_skips = 0
    self.window = PerKAccumulator()
    self.window_loss_sum = 0.0
    self.window_loss_count = 0
    self.last_loss = float("nan")

  def _fetch_batch(self) -> tuple[Tensor, Tensor, int]:
    indices = self.cursor.next_indices()
    items = [self.dataset[i] for i in indices]
    z0 = torch.stack([item["latents"] for item in items])
    rays = torch.stack([item["rays"] for item in items])
    if self.cfg.pin_memory and self.device.type == "cuda":
      z0, rays = z0.pin_memory(), rays.pin_memory()
    non_blocking = self.cfg.pin_memory and self.device.type == "cuda"
    z0 = z0.to(self.device, non_blocking=non_blocking)
    rays = rays.to(self.device, non_blocking=non_blocking)
    return z0, rays, len(indices)

  def state_dict(self) -> dict[str, Any]:
    return {
      "format": 1,
      "config": asdict(self.cfg),
      "cache_hash": self.cfg.cache_hash,
      "numerical_flags": numerical_flags(),
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "ema": self.ema.state_dict() if self.ema is not None else None,
      "scaler": self.scaler.state_dict() if self.scaler is not None else None,
      "generators": self.generators.state_dict(),
      "cursor": self.cursor.state_dict(),
      "counters": {
        "optimizer_step": self.optimizer_step,
        "attempted_step": self.attempted_step,
        "samples_seen": self.samples_seen,
        "skipped_steps": self.skipped_steps,
        "consecutive_skips": self.consecutive_skips,
      },
      "window": {
        "per_k": self.window.state_dict(),
        "loss_sum": self.window_loss_sum,
        "loss_count": self.window_loss_count,
      },
    }

  def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
    if strict:
      self._check_resume_compatible(state)
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    if self.ema is not None and state["ema"] is not None:
      self.ema.load_state_dict(state["ema"])
    if self.scaler is not None and state["scaler"] is not None:
      self.scaler.load_state_dict(state["scaler"])
    self.generators.load_state_dict(state["generators"])
    self.cursor = EpochCursor.from_state_dict(state["cursor"])
    counters = state["counters"]
    self.optimizer_step = counters["optimizer_step"]
    self.attempted_step = counters["attempted_step"]
    self.samples_seen = counters["samples_seen"]
    self.skipped_steps = counters["skipped_steps"]
    self.consecutive_skips = counters["consecutive_skips"]
    self.window.load_state_dict(state["window"]["per_k"])
    self.window_loss_sum = state["window"]["loss_sum"]
    self.window_loss_count = state["window"]["loss_count"]

  def _check_resume_compatible(self, state: Mapping[str, Any]) -> None:
    saved_cfg = state["config"]
    mine = asdict(self.cfg)
    bad = [
      f for f in RESUME_CRITICAL_FIELDS
      if saved_cfg.get(f) != mine.get(f)
    ]
    if bad:
      raise ResumeMismatch(
        "resume-critical config fields differ from the checkpoint: "
        + ", ".join(f"{f}: {saved_cfg.get(f)!r} -> {mine.get(f)!r}" for f in bad)
      )
    if state.get("cache_hash") != self.cfg.cache_hash:
      raise ResumeMismatch(
        f"latent-cache identity differs: checkpoint {state.get('cache_hash')!r}, "
        f"current {self.cfg.cache_hash!r}"
      )
    saved_flags = state.get("numerical_flags", {})
    current = numerical_flags()
    drift = [
      k for k in ("cudnn_deterministic", "cudnn_benchmark", "cudnn_allow_tf32",
                  "matmul_allow_tf32", "torch_version", "cuda_version", "device_name")
      if saved_flags.get(k) != current.get(k)
    ]
    if drift:
      raise ResumeMismatch(
        "numerical execution environment differs from the checkpoint "
        f"({', '.join(drift)}); bit-identical continuation cannot be claimed. "
        "Pass strict=False to resume anyway."
      )

  def _checkpoint(self, tag: str) -> None:
    if self.hooks.on_checkpoint is not None:
      self.hooks.on_checkpoint(self.state_dict(), tag=tag)

  def _log(self) -> None:
    record: dict[str, Any] = {
      "optimizer_step": self.optimizer_step,
      "attempted_step": self.attempted_step,
      "samples_seen": self.samples_seen,
      "skipped_steps": self.skipped_steps,
      "epoch": self.cursor.epoch,
      "cursor": self.cursor.cursor,
      "lr": self._lr_for_next_step(),
      "loss": (
        self.window_loss_sum / self.window_loss_count
        if self.window_loss_count else float("nan")
      ),
      "per_k_loss": self.window.means(),
    }
    if self.scaler is not None:
      record["grad_scale"] = self.scaler.get_scale()
    if self.hooks.on_log is not None:
      self.hooks.on_log(record)
    self.window.reset()
    self.window_loss_sum = 0.0
    self.window_loss_count = 0

  def _evaluate(self) -> None:
    """Derivations 64/65: EMA weights in, eval mode, isolated RNG, restore."""
    if self.hooks.on_eval is None:
      return
    snapshot = self.generators.snapshot()
    eval_generator = torch.Generator(device=self.device)
    eval_generator.manual_seed(self.cfg.eval_seed * 1_000_003 + self.optimizer_step)
    used_ema = False
    if self.ema is not None:
      self.ema.store(self.model)
      self.ema.copy_to(self.model)
      used_ema = True
    self.model.eval()
    try:
      with torch.no_grad():
        self.hooks.on_eval(self.model, step=self.optimizer_step, generator=eval_generator)
    finally:
      if used_ema:
        self.ema.restore(self.model) # type: ignore
      self.model.train()
      self.generators.restore(snapshot)

  def _lr_for_next_step(self) -> float:
    return learning_rate(
      step=min(self.optimizer_step, self.cfg.total_steps),
      max_lr=self.cfg.max_lr,
      warmup_steps=self.cfg.warmup_steps,
      total_steps=self.cfg.total_steps,
    )

  def run(self) -> TrainSummary:
    cfg = self.cfg
    started = time.perf_counter()
    start_epoch = self.cursor.epoch
    threshold = cfg.resolved_max_consecutive_skips()
    self.model.train()                                         
 
    while self.optimizer_step < cfg.total_steps:
      if cfg.max_attempted_steps is not None and self.attempted_step >= cfg.max_attempted_steps:
        raise TrainingDiverged(
          f"attempted-step budget exhausted ({self.attempted_step}) with only "
          f"{self.optimizer_step}/{cfg.total_steps} successful steps"
        )
 
      z0, rays, batch_size = self._fetch_batch()        
      self.attempted_step += 1                       
 
      cond_mask = sample_view_roles(                          
        B=z0.shape[0], V=z0.shape[1], p_drop=cfg.p_drop,
        rng=self.generators.role, device=self.device,
      )
 
      result: TrainStepResult = train_step(
        self.model, self.optimizer, z0, rays,
        G_noise=self.generators.noise,
        G_role=self.generators.role,
        cond_mask=cond_mask,
        ema=self.ema,
        scaler=self.scaler,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate=self._lr_for_next_step(),
        autocast_dtype=cfg.autocast_dtype(),
      )
 
      self.samples_seen += batch_size
      if result.skipped:                                
        self.skipped_steps += 1
        self.consecutive_skips += 1
      else:                                            
        self.optimizer_step += 1
        self.consecutive_skips = 0
        self.last_loss = result.loss
        self.window_loss_sum += result.loss
        self.window_loss_count += 1
        self.window.add(result.per_k_loss, result.cond_mask)       
 
      if self.consecutive_skips > threshold:                       
        self._checkpoint("emergency")
        raise TrainingDiverged(
          f"{self.consecutive_skips} consecutive skipped steps exceeds the "
          f"{cfg.precision} threshold of {threshold} at attempted step "
          f"{self.attempted_step} (optimizer step {self.optimizer_step})"
        )
 
      if result.skipped:
        continue
 
      if cfg.log_every and self.optimizer_step % cfg.log_every == 0:
        self._log()
      if cfg.eval_every and self.optimizer_step % cfg.eval_every == 0:    
        self._evaluate()
      if cfg.checkpoint_every and self.optimizer_step % cfg.checkpoint_every == 0:
        self._checkpoint("periodic")                                      
 
    if self.window_loss_count:
      self._log()
 
    return TrainSummary(
      optimizer_steps=self.optimizer_step,
      attempted_steps=self.attempted_step,
      samples_seen=self.samples_seen,
      skipped_steps=self.skipped_steps,
      epochs_completed=self.cursor.epoch - start_epoch,
      final_loss=self.last_loss,
      wall_time_s=time.perf_counter() - started,
    )


def train(
  cfg: TrainingConfig,
  model: nn.Module,
  dataset: Sequence[Mapping[str, Tensor]],
  *,
  device: torch.device | str = "cpu",
  hooks: Hooks | None = None,
  resume_from: Mapping[str, Any] | None = None,
  strict_resume: bool = True,
) -> TrainSummary:
  """Train `model` for cfg.total_steps SUCCESSFUL optimizer steps."""
  trainer = Trainer(cfg, model, dataset, device=device, hooks=hooks)  # type: ignore
  if resume_from is not None:
    trainer.load_state_dict(resume_from, strict=strict_resume)
  return trainer.run()