from __future__ import annotations

# Native Import(s)
import math

def learning_rate_scheduler(
  step: int, *, max_lr: float, warmup_steps: int, total_steps: int
) -> float:
  if step < 0:
    raise ValueError(f"step must be apositive inte, go {step}")
  if max_lr < 0.0:
    raise ValueError(f"max learning rate must be positive flost, got {max_lr}")
  if warmup_steps < 1:
    raise ValueError(f"warmup_steps must be more than 1, got {warmup_steps}")
  if total_steps <= warmup_steps:
    raise ValueError(
      f"warmup steps < total steps"
      f"warmup_steps: {warmup_steps}"
      f"total_steps: {total_steps}"
    )

  if step >= total_steps:
    step = 0

  if step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps

  decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)

  return 0.5 * max_lr *(1.0 + math.cos(math.pi * decay_progress))