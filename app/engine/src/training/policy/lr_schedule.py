import math


def learning_rate(
  step: int,
  *,
  max_lr: float,
  warmup_steps: int,
  total_steps: int,
) -> float:
  """Return the learning rate for the next optimizer step.

  Args:
    step: Number of successful optimizer steps already completed.
      The first optimizer step uses step=0.
    max_lr: Peak learning rate. Must be non-negative.
    warmup_steps: Number of successful steps used for linear warmup.
    total_steps: Total number of successful optimizer steps.

  Returns:
    Learning rate for the next optimizer step.

  Schedule:
    - step=0: max_lr / warmup_steps
    - step=warmup_steps-1: max_lr
    - step=warmup_steps: max_lr
    - step=total_steps-1: approximately 0
    - step>=total_steps: exactly 0
  """
  if step < 0:
    raise ValueError(
      f"step must be >= 0, got {step}"
    )

  if max_lr < 0.0:
    raise ValueError(
      f"max_lr must be >= 0, got {max_lr}"
    )

  if warmup_steps < 1:
    raise ValueError(
      f"warmup_steps must be >= 1, got {warmup_steps}"
    )

  if total_steps <= warmup_steps:
    raise ValueError(
      f"total_steps must be greater than warmup_steps; "
      f"got total_steps={total_steps}, "
      f"warmup_steps={warmup_steps}"
    )

  if step >= total_steps:
    return 0.0

  if step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps

  decay_progress = (
    (step - warmup_steps)
    / (total_steps - warmup_steps)
  )

  return 0.5 * max_lr * (
    1.0 + math.cos(math.pi * decay_progress)
  )