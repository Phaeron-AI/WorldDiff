import pytest

from src.training.lr_schedule import learning_rate


def test_lr_warmup_and_peak():
  max_lr = 1e-3
  warmup_steps = 100
  total_steps = 1000

  learning_rates = [
    learning_rate(
      step=step,
      max_lr=max_lr,
      warmup_steps=warmup_steps,
      total_steps=total_steps,
    )
    for step in range(warmup_steps)
  ]

  assert learning_rates[0] > 0.0
  assert learning_rates[0] == pytest.approx(
    max_lr / warmup_steps
  )

  for previous, current in zip(
    learning_rates,
    learning_rates[1:],
  ):
    assert current > previous

  assert learning_rates[-1] == pytest.approx(max_lr)

  assert learning_rate(
    step=warmup_steps,
    max_lr=max_lr,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
  ) == pytest.approx(max_lr)


def test_lr_zero_at_and_after_total_steps():
  max_lr = 1e-3
  warmup_steps = 100
  total_steps = 1000

  assert learning_rate(
    step=total_steps,
    max_lr=max_lr,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
  ) == 0.0

  for step in [total_steps + 1, total_steps + 50, total_steps + 99]:
    assert learning_rate(
      step=step,
      max_lr=max_lr,
      warmup_steps=warmup_steps,
      total_steps=total_steps,
    ) == 0.0


def test_lr_monotone_decay():
  max_lr = 1e-3
  warmup_steps = 100
  total_steps = 1000

  learning_rates = [
    learning_rate(
      step=step,
      max_lr=max_lr,
      warmup_steps=warmup_steps,
      total_steps=total_steps,
    )
    for step in range(warmup_steps, total_steps)
  ]

  for previous, current in zip(
    learning_rates,
    learning_rates[1:],
  ):
    assert current <= previous

  assert learning_rates[0] == pytest.approx(max_lr)
  assert learning_rates[-1] >= 0.0


def test_lr_invalid_inputs():
  common_kwargs = {
    "max_lr": 1e-3,
    "warmup_steps": 100,
    "total_steps": 1000,
  }

  with pytest.raises(ValueError, match="step"):
    learning_rate(
      step=-1,
      **common_kwargs,
    )

  with pytest.raises(ValueError, match="max_lr"):
    learning_rate(
      step=0,
      max_lr=-1e-3,
      warmup_steps=100,
      total_steps=1000,
    )

  with pytest.raises(ValueError, match="warmup_steps"):
    learning_rate(
      step=0,
      max_lr=1e-3,
      warmup_steps=0,
      total_steps=1000,
    )

  with pytest.raises(ValueError, match="total_steps"):
    learning_rate(
      step=0,
      max_lr=1e-3,
      warmup_steps=1000,
      total_steps=1000,
    )

  with pytest.raises(ValueError, match="total_steps"):
    learning_rate(
      step=0,
      max_lr=1e-3,
      warmup_steps=1001,
      total_steps=1000,
    )