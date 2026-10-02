from collections import OrderedDict

import pytest
import torch
from torch import nn


from src.training.ema import EMA


def make_model():
  torch.manual_seed(1234)
  return nn.Linear(4, 3)


def test_ema_closed_form():
  model = make_model()

  beta = 0.9
  num_updates = 25

  with torch.no_grad():
    for param in model.parameters():
      param.fill_(2.0)

  ema = EMA(
    model,
    beta=beta,
  )

  # Initial EMA value a = 0.0
  with torch.no_grad():
    for shadow in ema.shadow.values():
      shadow.zero_()

  # Hold model parameters at c = 2.0.
  expected_factor = beta ** num_updates
  expected = (
    expected_factor * 0.0
    + (1.0 - expected_factor) * 2.0
  )

  for _ in range(num_updates):
    ema.update(model)

  max_error = max(
    torch.max(
      torch.abs(
        shadow - expected
      )
    ).item()
    for shadow in ema.shadow.values()
  )

  assert max_error < 2e-6


def test_ema_fp32_shadow():
  model = make_model().bfloat16()

  ema = EMA(
    model,
    beta=0.9999,
  )

  for name, shadow in ema.shadow.items():
    assert shadow.dtype == torch.float32
    assert shadow.requires_grad is False

    model_param = dict(model.named_parameters())[name]

    assert shadow.data_ptr() != model_param.data_ptr()


def test_ema_state_roundtrip():
  model = make_model()

  ema = EMA(
    model,
    beta=0.9999,
  )

  with torch.no_grad():
    for param in model.parameters():
      param.add_(1.5)

  ema.update(model)

  state = ema.state_dict()

  restored = EMA(
    model,
    beta=0.5,
  )

  restored.load_state_dict(state)

  assert restored.beta == pytest.approx(
    ema.beta
  )

  assert set(restored.shadow.keys()) == set(
    ema.shadow.keys()
  )

  for name in ema.shadow:
    assert torch.equal(
      restored.shadow[name],
      ema.shadow[name],
    )

  # state_dict() must be a copy.
  name = next(iter(state["shadow"]))

  state["shadow"][name].add_(100.0)

  assert not torch.equal(
    state["shadow"][name],
    restored.shadow[name],
  )

  assert torch.equal(
    restored.shadow[name],
    ema.shadow[name],
  )


def test_ema_eval_cycle_restores_raw():
  model = make_model()

  ema = EMA(
    model,
    beta=0.9,
  )

  # Create distinct raw and EMA weights.
  with torch.no_grad():
    for param in model.parameters():
      param.fill_(3.0)

  ema.update(model)

  with torch.no_grad():
    for param in model.parameters():
      param.fill_(7.0)

  raw_before = OrderedDict(
    (
      name,
      param.detach().clone(),
    )
    for name, param in model.named_parameters()
  )

  ema_before = OrderedDict(
    (
      name,
      shadow.detach().clone(),
    )
    for name, shadow in ema.shadow.items()
  )

  ema.store(model)

  try:
    ema.copy_to(model)

    # Model should now contain EMA weights.
    for name, param in model.named_parameters():
      assert torch.equal(
        param,
        ema_before[name].to(
          device=param.device,
          dtype=param.dtype,
        ),
      )

    # EMA itself must remain unchanged.
    for name, shadow in ema.shadow.items():
      assert torch.equal(
        shadow,
        ema_before[name],
      )

  finally:
    ema.restore(model)

  # Raw training weights must be recovered exactly.
  for name, param in model.named_parameters():
    assert torch.equal(
      param,
      raw_before[name],
    )

  # EMA must still be unchanged.
  for name, shadow in ema.shadow.items():
    assert torch.equal(
      shadow,
      ema_before[name],
    )


def test_ema_guards_while_stored():
  model = make_model()

  ema = EMA(
    model,
    beta=0.9999,
  )

  # Establish a valid EMA state.
  with torch.no_grad():
    for param in model.parameters():
      param.fill_(2.0)

  ema.update(model)

  state = ema.state_dict()

  ema.store(model)

  # Double store must be rejected.
  with pytest.raises(
    RuntimeError,
    match="already stored",
  ):
    ema.store(model)

  # Loading an EMA checkpoint during an active
  # evaluation transaction must be rejected.
  with pytest.raises(
    RuntimeError,
    match="stored",
  ):
    ema.load_state_dict(state)

  # First restore is valid.
  ema.restore(model)

  # Second restore must be rejected.
  with pytest.raises(
    RuntimeError,
    match="No stored model parameters",
  ):
    ema.restore(model)