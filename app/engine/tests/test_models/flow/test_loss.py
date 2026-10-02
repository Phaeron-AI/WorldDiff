import torch

from src.models.flow.loss import (
  _masked_flow_matching_loss_per_sample,
  masked_flow_matching_loss,
)


def test_loss_per_sample_mean():
  # Two samples with different numbers of target views.
  #
  # Sample 0:
  #   1 target view, MSE = 1
  #
  # Sample 1:
  #   3 target views, MSEs = 2, 3, 4
  #
  # Estimator (b):
  #   sample 0 = 1
  #   sample 1 = (2 + 3 + 4) / 3 = 3
  #   batch mean = (1 + 3) / 2 = 2
  #
  # Estimator (a), global supervised-element mean:
  #   (1 + 2 + 3 + 4) / 4 = 2.5

  v_pred = torch.tensor(
    [
      [
        [[[1.0]]],
        [[[0.0]]],
        [[[0.0]]],
        [[[0.0]]],
      ],
      [
        [[[0.0]]],
        [[[2.0**0.5]]],
        [[[3.0**0.5]]],
        [[[2.0]]],
      ],
    ]
  )

  target = torch.zeros_like(v_pred)

  target_mask = torch.tensor(
    [
      [True, False, False, False],
      [False, True, True, True],
    ]
  )

  per_sample = _masked_flow_matching_loss_per_sample(
    v_pred,
    target,
    target_mask,
  )

  loss = masked_flow_matching_loss(
    v_pred,
    target,
    target_mask,
  )

  assert torch.allclose(
    per_sample,
    torch.tensor([1.0, 3.0]),
    atol=1e-6,
  )

  assert torch.allclose(
    loss,
    torch.tensor(2.0),
    atol=1e-6,
  )

  # Explicitly distinguish estimator (b) from the old
  # global supervised-element mean.
  assert not torch.allclose(
    loss,
    torch.tensor(2.5),
  )