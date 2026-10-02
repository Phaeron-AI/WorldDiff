import math

# Third Party Import(s)
import pytest
import torch
from scipy.stats import chisquare

# Local Import(s)
from src.models.flow.loss import masked_flow_matching_loss
from src.training.view_roles import sample_view_roles

def test_view_roles_k_distribution():
  B = 200_000
  V = 4
  p_drop = 0.1

  generator = torch.Generator(device="cpu")
  generator.manual_seed(12345)

  cond_mask = sample_view_roles(
    B,
    V,
    p_drop=p_drop,
    rng=generator,
    device="cpu",
  )

  k = cond_mask.sum(dim=1)
  counts = torch.bincount(k, minlength=V).numpy()

  expected_probabilities = torch.tensor(
    [
      p_drop,
      *[
        (1.0 - p_drop) / (V - 1)
        for _ in range(V - 1)
      ],
    ],
    dtype=torch.float64,
  )

  expected_counts = B * expected_probabilities.numpy()

  statistic, p_value = chisquare(
    f_obs=counts,
    f_exp=expected_counts,
  )

  assert p_value > 1e-3
  assert k.max().item() < V

def test_view_roles_uniform_subset():
  B = 200_000
  V = 6
  p_drop = 0.0

  generator = torch.Generator(device="cpu")
  generator.manual_seed(23456)

  cond_mask = sample_view_roles(
    B,
    V,
    p_drop=p_drop,
    rng=generator,
    device="cpu",
  )

  k = cond_mask.sum(dim=1)

  for k_value in range(1, V):
    rows = k == k_value

    selected = cond_mask[rows].float().mean(dim=0)
    expected = k_value / V

    assert torch.allclose(
      selected,
      torch.full_like(selected, expected),
      atol=0.01,
    ), (
      f"k={k_value}: observed={selected.tolist()}, "
      f"expected={expected}"
    )


def test_view_roles_deterministic_and_resumable():
  B = 32
  V = 5
  p_drop = 0.1

  generator_a = torch.Generator(device="cpu")
  generator_a.manual_seed(34567)

  generator_b = torch.Generator(device="cpu")
  generator_b.manual_seed(34567)

  masks_a = []
  masks_b = []

  for _ in range(5):
    masks_a.append(
      sample_view_roles(
        B,
        V,
        p_drop=p_drop,
        rng=generator_a,
        device="cpu",
      )
    )
    masks_b.append(
      sample_view_roles(
        B,
        V,
        p_drop=p_drop,
        rng=generator_b,
        device="cpu",
      )
    )

  for mask_a, mask_b in zip(masks_a, masks_b):
    assert torch.equal(mask_a, mask_b)

  generator_c = torch.Generator(device="cpu")
  generator_c.manual_seed(45678)

  for _ in range(3):
    sample_view_roles(
      B,
      V,
      p_drop=p_drop,
      rng=generator_c,
      device="cpu",
    )

  saved_state = generator_c.get_state()

  expected = []
  for _ in range(3):
    expected.append(
      sample_view_roles(
        B,
        V,
        p_drop=p_drop,
        rng=generator_c,
        device="cpu",
      )
    )

  generator_d = torch.Generator(device="cpu")
  generator_d.set_state(saved_state)

  resumed = []
  for _ in range(3):
    resumed.append(
      sample_view_roles(
        B,
        V,
        p_drop=p_drop,
        rng=generator_d,
        device="cpu",
      )
    )

  for expected_mask, resumed_mask in zip(expected, resumed):
    assert torch.equal(expected_mask, resumed_mask)

def test_view_roles_always_has_target():
  B = 10_000
  V = 8

  generator = torch.Generator(device="cpu")
  generator.manual_seed(56789)

  cond_mask = sample_view_roles(
    B,
    V,
    p_drop=0.2,
    rng=generator,
    device="cpu",
  )

  target_counts = (~cond_mask).sum(dim=1)

  assert torch.all(target_counts >= 1)