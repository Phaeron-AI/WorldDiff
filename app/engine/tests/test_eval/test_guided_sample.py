from __future__ import annotations

import torch
from torch import nn

from src.eval.validate import guided_sample
from src.models.flow import sample
from src.models.flow.guidance import cfg_velocity
from src.models.flow.sampler import init_target_noise


NUM_VIEWS = 4
NUM_INPUT = 2
LATENT_C = 4
HW = 8
SEED_U = 11
SEED_C = 22



class SensitiveModel(nn.Module):
  """Parameter-free, deterministic, and dependent on every contract input."""

  def forward(
    self,
    x: torch.Tensor,
    rays: torch.Tensor,
    t: torch.Tensor,
    cond_mask: torch.Tensor,
  ) -> torch.Tensor:
    mixed = x + 0.5 * x.mean(dim=1, keepdim=True)          # cross-view
    ray_term = 0.25 * rays[:, :, : x.shape[2]]             # rays
    t_term = t.view(-1, 1, 1, 1, 1)                        # timestep
    mask_term = cond_mask.float().view(*cond_mask.shape, 1, 1, 1)
    return torch.tanh(mixed + ray_term + t_term + mask_term)


class BranchConstantModel(nn.Module):
  """Returns one constant unconditionally, another conditionally.

  Makes v_u and v_c trivially distinguishable, so a transposed
  cfg_velocity call at the call site produces a different trajectory.
  """

  UNCOND = 1.0
  COND = 2.0

  def forward(
    self,
    x: torch.Tensor,
    rays: torch.Tensor,
    t: torch.Tensor,
    cond_mask: torch.Tensor,
  ) -> torch.Tensor:
    is_uncond = not bool(cond_mask.any())
    return torch.full_like(x, self.UNCOND if is_uncond else self.COND)


class Recorder(nn.Module):
  """Captures the inputs and outputs of every model call.

  guided_sample returns only z_c, so this is the only way to observe the
  unconditional branch. The recorded x at step i IS the z^u trajectory.
  """

  def __init__(self, inner: nn.Module) -> None:
    super().__init__()
    self.inner = inner
    self.calls: list[dict] = []

  def forward(
    self,
    x: torch.Tensor,
    rays: torch.Tensor,
    t: torch.Tensor,
    cond_mask: torch.Tensor,
  ) -> torch.Tensor:
    v = self.inner(x, rays, t, cond_mask)
    self.calls.append(
      {
        "uncond": not bool(cond_mask.any()),
        "x": x.clone(),
        "v": v.clone(),
      }
    )
    return v

  def reset(self) -> None:
    self.calls.clear()

  def uncond(self) -> list[dict]:
    return [c for c in self.calls if c["uncond"]]

  def cond(self) -> list[dict]:
    return [c for c in self.calls if not c["uncond"]]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _cond_mask() -> torch.Tensor:
  mask = torch.zeros(1, NUM_VIEWS, dtype=torch.bool)
  mask[:, :NUM_INPUT] = True
  return mask


def _scene(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
  g = torch.Generator().manual_seed(seed)
  z0 = torch.randn(1, NUM_VIEWS, LATENT_C, HW, HW, generator=g)
  rays = torch.randn(1, NUM_VIEWS, 6, HW, HW, generator=g)
  return z0, rays


def _run(model, z0, rays, cond_mask, *, w: float, num_steps: int) -> torch.Tensor:
  return guided_sample(
    model,
    z0,
    rays,
    cond_mask,
    w=w,
    num_steps=num_steps,
    g_uncond=torch.Generator().manual_seed(SEED_U),
    g_cond=torch.Generator().manual_seed(SEED_C),
  )


# ----------------------------------------------------------------------
#  The unconditional branch cannot see the observations
# ----------------------------------------------------------------------


def test_unconditional_branch_ignores_observed_latents() -> None:
  """Replace every observed latent; z^u and every v_u must be bit-identical.

  This is the direct test of derivation 71. The single-state design, where
  the conditional state is fed to the unconditional branch under an all-zero
  mask, fails it outright.
  """
  cond_mask = _cond_mask()
  z0_a, rays = _scene(1)
  z0_b, _ = _scene(999)

  assert not torch.equal(z0_a, z0_b), "the two scenes must actually differ"

  recorder = Recorder(SensitiveModel())

  _run(recorder, z0_a, rays, cond_mask, w=1.5, num_steps=4)
  run_a = recorder.uncond()
  cond_a = recorder.cond()

  recorder.reset()
  _run(recorder, z0_b, rays, cond_mask, w=1.5, num_steps=4)
  run_b = recorder.uncond()
  cond_b = recorder.cond()

  assert len(run_a) == len(run_b) == 4

  for i, (a, b) in enumerate(zip(run_a, run_b)):
    assert torch.equal(a["x"], b["x"]), f"z^u diverged at step {i}"
    assert torch.equal(a["v"], b["v"]), f"v_u diverged at step {i}"

  # Anti-vacuity: the conditional branch MUST respond to the swap, or the
  # test above would pass for a model that ignores its inputs entirely.
  assert any(
    not torch.equal(a["v"], b["v"]) for a, b in zip(cond_a, cond_b)
  ), "conditional branch did not react to different observations"


def test_unconditional_state_advances() -> None:
  """z^u must integrate, not sit at its initial noise.

  Not covered by the leakage test: a frozen z^u is perfectly independent of
  the observed latents, so leakage passes on it.
  """
  cond_mask = _cond_mask()
  z0, rays = _scene(2)

  recorder = Recorder(SensitiveModel())
  _run(recorder, z0, rays, cond_mask, w=1.0, num_steps=4)

  states = [c["x"] for c in recorder.uncond()]
  assert len(states) == 4

  for i in range(len(states) - 1):
    assert not torch.equal(states[i], states[i + 1]), f"z^u frozen at step {i}"


def test_conditioning_views_remain_pinned() -> None:
  """Re-pinning, not a guidance mask, is what protects the observations."""
  cond_mask = _cond_mask()
  z0, rays = _scene(3)

  z_c = _run(SensitiveModel(), z0, rays, cond_mask, w=2.0, num_steps=4)

  assert torch.equal(z_c[cond_mask], z0[cond_mask])
  assert not torch.equal(z_c[~cond_mask], z0[~cond_mask])


# ----------------------------------------------------------------------
# The call site — argument order
# ----------------------------------------------------------------------


def test_call_site_argument_order() -> None:
  """A transposed cfg_velocity call must change the trajectory.

  w = 3 is chosen deliberately. The correct-vs-swapped residual is
  (1 - 2w)(v_u - v_c), so the detection margin is |1 - 2w|: zero at w = 0.5,
  and largest at w = 3 among the sweep values. A test written at w = 0.5
  passes under a transposed call and proves nothing.
  """
  w = 3.0
  cond_mask = _cond_mask()
  z0, rays = _scene(4)

  z_c = _run(BranchConstantModel(), z0, rays, cond_mask, w=w, num_steps=1)

  v_u = torch.full_like(z0, BranchConstantModel.UNCOND)
  v_c = torch.full_like(z0, BranchConstantModel.COND)

  correct = cfg_velocity(v_u, v_c, w)      # 1 + 3(2-1) = +4
  swapped = cfg_velocity(v_c, v_u, w)      # 2 + 3(1-2) = -1

  # num_steps = 1 -> step_times() == [1.0], step_size() == [1.0]
  init = init_target_noise(
    z0, cond_mask, rng=torch.Generator().manual_seed(SEED_C)
  )
  expected = init - correct
  wrong = init - swapped

  target = ~cond_mask
  assert torch.allclose(z_c[target], expected[target], rtol=0, atol=1e-6)
  assert not torch.allclose(z_c[target], wrong[target], rtol=0, atol=1e-3)


# ----------------------------------------------------------------------
# The primitive — numerical behaviour, not algebraic identity
# ----------------------------------------------------------------------


def test_cfg_velocity_w0_is_bitwise_identity_for_finite_inputs() -> None:
  """0.0 * finite == 0.0 and a + 0.0 == a, so w = 0 returns v_u unchanged."""
  g = torch.Generator().manual_seed(7)
  v_u = torch.randn(50_000, generator=g)
  v_c = torch.randn(50_000, generator=g)

  assert torch.equal(cfg_velocity(v_u, v_c, 0.0), v_u)


def test_cfg_velocity_w1_is_not_bitwise_identity() -> None:
  """w = 1 reduces to v_c in R, not in IEEE 754.

  fl(a + fl(b - a)) != b for roughly a third of ordinary float32 values.
  Contract 1 therefore does not assert bitwise equality between
  cfg_velocity(v_u, v_c, 1.0) and v_c; the routine path and the sweep's
  w = 1 reference both use the conditional branch directly instead.
  """
  g = torch.Generator().manual_seed(8)
  v_u = torch.randn(200_000, generator=g)
  v_c = torch.randn(200_000, generator=g)

  out = cfg_velocity(v_u, v_c, 1.0)

  assert not torch.equal(out, v_c)
  assert torch.allclose(out, v_c, rtol=0, atol=1e-5)


def test_cfg_velocity_extrapolates_beyond_conditional() -> None:
  """w > 1 moves past v_c, away from v_u."""
  v_u = torch.tensor([1.0])
  v_c = torch.tensor([2.0])

  assert torch.allclose(cfg_velocity(v_u, v_c, 1.5), torch.tensor([2.5]))
  assert torch.allclose(cfg_velocity(v_u, v_c, 3.0), torch.tensor([4.0]))


def test_cfg_velocity_signed_zero_and_non_finite() -> None:
  """Two edge cases the w = 0 identity does not cover."""
  # -0.0 comes back as +0.0: equal in value, different in bits.
  out = cfg_velocity(torch.tensor([-0.0]), torch.tensor([0.5]), 0.0)
  assert out.item() == 0.0
  assert not torch.signbit(out).item()

  # 0 * inf == nan, so a non-finite conditional poisons w = 0, where the
  # conditional branch mathematically contributes nothing.
  out = cfg_velocity(
    torch.tensor([1.0]), torch.tensor([float("inf")]), 0.0
  )
  assert torch.isnan(out).item()



def test_guided_sample_at_w1_tracks_the_conditional_sampler() -> None:
  """At w = 1 the guided trajectory follows the conditional one closely.

  Closeness, not equality: guided_sample computes v_u + 1*(v_c - v_u) while
  sample() uses v_c directly, and those are different arithmetic expressions
  that are only algebraically equal. The bitwise fact is locked separately
  in test_cfg_velocity_w1_is_not_bitwise_identity.
  """
  cond_mask = _cond_mask()
  z0, rays = _scene(5)
  model = SensitiveModel()

  z_guided = _run(model, z0, rays, cond_mask, w=1.0, num_steps=4)
  z_plain = sample(
    model,
    z0,
    rays,
    cond_mask,
    num_steps=4,
    rng=torch.Generator().manual_seed(SEED_C),
  )

  assert torch.allclose(z_guided, z_plain, rtol=0, atol=1e-4)