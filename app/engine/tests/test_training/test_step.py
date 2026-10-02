"""Regression tests for src.training.step.train_step (P3.2).

Each test pins an invariant verified during P3.2 review (T4-T12). The model
used here is deliberately tiny but *sensitive to every input train_step
forwards* and *couples views*, so that a wiring bug (conditioning skipped,
wrong t, inverted mask, dropped rays) changes the loss and is caught.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from src.models.flow.conditioning import apply_conditioning
from src.models.flow.interpolant import linear_interpolant
from src.models.flow.loss import masked_flow_matching_loss
from src.training.ema import EMA
from src.training.step import train_step

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

B, V, C, H, W = 2, 3, 2, 2, 2


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
class SensitiveModel(nn.Module):
  """v = a*x + b*mean_views(x) + c*mean_ch(rays) + d*t + e*cond_mask.

  - `a`            : per-element path (like TinyModel)
  - `b` (view mix) : target-view output depends on the cond view's contents,
                     so skipping apply_conditioning changes the loss
  - `c`, `d`, `e`  : output depends on rays, t and cond_mask respectively
  All coefficients start non-zero so every path carries gradient.
  """

  def __init__(self) -> None:
    super().__init__()
    self.a = nn.Parameter(torch.tensor(0.9))
    self.b = nn.Parameter(torch.tensor(0.5))
    self.c = nn.Parameter(torch.tensor(0.3))
    self.d = nn.Parameter(torch.tensor(0.7))
    self.e = nn.Parameter(torch.tensor(0.2))

  def forward(self, x, rays, t, cond_mask):
    view_mix = x.mean(dim=1, keepdim=True)                 # [B,1,C,H,W]
    ray_term = rays.mean(dim=2, keepdim=True)              # [B,V,1,H,W]
    t_term = t.to(x.dtype)[:, None, None, None, None]      # [B,1,1,1,1]
    m_term = cond_mask.to(x.dtype)[:, :, None, None, None] # [B,V,1,1,1]
    return (
      self.a * x
      + self.b * view_mix
      + self.c * ray_term
      + self.d * t_term
      + self.e * m_term
    )


def make_batch(seed: int = 0):
  g = torch.Generator().manual_seed(seed)
  z0 = torch.randn(B, V, C, H, W, generator=g)
  rays = torch.randn(B, V, 6, H, W, generator=g)   # non-zero: rays must matter
  cond_mask = torch.tensor(
    [[True, False, False],
     [True, True, False]],
    dtype=torch.bool,
  )
  return z0.to(DEVICE), rays.to(DEVICE), cond_mask.to(DEVICE)


def make_generators(seed: int = 1234):
  G_noise = torch.Generator(device=DEVICE).manual_seed(seed)
  G_role = torch.Generator(device="cpu").manual_seed(seed + 1)
  return G_noise, G_role


def make_model_and_opt(lr: float = 1e-2):
  model = SensitiveModel().to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  return model, optimizer


def snapshot(model: nn.Module):
  return {n: p.detach().clone() for n, p in model.named_parameters()}


def equal_params(a: dict, b: dict) -> bool:
  return all(torch.equal(a[n], b[n]) for n in a)


def adam_state(optimizer):
  return {
    (id(p), k): v.detach().clone()
    for p, st in optimizer.state.items()
    for k, v in st.items()
    if torch.is_tensor(v)
  }


def prime(model, optimizer, z0, rays, cond_mask):
  """One real step so Adam state and EMA are non-trivial before a skip test."""
  G_noise, G_role = make_generators(7)
  r = train_step(
    model, optimizer, z0, rays,
    G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, max_grad_norm=None,
  )
  assert not r.skipped
  assert len(optimizer.state) > 0


def inject_inf(grad):
  return torch.full_like(grad, float("inf"))


# ----------------------------------------------------------------------
# T10 — parity with the verified P2 components
# ----------------------------------------------------------------------
def test_step_parity():
  """train_step == hand-assembled P2 step (linear_interpolant ->
  apply_conditioning -> model -> masked_flow_matching_loss -> Adam) under:
  fp32, no clipping, no EMA, no scaler, explicit cond_mask, injected eps/t.
  Compares loss, every parameter, and Adam's first/second moments."""
  z0, rays, cond_mask = make_batch()
  g = torch.Generator(device=DEVICE).manual_seed(3)
  epsilon = torch.randn(z0.shape, generator=g, device=DEVICE)
  t = torch.tensor([0.3, 0.8], device=DEVICE)

  model_exp, opt_exp = make_model_and_opt()
  model_act = deepcopy(model_exp)
  opt_act = torch.optim.Adam(model_act.parameters(), lr=1e-2)

  z_t = apply_conditioning(linear_interpolant(z0, epsilon, t), z0, cond_mask)
  v = model_exp(z_t, rays, t, cond_mask)
  loss_exp = masked_flow_matching_loss(v, epsilon - z0, ~cond_mask)
  opt_exp.zero_grad(set_to_none=True)
  loss_exp.backward()
  opt_exp.step()

  G_noise, G_role = make_generators()
  result = train_step(
    model_act, opt_act, z0, rays,
    G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, epsilon=epsilon, t=t,
    ema=None, scaler=None, max_grad_norm=None, learning_rate=1e-2,
  )

  assert not result.skipped
  assert result.loss == pytest.approx(loss_exp.item(), rel=1e-6, abs=1e-7)

  exp_params = dict(model_exp.named_parameters())
  for name, p in model_act.named_parameters():
    assert torch.allclose(p, exp_params[name], rtol=1e-6, atol=1e-7), name
    for key in ("exp_avg", "exp_avg_sq"):
      assert torch.allclose(
        opt_act.state[p][key], opt_exp.state[exp_params[name]][key],
        rtol=1e-6, atol=1e-9,
      ), (name, key)


# ----------------------------------------------------------------------
# T4 — mask injection vs G_role consumption
# ----------------------------------------------------------------------
def test_step_mask_injection_rng():
  z0, rays, explicit_mask = make_batch()

  model, opt = make_model_and_opt()
  G_noise, G_role = make_generators()
  role_before = G_role.get_state()
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=explicit_mask, max_grad_norm=None,
  )
  assert not r.skipped
  assert torch.equal(G_role.get_state(), role_before), "injected mask consumed G_role"
  assert torch.equal(r.cond_mask.cpu(), explicit_mask.cpu())

  model, opt = make_model_and_opt()
  G_noise, G_role = make_generators()
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=None, max_grad_norm=None,
  )
  assert not r.skipped
  assert not torch.equal(G_role.get_state(), role_before), "sampled mask did not use G_role"
  assert r.cond_mask.shape == (B, V)
  assert (~r.cond_mask).sum(dim=1).min() >= 1


# ----------------------------------------------------------------------
# T5 / T5b / T5c — non-finite skip leaves ALL training state untouched
# ----------------------------------------------------------------------
def _assert_untouched(model, opt, ema, params0, adam0, ema0):
  assert equal_params(params0, snapshot(model)), "params changed on a skipped step"
  adam1 = adam_state(opt)
  assert adam0.keys() == adam1.keys()
  assert all(torch.equal(adam0[k], adam1[k]) for k in adam0), "Adam state changed"
  assert all(torch.equal(ema0[n], ema.shadow[n]) for n in ema0), "EMA updated on a skip"
  assert all(p.grad is None for p in model.parameters()), "grads not cleared"


def test_step_skip_nonfinite_loss():
  z0, rays, cond_mask = make_batch()
  model, opt = make_model_and_opt()
  prime(model, opt, z0, rays, cond_mask)
  ema = EMA(model, beta=0.9)
  params0, adam0 = snapshot(model), adam_state(opt)
  ema0 = {n: v.clone() for n, v in ema.shadow.items()}

  G_noise, G_role = make_generators()
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, ema=ema, max_grad_norm=None,
    epsilon=torch.full_like(z0, float("nan")),
    t=torch.full((B,), 0.5, device=DEVICE),
  )
  assert r.skipped
  _assert_untouched(model, opt, ema, params0, adam0, ema0)


def test_step_skip_nonfinite_grad():
  z0, rays, cond_mask = make_batch()
  model, opt = make_model_and_opt()
  prime(model, opt, z0, rays, cond_mask)
  ema = EMA(model, beta=0.9)
  params0, adam0 = snapshot(model), adam_state(opt)
  ema0 = {n: v.clone() for n, v in ema.shadow.items()}

  hook = model.a.register_hook(inject_inf)
  try:
    G_noise, G_role = make_generators()
    r = train_step(
      model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
      cond_mask=cond_mask, ema=ema, max_grad_norm=None,
    )
  finally:
    hook.remove()
  assert r.skipped
  _assert_untouched(model, opt, ema, params0, adam0, ema0)


def test_step_skip_nonfinite_loss_with_scaler():
  z0, rays, cond_mask = make_batch()
  model, opt = make_model_and_opt()
  prime(model, opt, z0, rays, cond_mask)
  ema = EMA(model, beta=0.9)
  scaler = torch.amp.GradScaler(DEVICE, init_scale=256.0)
  params0, adam0 = snapshot(model), adam_state(opt)
  ema0 = {n: v.clone() for n, v in ema.shadow.items()}

  G_noise, G_role = make_generators()
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, ema=ema, scaler=scaler, max_grad_norm=None,
    epsilon=torch.full_like(z0, float("nan")),
    t=torch.full((B,), 0.5, device=DEVICE),
  )
  assert r.skipped
  assert scaler.get_scale() < 256.0
  _assert_untouched(model, opt, ema, params0, adam0, ema0)


# ----------------------------------------------------------------------
# T6 — scaler overflow: skip, back off, recover
# ----------------------------------------------------------------------
def test_step_scaler_overflow_recovers():
  z0, rays, cond_mask = make_batch()
  model, opt = make_model_and_opt()
  ema = EMA(model, beta=0.9)
  scaler = torch.amp.GradScaler(DEVICE, init_scale=65536.0, growth_interval=1)

  params0 = snapshot(model)
  ema0 = {n: v.clone() for n, v in ema.shadow.items()}
  scales = [scaler.get_scale()]

  hook = model.a.register_hook(inject_inf)
  try:
    for i in range(3):
      G_noise, G_role = make_generators(100 + i)
      r = train_step(
        model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
        cond_mask=cond_mask, ema=ema, scaler=scaler, max_grad_norm=None,
      )
      assert r.skipped
      scales.append(scaler.get_scale())
  finally:
    hook.remove()

  assert scales == [65536.0, 32768.0, 16384.0, 8192.0]
  assert equal_params(params0, snapshot(model))
  assert all(torch.equal(ema0[n], ema.shadow[n]) for n in ema0)

  G_noise, G_role = make_generators(200)
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, ema=ema, scaler=scaler, max_grad_norm=None,
  )
  assert not r.skipped
  assert scaler.get_scale() > 8192.0            # growth_interval=1
  assert not equal_params(params0, snapshot(model))
  assert not all(torch.equal(ema0[n], ema.shadow[n]) for n in ema0)


# ----------------------------------------------------------------------
# T7 — clipping reports the PRE-clip norm; post-clip norm <= G
# ----------------------------------------------------------------------
def _true_preclip_norm(z0, rays, cond_mask, epsilon, t):
  model, opt = make_model_and_opt()
  G_noise, G_role = make_generators()
  train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, epsilon=epsilon, t=t, max_grad_norm=None,
  )
  return torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters())).item()  # type: ignore


@pytest.mark.parametrize("max_norm, expect_clip", [(1e-3, True), (1e6, False)])
def test_step_clip_reports_preclip_norm(max_norm, expect_clip):
  z0, rays, cond_mask = make_batch()
  g = torch.Generator(device=DEVICE).manual_seed(5)
  epsilon = torch.randn(z0.shape, generator=g, device=DEVICE)
  t = torch.tensor([0.4, 0.6], device=DEVICE)
  true_norm = _true_preclip_norm(z0, rays, cond_mask, epsilon, t)
  assert true_norm > 1e-3

  model, opt = make_model_and_opt()
  G_noise, G_role = make_generators()
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
    cond_mask=cond_mask, epsilon=epsilon, t=t, max_grad_norm=max_norm,
  )
  assert not r.skipped
  assert r.grad_norm == pytest.approx(true_norm, rel=1e-5)
  assert r.clipped is expect_clip

  post = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters())).item()  # type: ignore
  if expect_clip:
    assert post <= max_norm * (1 + 1e-4)
  else:
    assert post == pytest.approx(true_norm, rel=1e-5)


# ----------------------------------------------------------------------
# T8 — determinism over several steps (sampled masks, sampled eps/t)
# ----------------------------------------------------------------------
def test_step_deterministic():
  z0, rays, _ = make_batch()

  def run():
    model, opt = make_model_and_opt()
    G_noise, G_role = make_generators(1234)
    out = [
      train_step(
        model, opt, z0, rays, G_noise=G_noise, G_role=G_role,
        cond_mask=None, max_grad_norm=1.0,
      )
      for _ in range(5)
    ]
    return out, snapshot(model), G_noise.get_state(), G_role.get_state()

  ra, pa, na, ga = run()
  rb, pb, nb, gb = run()
  assert [r.loss for r in ra] == [r.loss for r in rb]
  assert all(torch.equal(x.cond_mask, y.cond_mask) for x, y in zip(ra, rb))
  assert equal_params(pa, pb)
  assert torch.equal(na, nb) and torch.equal(ga, gb)


# ----------------------------------------------------------------------
# T9 — returns carry no graph
# ----------------------------------------------------------------------
def test_step_returns_detached():
  z0, rays, cond_mask = make_batch()
  model, opt = make_model_and_opt()
  G_noise, G_role = make_generators()
  r = train_step(
    model, opt, z0, rays, G_noise=G_noise, G_role=G_role, cond_mask=cond_mask,
  )
  assert isinstance(r.loss, float) and isinstance(r.grad_norm, float)
  assert all(isinstance(v, float) for v in r.per_k_loss.values())
  assert r.cond_mask.grad_fn is None and not r.cond_mask.requires_grad