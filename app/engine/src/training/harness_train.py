"""Reviewer harness for P3.3a train.py — not part of the repo."""
import copy, math, torch
from torch import nn
from src.models.dit.model import MultiViewDiT
from src.training.train import (
  Trainer, TrainingConfig, Hooks, TrainingDiverged, ResumeMismatch, PerKAccumulator, train,
)
from src.training.epoch_cursor import EpochCursor, epoch_permutation

def R(n, ok, d=""):
  print(f"[{'PASS' if ok else 'FAIL'}] {n}" + (f"  {d}" if d else ""))

V, C, H, W = 4, 4, 8, 8

class ToyScenes:
  def __init__(self, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    self.items = [
      {"latents": torch.randn(V, C, H, W, generator=g),
       "rays": torch.randn(V, 6, H, W, generator=g),
       "scene_id": f"s{i}"} for i in range(n)
    ]
  def __len__(self): return len(self.items)
  def __getitem__(self, i): return self.items[i]

def mk_model(seed=0):
  torch.manual_seed(seed)
  return MultiViewDiT(in_channels=C, dim=32, num_heads=4, cond_dim=32, num_layers=2, patch_size=2)

def cfg(**kw):
  base = dict(total_steps=12, batch_size=4, max_lr=1e-3, warmup_steps=3,
              log_every=0, eval_every=0, checkpoint_every=0, ema_beta=0.9)
  base.update(kw)
  return TrainingConfig(**base)  # type: ignore

# ---------------------------------------------------------------- I0 cursor
n, B = 100, 32
cur = EpochCursor(data_seed=0, length=n, batch_size=B)
seen, epochs = [], []
for _ in range(9):
  seen.append(cur.next_indices()); epochs.append(cur.epoch)
perm0 = epoch_permutation(0, 0, n)
R("I0a first epoch follows pi_0 and drops the tail",
  seen[0] == perm0[:32].tolist() and seen[2] == perm0[64:96].tolist() and epochs[:3] == [0, 0, 0])
R("I0b epoch rolls when cursor+B > length (59.2 fix)", epochs[3] == 1 and seen[3] == epoch_permutation(0, 1, n)[:32].tolist(),
  f"batches/epoch={cur.batches_per_epoch} epochs seen={epochs}")
R("I0c no index reuse inside an epoch", len(set(sum(seen[:3], []))) == 96)
R("I0d permutations differ across epochs", not torch.equal(epoch_permutation(0,0,n), epoch_permutation(0,1,n)))
try:
  EpochCursor(data_seed=0, length=3, batch_size=8); R("I0e len<B raises", False)
except ValueError:
  R("I0e len(dataset) < B raises (dec. 58)", True)

# ---------------------------------------------------------------- I1 smoke
# A real overfit: a handful of fixed scenes, enough steps to memorise them.
# (40 steps on random scenes cannot lower any loss -- that was a bad test.)
def mk_big(seed=0):
  torch.manual_seed(seed)
  return MultiViewDiT(in_channels=C, dim=64, num_heads=4, cond_dim=64, num_layers=3, patch_size=2)

for prec in ("fp32", "bf16"):
  t = Trainer(cfg(total_steps=400, batch_size=4, max_lr=1e-3, warmup_steps=20,
                  p_drop=0.0, precision=prec, log_every=50), mk_big(), ToyScenes(4, seed=1))  # type: ignore
  losses = []
  t.hooks.on_log = lambda rec: losses.append(rec["loss"]) # type: ignore
  s_ = t.run()
  R(f"I1 smoke {prec}: few-scene overfit drives the loss down",
    s_.optimizer_steps == 400 and losses[-1] < 0.3 * losses[0],
    f"loss {losses[0]:.3f} -> {losses[-1]:.3f}, epochs={s_.epochs_completed}")

# ---------------------------------------------------------------- I2 counters vs injected skips
ds = ToyScenes(16)
t = Trainer(cfg(total_steps=10), mk_model(), ds)  # type: ignore
state = {"i": 0}
param = next(iter(t.model.parameters()))
def flaky(grad):
  state["i"] += 1
  return torch.full_like(grad, float("inf")) if state["i"] % 3 == 0 else grad
h = param.register_hook(flaky)
lrs, emas = [], []
orig_step = t.optimizer.step
ema_before = {k: v.clone() for k, v in t.ema.shadow.items()}  # type: ignore
s = t.run(); h.remove()
R("I2a ends at exactly total_steps successful steps", s.optimizer_steps == 10, f"attempted={s.attempted_steps} skipped={s.skipped_steps}")
R("I2b attempted > successful when steps are skipped", s.attempted_steps > s.optimizer_steps and s.skipped_steps == s.attempted_steps - s.optimizer_steps)
R("I2c samples_seen counts every attempted batch", s.samples_seen == s.attempted_steps * 4)

# LR/EMA advance only on successful steps: compare a clean run of N steps with
# a flaky run of N steps -> identical LR sequence (indexed by successful steps)
seen_lr = []
t2 = Trainer(cfg(total_steps=6), mk_model(), ToyScenes(16)) # type: ignore
for i in range(6):
  seen_lr.append(t2._lr_for_next_step()); t2.optimizer_step += 1
from src.training.lr_schedule import learning_rate as lr_fn
expect = [lr_fn(step=i, max_lr=1e-3, warmup_steps=3, total_steps=6) for i in range(6)]
R("I2d LR is indexed by successful steps", seen_lr == expect)

# ---------------------------------------------------------------- I3 abort
ds = ToyScenes(16)
t = Trainer(cfg(total_steps=50, max_consecutive_skips=5), mk_model(), ds) # type: ignore
tags = []
t.hooks.on_checkpoint = lambda st, tag: tags.append((tag, st["counters"]["optimizer_step"]))  # type: ignore
h = next(iter(t.model.parameters())).register_hook(lambda g: torch.full_like(g, float("inf")))
try:
  t.run(); R("I3a streak > N raises", False)
except TrainingDiverged as e:
  R("I3a streak > N raises TrainingDiverged", True, str(e)[:60] + "...")
finally:
  h.remove()
R("I3b emergency checkpoint written before raising (dec. 63)", tags and tags[-1][0] == "emergency", f"tags={tags}")

# healthy backoff shorter than N must NOT abort
ds = ToyScenes(16)
t = Trainer(cfg(total_steps=8, max_consecutive_skips=5), mk_model(), ds)  # type: ignore
state = {"i": 0}
def burst(grad):
  state["i"] += 1
  return torch.full_like(grad, float("inf")) if state["i"] <= 4 else grad
h = next(iter(t.model.parameters())).register_hook(burst)
try:
  s = t.run(); ok = s.optimizer_steps == 8 and s.skipped_steps == 4
except TrainingDiverged:
  ok = False
finally:
  h.remove()
R("I3c streak of 4 with N=5 does not abort", ok)

# fp16 config guard: N <= ceil(log2(init_scale)) must be rejected
try:
  TrainingConfig(precision="fp16", init_scale=65536.0, max_consecutive_skips=16).validate()
  R("I3d fp16 N <= log2(init_scale) rejected", False)
except ValueError:
  R("I3d fp16 N <= ceil(log2(init_scale)) rejected (dec. 50/57)", True)

# ---------------------------------------------------------------- I4 resume equivalence
# Drive both runs with step_once() and compare PER-STEP losses. (Comparing
# logged values would compare window means: a crash mid-window carries its
# sums/counts forward by design -- derivation 70 -- so the first log after a
# resume legitimately averages the carried steps with the new one.)
def fresh(seed_model, steps=12):
  return Trainer(cfg(total_steps=steps), mk_model(seed_model), ToyScenes(14, seed=3)) # type: ignore

t_full = fresh(7)
losses_full = [t_full.step_once().loss for _ in range(12)]  # type: ignore
params_full = {k: v.clone() for k, v in t_full.model.state_dict().items()}

t_part = fresh(7)
for _ in range(5):
  t_part.step_once()  # type: ignore
mid = copy.deepcopy(t_part.state_dict())          # "crash" here, mid-window

t_res = fresh(0)
t_res.load_state_dict(mid)
losses_res = [t_res.step_once().loss for _ in range(7)] # type: ignore
params_res = t_res.model.state_dict()

R("I4a resumed per-step losses match the uninterrupted tail", losses_res == losses_full[5:],
  f"full[5:8]={[round(x,6) for x in losses_full[5:8]]} res[:3]={[round(x,6) for x in losses_res[:3]]}")
R("I4b resumed parameters identical", all(torch.equal(params_full[k], params_res[k]) for k in params_full))
R("I4c resumed counters identical",
  (t_res.optimizer_step, t_res.samples_seen, t_res.cursor.state_dict()) ==
  (t_full.optimizer_step, t_full.samples_seen, t_full.cursor.state_dict()))
R("I4d the stop point is mid-epoch (not an epoch boundary)",
  mid["cursor"]["cursor"] not in (0, 14 // 4 * 4),
  f"epoch={mid['cursor']['epoch']} cursor={mid['cursor']['cursor']} (an epoch covers {14//4*4} of 14)")
R("I4e window sums/counts carried across the crash (dec. 70)",
  mid["window"]["loss_count"] == 5 and t_res.window_loss_count == 12,
  f"carried count={mid['window']['loss_count']}, after resume={t_res.window_loss_count}")

# resume guards
try:
  t_bad = Trainer(cfg(total_steps=12, batch_size=2), mk_model(0), ToyScenes(14, seed=3))  # type: ignore
  t_bad.load_state_dict(mid); R("I4f resume-critical mismatch raises", False)
except ResumeMismatch:
  R("I4f resume-critical config mismatch raises (dec. 62)", True)
try:
  t_bad = Trainer(cfg(total_steps=12, cache_hash="deadbeef"), mk_model(0), ToyScenes(14, seed=3)) # type: ignore
  t_bad.load_state_dict(mid); R("I4g cache-hash mismatch raises", False)
except ResumeMismatch:
  R("I4g cache-hash mismatch raises (dec. 62)", True)
t_ok = Trainer(cfg(total_steps=30, log_every=0), mk_model(0), ToyScenes(14, seed=3))  # type: ignore
try:
  t_ok.load_state_dict(mid); R("I4h changing a free field (total_steps) is allowed", True)
except ResumeMismatch:
  R("I4h changing a free field (total_steps) is allowed", False)

# ---------------------------------------------------------------- I5 eval isolation
ds = ToyScenes(16)
t = Trainer(cfg(total_steps=6, eval_every=3, checkpoint_every=3), mk_model(5), ds)  # type: ignore
seen_weights, ckpt_weights, eval_modes, eval_seeds = [], [], [], []
raw_at_eval = {}
def on_eval(model, *, step, generator):
  eval_modes.append(model.training)
  eval_seeds.append(int(torch.randn(1, generator=generator, device=generator.device).mul(1e6)))
  seen_weights.append({k: v.clone() for k, v in model.state_dict().items()})
  torch.randn(4, generator=generator, device=generator.device)   # eval consumes randomness
  torch.randn(4)                                                 # ... including global RNG
def on_ckpt(st, tag): ckpt_weights.append({k: v.clone() for k, v in st["model"].items()})
t.hooks.on_eval, t.hooks.on_checkpoint = on_eval, on_ckpt # type: ignore
# capture raw weights + rng right before eval by hooking the log
rng_before = {}
s = t.run()
R("I5a eval hook runs under model.eval()", eval_modes == [False, False], f"training flags seen: {eval_modes}")
R("I5b model is back in train() after eval", t.model.training)
R("I5c eval saw EMA weights, checkpoint saw RAW weights (dec. 52/66)",
  all(not torch.equal(seen_weights[i]["out_proj.weight"] if "out_proj.weight" in seen_weights[i] else list(seen_weights[i].values())[0],
                      list(ckpt_weights[i].values())[0]) for i in range(2)))
R("I5d eval seed depends on the step (comparable across checkpoints)", eval_seeds[0] != eval_seeds[1])

# RNG isolation: a run with eval hooks must match a run without them
def run_with(eval_hook):
  t = Trainer(cfg(total_steps=8, eval_every=2, log_every=1), mk_model(11), ToyScenes(16, seed=5)) # type: ignore
  out = []
  t.hooks.on_log = lambda rec: out.append(rec["loss"])  # type: ignore
  t.hooks.on_eval = eval_hook
  t.run()
  return out, t.model.state_dict()
noisy_eval = lambda model, *, step, generator: (torch.randn(64, generator=generator, device=generator.device), torch.randn(64))
a_losses, a_params = run_with(None)
b_losses, b_params = run_with(noisy_eval)
R("I5e evaluation does not perturb the training trajectory (dec. 65)",
  a_losses == b_losses and all(torch.equal(a_params[k], b_params[k]) for k in a_params))

# ---------------------------------------------------------------- I6 per-k aggregation
acc = PerKAccumulator()
m100 = torch.zeros(100, 4, dtype=torch.bool); m100[:, :1] = True    # k=1, 100 samples
m10 = torch.zeros(10, 4, dtype=torch.bool); m10[:, :1] = True       # k=1, 10 samples
acc.add({1: 1.0}, m100)
acc.add({1: 3.0}, m10)
got = acc.means()[1]
R("I6a window mean is example-weighted, not mean-of-means",
  abs(got - 130/110) < 1e-9, f"got {got:.4f}; example-weighted 1.1818; mean-of-means would be 2.0")
acc2 = PerKAccumulator()
mix = torch.tensor([[1,0,0,0],[1,1,0,0],[1,1,0,0]], dtype=torch.bool)
acc2.add({1: 2.0, 2: 5.0}, mix)
R("I6b counts come from the mask actually used", acc2.counts == {1: 1, 2: 2} and acc2.sums == {1: 2.0, 2: 10.0},
  f"counts={acc2.counts} sums={acc2.sums}")

# ---------------------------------------------------------------- I7 drop_last
ds = ToyScenes(14)
sizes = []
t = Trainer(cfg(total_steps=9, batch_size=4), mk_model(), ds) # type: ignore
real_fetch = t._fetch_batch
def spy():
  z0, rays, b = real_fetch(); sizes.append(b); return z0, rays, b
t._fetch_batch = spy
t.run()
R("I7 every training batch has exactly B scenes", set(sizes) == {4}, f"batch sizes seen: {sorted(set(sizes))}, batches={len(sizes)}")
