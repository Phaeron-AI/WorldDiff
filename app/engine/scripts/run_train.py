"""Train WorldDiff on a built latent cache.

  python -m scripts.run_train --cache-dir cache/dev --total-steps 2000 --no-ema
"""

from __future__ import annotations

# Native Import(s)
import argparse
import json
from pathlib import Path

# Third Party Import(s)
import torch

# Local Import(s)
from src.data.loader import SceneLatentData
from src.eval.validate import validate_routine
from src.models.dit.model import MultiViewDiT
from src.models.vae.vae_adapter import VAEAdapter
from src.training.state.checkpoint import CheckpointManager
from src.training.train import Hooks, TrainingConfig, train

# Fixed for the whole run. Derivation 75: every checkpoint must see identical
# scenes, masks and noise, or the curve mixes model improvement with sampling
# noise. Deliberately NOT derived from the step, even though the eval hook is
# handed a step-dependent generator.
VAL_SEED = 2026


def make_log_hook():
  def _log(record) -> None:
    print(
      f"step {int(record['optimizer_step']):>6}"
      f"  loss {float(record['loss']):8.4f}"
      f"  lr {float(record['lr']):.2e}"
      f"  epoch {int(record['epoch']):>3}"
    )

  return _log


def make_eval_hook(val_set, *, vae, device, num_input):
  def _eval(model, *, step: int, generator: torch.Generator):
    metrics = validate_routine(
      model,
      val_set,
      vae=vae,
      val_seed=VAL_SEED,
      device=device,
      num_input=num_input,
    )
    print(
      f"           EVAL  latent_mse {metrics['val/latent_mse']:.4f}"
      f"  mse {metrics['val/mse']:.5f}"
      f"  psnr {metrics['val/psnr']:.2f} dB"
    )
    return metrics

  return _eval


def make_checkpoint_hook(manager: CheckpointManager):
  def _checkpoint(state, *, tag: str) -> None:
    # The step is nested under "counters" -- reading it from the top level
    # silently yields 0 and every checkpoint overwrites step_000000.pt.
    step = int(state["counters"]["optimizer_step"])

    if tag == "emergency":
      path = manager.save_emergency(state=state)
    else:
      path = manager.save(step=step, state=state)

    print(f"           CKPT  {tag}  -> {path.name}")

  return _checkpoint


def main() -> None:
  p = argparse.ArgumentParser(description="WorldDiff training run")

  p.add_argument("--cache-dir", type=Path, default=Path("cache/dev"))
  p.add_argument("--run-dir", type=Path, default=Path("runs/dev"))

  # Model
  p.add_argument("--dim", type=int, default=128)
  p.add_argument("--num-heads", type=int, default=4)
  p.add_argument("--cond-dim", type=int, default=128)
  p.add_argument("--num-layers", type=int, default=4)
  p.add_argument("--patch-size", type=int, default=2)

  # Budget
  p.add_argument("--total-steps", type=int, default=2000)
  p.add_argument("--batch-size", type=int, default=8)
  p.add_argument("--max-lr", type=float, default=1e-4)
  p.add_argument("--warmup-steps", type=int, default=100)
  p.add_argument("--p-drop", type=float, default=0.1)
  p.add_argument("--precision", type=str, default="fp32",
                 choices=["fp32", "bf16", "fp16"])

  # EMA -- derivation 45. The horizon 1/(1-beta) must sit well inside
  # total_steps, or the averaged weights never stop being the init.
  # beta=0.9999 has a 10,000-step horizon: wrong for a 2,000-step run.
  p.add_argument("--ema-beta", type=float, default=0.998)
  p.add_argument("--no-ema", action="store_true")

  # Cadences
  p.add_argument("--log-every", type=int, default=50)
  p.add_argument("--eval-every", type=int, default=250)
  p.add_argument("--checkpoint-every", type=int, default=500)
  p.add_argument("--val-scenes", type=int, default=32)

  p.add_argument("--init-seed", type=int, default=0)
  p.add_argument("--device", type=str,
                 default="cuda" if torch.cuda.is_available() else "cpu")

  args = p.parse_args()

  device = torch.device(args.device)
  torch.manual_seed(args.init_seed)

  # ------------------------------------------------------------------
  # Data -- a held-out slice, not a principled split yet.
  # ------------------------------------------------------------------
  dataset = SceneLatentData(args.cache_dir)
  manifest = json.loads((args.cache_dir / "manifest.json").read_text())
  num_input = manifest["config"]["num_input"]

  if args.val_scenes >= len(dataset):
    raise SystemExit("val_scenes must be smaller than the dataset")

  n_train = len(dataset) - args.val_scenes

  # Materialised: __getitem__ does a torch.load per call, so this is both
  # simpler and faster at this size. Revisit when the cache outgrows RAM.
  train_set = [dataset[i] for i in range(n_train)]
  val_set = [dataset[i] for i in range(n_train, len(dataset))]

  probe = train_set[0]
  num_views, latent_c, lat_h, lat_w = probe["latents"].shape

  # ------------------------------------------------------------------
  # Model
  # ------------------------------------------------------------------
  model = MultiViewDiT(
    in_channels=latent_c,
    dim=args.dim,
    num_heads=args.num_heads,
    cond_dim=args.cond_dim,
    num_layers=args.num_layers,
    patch_size=args.patch_size,
  )
  n_params = sum(q.numel() for q in model.parameters())

  cfg = TrainingConfig(
    total_steps=args.total_steps,
    batch_size=args.batch_size,
    max_lr=args.max_lr,
    warmup_steps=args.warmup_steps,
    p_drop=args.p_drop,
    precision=args.precision,
    ema_beta=args.ema_beta,
    ema_enabled=not args.no_ema,
    log_every=args.log_every,
    eval_every=args.eval_every,
    checkpoint_every=args.checkpoint_every,
    cache_hash=manifest["hash"],
  )

  hooks = Hooks(on_log=make_log_hook())

  vae = None
  if args.eval_every > 0:
    # Same device as training: decode() casts dtype but never moves device.
    vae = VAEAdapter(
      manifest["config"]["vae_model"],
      dtype=torch.float32,
      device=device,
    )
    hooks.on_eval = make_eval_hook(
      val_set, vae=vae, device=device, num_input=num_input
    )

  if args.checkpoint_every > 0:
    hooks.on_checkpoint = make_checkpoint_hook(CheckpointManager(args.run_dir))

  ema_desc = "off" if args.no_ema else (
    f"beta {args.ema_beta}  (horizon {1 / (1 - args.ema_beta):.0f} steps, "
    f"leftover init {args.ema_beta ** args.total_steps * 100:.1f}%)"
  )

  print(f"device       {device}")
  print(f"train/val    {len(train_set)} / {len(val_set)} scenes   V={num_views}")
  print(f"latents      {latent_c}x{lat_h}x{lat_w}   num_input={num_input}")
  print(f"params       {n_params:,}")
  print(f"steps        {cfg.total_steps}  batch {cfg.batch_size}  "
        f"lr {cfg.max_lr:.1e}  {cfg.precision}")
  print(f"ema          {ema_desc}")
  print(f"cache hash   {cfg.cache_hash}")
  print(f"run dir      {args.run_dir}")
  print()

  summary = train(cfg, model, train_set, device=device, hooks=hooks)  # type: ignore

  print()
  print(f"optimizer steps  {summary.optimizer_steps}")
  print(f"attempted        {summary.attempted_steps}  (skipped {summary.skipped_steps})")
  print(f"epochs           {summary.epochs_completed}")
  print(f"final loss       {summary.final_loss:.4f}")
  print(f"wall time        {summary.wall_time_s:.1f}s")
  if summary.aborted:
    print("ABORTED")


if __name__ == "__main__":
  main()