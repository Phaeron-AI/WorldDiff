"""Is the model sampling a distribution, or collapsing to the conditional mean?

Draws each scene D times with independent noise, then compares the draws
against each other and against the truth.

With D >= 3 this also measures the mode-collapse baseline directly instead of
inferring it, and cross-checks the variance decomposition two independent ways.

  python -m scripts.diagnose_variance --num-scenes 32 --draws 4

Defaults (--draws 2, --num-scenes 8) reproduce the earlier two-draw run
bit-for-bit: draw d uses seed VAL_SEED + d * SEED_STRIDE, so d=0 and d=1 are
the original A and B.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.loader import SceneLatentData
from src.eval.validate import K_ROUTINE, _generator, _substream, _validation_seed
from src.models.dit.model import MultiViewDiT
from src.models.flow import sample
from src.models.vae.vae_adapter import VAEAdapter
from src.training.state.checkpoint import CheckpointManager

VAL_SEED = 2026
SEED_STRIDE = 1_000_003


def to_uint8(img: torch.Tensor) -> np.ndarray:
  return (img.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()


def make_grid(rows, *, pad: int = 4, scale: int = 3) -> Image.Image:
  n_rows, n_cols = len(rows), len(rows[0])
  h, w = rows[0][0].shape[-2:]
  canvas = np.full(
    (n_rows * h + (n_rows + 1) * pad, n_cols * w + (n_cols + 1) * pad, 3),
    255, dtype=np.uint8,
  )
  for r, row in enumerate(rows):
    for c, img in enumerate(row):
      y, x = pad + r * (h + pad), pad + c * (w + pad)
      canvas[y:y + h, x:x + w] = to_uint8(img)
  if scale > 1:
    canvas = canvas.repeat(scale, axis=0).repeat(scale, axis=1)
  return Image.fromarray(canvas)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run-dir", type=Path, default=Path("runs/v2"))
  p.add_argument("--cache-dir", type=Path, default=Path("cache/v2"))
  p.add_argument("--out-dir", type=Path, default=Path("outputs/variance"))
  p.add_argument("--step", type=int, default=None)

  p.add_argument("--dim", type=int, default=256)
  p.add_argument("--num-heads", type=int, default=8)
  p.add_argument("--cond-dim", type=int, default=256)
  p.add_argument("--num-layers", type=int, default=8)
  p.add_argument("--patch-size", type=int, default=2)

  p.add_argument("--num-scenes", type=int, default=8)
  p.add_argument("--val-scenes", type=int, default=32)
  p.add_argument("--draws", type=int, default=2,
                 help="independent noise draws per scene (>=2; >=3 enables "
                      "the measured mean-predictor baseline)")
  p.add_argument("--grid-scenes", type=int, default=4,
                 help="how many scenes to write image grids for (0 = none)")
  p.add_argument("--json-out", type=Path, default=None,
                 help="also write the per-scene numbers here")
  p.add_argument("--steps", type=int, default=K_ROUTINE)
  p.add_argument("--device", type=str,
                 default="cuda" if torch.cuda.is_available() else "cpu")

  args = p.parse_args()
  if args.draws < 2:
    p.error("--draws must be at least 2")
  device = torch.device(args.device)

  dataset = SceneLatentData(args.cache_dir)
  manifest = json.loads((args.cache_dir / "manifest.json").read_text())
  n_train = len(dataset) - args.val_scenes
  val_set = [dataset[i] for i in range(n_train, len(dataset))][:args.num_scenes]

  _, latent_c, _, _ = val_set[0]["latents"].shape

  model = MultiViewDiT(
    in_channels=latent_c, dim=args.dim, num_heads=args.num_heads,
    cond_dim=args.cond_dim, num_layers=args.num_layers,
    patch_size=args.patch_size,
  ).to(device)

  manager = CheckpointManager(args.run_dir)
  if args.step is None:
    ckpt, ckpt_path = manager.load_latest()
  else:
    ckpt_path = manager.step_path(args.step).resolve()
    ckpt = manager.load(ckpt_path)

  model.load_state_dict(ckpt["model"])
  model.eval()
  step = int(ckpt["counters"]["optimizer_step"])

  vae = VAEAdapter(manifest["config"]["vae_model"],
                   dtype=torch.float32, device=device)
  args.out_dir.mkdir(parents=True, exist_ok=True)

  D = args.draws
  pairs = list(itertools.combinations(range(D), 2))

  print(f"checkpoint {ckpt_path.name} (step {step})   K={args.steps}   "
        f"D={D} draws x {len(val_set)} scenes = {D * len(val_set)} samples\n")
  print(f"{'scene':<14}{'sample-truth':>14}{'sample-sample':>15}"
        f"{'mean-pred':>11}{'R':>8}")

  rows: list[dict] = []
  pair_store: list[dict] = []

  with torch.no_grad():
    for idx, scene in enumerate(val_set):
      sid = scene["scene_id"]
      z0 = scene["latents"].to(device).unsqueeze(0)
      rays = scene["rays"].to(device).unsqueeze(0)
      cm = scene["cond_mask"].to(device, dtype=torch.bool).unsqueeze(0)
      k = int(cm.sum().item())
      t = ~cm

      truth = z0[t]
      draws, full = [], []
      for d in range(D):
        base = _validation_seed(VAL_SEED + d * SEED_STRIDE, sid, k)
        s = sample(model, z0, rays, cm, num_steps=args.steps,
                   rng=_generator(_substream(base, "c"), device))
        draws.append(s[t])
        full.append(s)

      # sample <-> truth, averaged over draws:        V_model + V_truth
      d_t = float(np.mean([((s - truth) ** 2).mean().item() for s in draws]))
      # sample <-> sample, per pair and pooled:       2 * V_model
      pair_d = {(i, j): float(((draws[i] - draws[j]) ** 2).mean().item())
                for i, j in pairs}
      d_ss = float(np.mean(list(pair_d.values())))
      # the D-draw mean vs truth:                     V_truth + V_model / D
      mean_pred = torch.stack(draws, dim=0).mean(dim=0)
      d_mp = float(((mean_pred - truth) ** 2).mean().item())

      r = d_ss / d_t if d_t > 0 else float("nan")
      print(f"{sid:<14}{d_t:>14.4f}{d_ss:>15.4f}{d_mp:>11.4f}{r:>8.3f}")
      rows.append({"scene_id": sid, "d_truth": d_t, "d_sample": d_ss,
                   "d_mean_pred": d_mp, "R": r,
                   "pairs": {f"{i}-{j}": v for (i, j), v in pair_d.items()}})
      pair_store.append(pair_d)

      if idx < args.grid_scenes:
        grid = make_grid([list(vae.decode(z0)[0])]
                         + [list(vae.decode(s)[0]) for s in full])
        grid.save(args.out_dir / f"{sid}_step{step:06d}_D{D}.png")

  n = len(rows)
  d_t = float(np.mean([r["d_truth"] for r in rows]))
  d_ss = float(np.mean([r["d_sample"] for r in rows]))
  d_mp = float(np.mean([r["d_mean_pred"] for r in rows]))
  per_scene_R = np.array([r["R"] for r in rows])
  R = d_ss / d_t

  v_model = d_ss / 2.0
  v_truth = d_t - v_model

  print(f"\n{'':<16}{'mean':>12}{'identity':>24}")
  print(f"{'sample-truth':<16}{d_t:>12.4f}{'V_model + V_truth':>24}")
  print(f"{'sample-sample':<16}{d_ss:>12.4f}{'2 * V_model':>24}")
  print(f"{f'mean of {D} draws':<16}{d_mp:>12.4f}{f'V_truth + V_model/{D}':>24}")

  # Bootstrap the pooled ratio over scenes (the per-scene ratios are noisy
  # on their own; this is a CI for the quantity actually reported).
  if n > 1:
    bs = np.random.default_rng(0).integers(0, n, size=(4000, n))
    a = np.array([r["d_sample"] for r in rows])
    b = np.array([r["d_truth"] for r in rows])
    boot = a[bs].mean(axis=1) / b[bs].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ci = f"95% CI [{lo:.3f}, {hi:.3f}] over {n} scenes"
  else:
    ci = "single scene - no interval"

  print(f"\nR = {R:.3f}   {ci}")
  print(f"    per-scene range {per_scene_R.min():.3f} - {per_scene_R.max():.3f}")
  print(f"V_model = {v_model:.4f}")
  print(f"V_truth = {v_truth:.4f}")

  # Real stability check: two disjoint halves of the draws, each giving its
  # own V_model. Needs 2 disjoint pairs, so D >= 4.
  if D >= 4:
    h = D // 2

    def half_v_model(idx: list[int]) -> float:
      pp = list(itertools.combinations(idx, 2))
      return float(np.mean([np.mean([pd[p] for p in pp])
                            for pd in pair_store])) / 2.0

    v1 = half_v_model(list(range(0, h)))
    v2 = half_v_model(list(range(h, 2 * h)))
    gap = abs(v1 - v2)
    rel = gap / max(v1, v2, 1e-12)
    verdict = "stable" if rel < 0.15 else "UNSTABLE - V_model is outlier-driven"
    print(f"\nhalf-split V_model: {v1:.4f} vs {v2:.4f}  "
          f"(disjoint draws, gap {rel * 100:.1f}%) -> {verdict}")

  if d_mp < d_t * 0.995:
    print(f"\nAveraging the {D} draws scores {d_mp:.4f} against {d_t:.4f} for a "
          f"single draw -- {(1 - d_mp / d_t) * 100:.0f}% 'better' by the gate.")
    print("That is the whole problem in one line: the metric pays for blur.")
  else:
    print(f"\nAveraging the {D} draws scores {d_mp:.4f}, no better than a single "
          f"draw at {d_t:.4f}: there is nothing to average away.")

  print("\nR ~ 0.0  -> collapsed to the conditional mean; a modelling problem.")
  print("R ~ 1.0  -> model spread matches the data's; the metric is the ceiling.")
  print("R  > 1.0 -> over-dispersed. R is NOT capped at 1: it tends to 2 as the")
  print("            data's own variance goes to 0, so R near 2 means the targets")
  print("            were nearly deterministic and the model is adding noise.")
  if args.grid_scenes:
    print(f"\ngrids: row 1 truth, rows 2-{D + 1} the {D} independent draws")

  if args.json_out:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({
      "checkpoint": ckpt_path.name, "step": step, "K": args.steps,
      "draws": D, "num_scenes": n,
      "pooled": {"d_truth": d_t, "d_sample": d_ss, "d_mean_pred": d_mp,
                 "R": R, "V_model": v_model, "V_truth": v_truth},
      "scenes": rows,
    }, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
  main()