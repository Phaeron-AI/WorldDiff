from __future__ import annotations

# Native Import(s)
import json
from pathlib import Path

# Third Party Import(s)
import torch
from torch import Tensor, Generator
from torch.utils.data import Dataset, DataLoader

# Local Import(s)
from src.data.latent_cache import CacheConfig
from src.data.rays import RayEncoder


class SceneLatentData(Dataset):
  def __init__(
    self,
    cache_dir: str | Path,
    expected_cfg: CacheConfig | None = None,
  ) -> None:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())

    if expected_cfg is not None and manifest["hash"] != expected_cfg.hash():
      raise RuntimeError("stale/mismatched latent cache -- rebuild")

    self._files = [cache_dir / f for f in manifest["files"]]
    self._ray_enc = RayEncoder(normalize=manifest["config"]["ray_normalize"])

  def __len__(self) -> int:
    return len(self._files)

  def __getitem__(self, i: int) -> dict[str, object]:
    rec = torch.load(self._files[i], map_location="cpu")
    z = rec["latents"].float()
    h, w = z.shape[-2], z.shape[-1]
    rays = self._ray_enc(rec["intrinsics"], rec["c2w"], h, w)   # [V,6,h,w]
    return {
      "latents": z, "rays": rays, "c2w": rec["c2w"],
      "intrinsics": rec["intrinsics"], "cond_mask": rec["cond_mask"],
      "scene_id": rec["scene_id"],
    }


def collate_scenes(batch: list[dict]) -> dict[str, object]:
  tensor_keys = ("latents", "rays", "c2w", "intrinsics", "cond_mask")
  out = {k: torch.stack([b[k] for b in batch], dim=0) for k in tensor_keys}  # -> [B,V,...]
  out["scene_id"] = [b["scene_id"] for b in batch]  # type: ignore[assignment]
  return out  # type: ignore


def make_loader(
  dataset: SceneLatentData,
  *,
  batch_size: int,
  shuffle: bool = True,
  num_workers: int = 0,
  seed: int = 0,
  pin_memory: bool | None = None,
) -> DataLoader:
  if pin_memory is None:
    pin_memory = torch.cuda.is_available()

  g = Generator().manual_seed(seed)

  def _worker_init(wid: int) -> None:
    torch.manual_seed(seed + wid)

  return DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle,
    num_workers=num_workers, collate_fn=collate_scenes,
    pin_memory=pin_memory, generator=g,
    worker_init_fn=_worker_init, drop_last=False,
  )