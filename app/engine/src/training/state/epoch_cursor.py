from __future__ import annotations

# Native Import(s)
import hashlib
from dataclasses import dataclass

# Third Party Import(s)
import torch
from torch import Tensor

def _permutation_seed(data_seed: int, epoch: int, length: int) -> int:
  blob = f"{int(data_seed)}:{int(epoch)}:{int(length)}".encode()
  return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") >> 1

def epoch_permutation(data_seed: int, epoch: int, length: int) -> Tensor:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(_permutation_seed(data_seed, epoch, length))
  return torch.randperm(length, generator=generator)

@dataclass
class EpochCursor:
  data_seed: int
  length: int
  batch_size: int
  epoch: int = 0
  cursor: int = 0

  def __post_init__(self) -> None:
    if self.length < self.batch_size:
      raise ValueError(
        f"dataset has {self.length} scenes but batch_size is "
        f"{self.batch_size}; with drop_last=True this yields zero batches "
        "(derivation 58: raise rather than clamp B or drop the estimator)"
      )
    if self.batch_size < 1:
      raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
    if not 0 <= self.cursor <= self.length:
      raise ValueError(f"cursor {self.cursor} outside [0, {self.length}]")
    self._permutation = epoch_permutation(self.data_seed, self.epoch, self.length)
    self._roll_if_exhausted()

  @property
  def batches_per_epoch(self) -> int:
    return self.length // self.batch_size
 
  def _roll_if_exhausted(self) -> None:
    # NOTE: '+ batch_size >' , not '=='.
    while self.cursor + self.batch_size > self.length:
      self.epoch += 1
      self.cursor = 0
      self._permutation = epoch_permutation(self.data_seed, self.epoch, self.length)
 
  def next_indices(self) -> list[int]:
    """Return the next batch of dataset indices and advance the cursor."""
    self._roll_if_exhausted()
    indices = self._permutation[self.cursor : self.cursor + self.batch_size]
    self.cursor += self.batch_size
    return [int(i) for i in indices]
 
  def state_dict(self) -> dict:
    return {
      "data_seed": self.data_seed,
      "length": self.length,
      "batch_size": self.batch_size,
      "epoch": self.epoch,
      "cursor": self.cursor,
    }

  @classmethod
  def from_state_dict(cls, state: dict) -> "EpochCursor":
    return cls(
      data_seed=state["data_seed"],
      length=state["length"],
      batch_size=state["batch_size"],
      epoch=state["epoch"],
      cursor=state["cursor"],
    )