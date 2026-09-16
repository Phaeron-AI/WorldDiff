from __future__ import annotations

# Native Import(s)
import math

# Third Party Import(s)
import torch
from torch import nn, Tensor

class TimeStepEmbedding(nn.Module):
  def __init__(self, cond_dim: int, frequency_embedding_size: int, output_size: int | None = None) -> None:
    super().__init__()

    self.cond_dim = cond_dim
    self.frequency_embedding_size = frequency_embedding_size
    self.output_size = output_size if output_size else cond_dim

    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, cond_dim),
      nn.SiLU(),
      nn.Linear(cond_dim, cond_dim)
    )

  def timestep_embedding(self, t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)

    args = t.float()[:, None] * frequencies[None, :]

    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding

  def forward(self, t: Tensor) -> Tensor:
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

    return self.mlp(t_freq)


def patchify(x: Tensor, patch_size: int):
  B, V, C, h, w = x.shape # [B, V, C, h, w]
  p = patch_size
  
  assert h % p == 0, "h must be divisible by patch_size"
  assert w % p == 0, "w must be divisible by patch_size"

  H_p = h // p
  W_p = w // p
  N = H_p * W_p
  
  x = x.reshape(B, V, C, H_p, p, W_p, p)
  x = x.permute(0, 1, 3, 5, 4, 6, 2)
  x = x.reshape(B, V, N, p * p * C)

  return x

def unpatchify(tokens, patch_size: int, height: int, width: int, out_channels: int):
  B, V, N, patch_dim = tokens.shape
  p = patch_size
  C = out_channels

  assert height % p == 0, "height must be divisible by patch_size"
  assert width % p == 0, "width must be divisible by patch_size"

  H_p = height // p
  W_p = width // p

  expected_N = H_p * W_p
  expected_patch_dim = p * p * C

  assert N == expected_N, (
    f"Expected {expected_N} patches, got {N}"
  )

  assert patch_dim == expected_patch_dim, (
    f"Expected patch dimension {expected_patch_dim}, got {patch_dim}"
  )

  tokens = tokens.reshape(B, V, H_p, W_p, p, p, C)
  tokens = tokens.permute(0, 1, 6, 2, 4, 3, 5)

  x = tokens.reshape(B, V, C, height, width)

  return x


class PatchEmbedding(nn.Module):
  def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
    super().__init__()

    self.in_channels = in_channels
    self.embed_dim = embed_dim
    self.patch_size = patch_size

    patch_dim = patch_size * patch_size * in_channels  # p^2*C

    self.proj = nn.Linear(
      patch_dim,
      embed_dim
    )

  def forward(self, x: Tensor) -> Tensor:
    _, _, C, h, w = x.shape

    assert C == self.in_channels, (
      f"Expected {self.in_channels} input channels, got {C}"
    )

    assert h % self.patch_size == 0, (
      "h must be divisible by patch_size"
    )

    assert w % self.patch_size == 0, (
      "w must be divisible by patch_size"
    )

    tokens = patchify(x, self.patch_size)
    tokens = self.proj(tokens)

    return tokens


def get_2d_sincos_pos_embedding(embed_dim: int, grid_h: int, grid_w: int, patch_size: int, emb_const: float = 10_000.0):
  half = embed_dim // 2
  
  assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos."

  y_coords = torch.arange(grid_h, dtype=torch.float32)
  x_coords = torch.arange(grid_w, dtype=torch.float32)
  y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
  y_flat = y_grid.flatten()  # [N]
  x_flat = x_grid.flatten()  # [N]

  omega = 1.0 / (emb_const ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

  emb_y = y_flat.unsqueeze(1) * omega.unsqueeze(0)
  emb_y = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=1)

  emb_x = x_flat.unsqueeze(1) * omega.unsqueeze(0)
  emb_x = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=1)

  pos_embed = torch.cat([emb_y, emb_x], dim=1)

  return pos_embed

class ViewConditionEmbedder(nn.Module):
  def __init__(self, embed_dim: int) -> None:
    super().__init__()

    self.embedding = nn.Embedding(
      num_embeddings=2,
      embedding_dim=embed_dim
    )

  def forward(self, cond_mask: Tensor) -> Tensor:
    cond_mask = cond_mask.long()

    assert cond_mask.ndim == 2, (
      "cond_mask must have shape [B, V]"
    )

    assert torch.all(
      (cond_mask == 0) | (cond_mask == 1)
    ), "cond_mask must contain only 0 or 1"

    return self.embedding(cond_mask)