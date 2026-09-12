from __future__ import annotations

# Local Import(s)
import math

# Third Party Import(s)
import torch
from torch import Tensor, nn

def scaled_dot_product_attention(
  q: Tensor, k: Tensor, v: Tensor
) -> Tensor:
  """
    Scaled Dot-Product Attention.

    Expected shapes:
      q: [B, H, Nq, Dh]
      k: [B, H, Nk, Dh]
      v: [B, H, Nk, Dh]

    Returns:
      [B, H, Nq, Dh]

    Attention:
      softmax(QK^T / sqrt(Dh)) V
  """

  if q.ndim != 4:
    raise ValueError(
      f"q must have shape [B,H,Nq,Dh], got {tuple(q.shape)}"
    )

  if k.ndim != 4:
    raise ValueError(
      f"k must have shape [B,H,Nk,Dh], got {tuple(k.shape)}"
    )

  if v.ndim != 4:
    raise ValueError(
      f"v must have shape [B,H,Nk,Dh], got {tuple(v.shape)}"
    )

  if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
    raise ValueError("q, k, and v must have the same batch size")

  if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
    raise ValueError("q, k, and v must have the same number of heads")

  if q.shape[-1] != k.shape[-1]:
    raise ValueError(
      "q and k must have the same head dimension"
    )

  if k.shape[-2] != v.shape[-2]:
    raise ValueError(
      "k and v must have the same sequence length"
    )

  scale_factor =  1.0 / math.sqrt(q.shape[-1])

  scores = torch.matmul(q, q.transpose(-1, -2)) * scale_factor
  weights = torch.softmax(scores, dim=-1)

  return torch.matmul(weights, v)

class MultiHeadAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, *, bias: bool = False) -> None:
    super().__init__()

    if dim <= 0:
      raise ValueError(f"dim must be positive, got {dim}")

    if num_heads <= 0:
      raise ValueError(
        f"num_heads must be positive, got {num_heads}"
      )

    if dim % num_heads != 0:
      raise ValueError(
        f"dim ({dim}) must be divisible by "
        f"num_heads ({num_heads})"
      )

    self.dim = dim
    self.num_heads = num_heads
 
    self.head_dim = dim // num_heads

    self.qkv = nn.Linear(
      dim,
      3 * dim,
      bias=bias
    )

    self.proj = nn.Linear(
      dim,
      dim,
      bias=bias
    )

  def forward(self, x: Tensor) -> Tensor:
    if x.ndim != 3:
      raise ValueError(
        f"x must have shape [B,N,D], got {tuple(x.shape)}"
      )

    B, N, D = x.shape  # [B, N, D]

    if D != self.dim:
      raise ValueError(
        f"Expected embedding dimension {self.dim}, got {D}"
      )
    
    qkv = self.qkv(x)

    qkv = qkv.reshape(
      B, N, 3, self.num_heads, self.head_dim
    )

    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv.unbind(dim=0)

    out = scaled_dot_product_attention(q, k, v)

    out = out.transpose(1, 2)

    out = out.reshape(B, N, D)

    return self.proj(out)


class CrossViewAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, *, bias: bool = True) -> None:
    super().__init__()

    if dim <= 0:
      raise ValueError(f"dim must be positive, got {dim}")

    if num_heads <= 0:
      raise ValueError(
        f"num_heads must be positive, got {num_heads}"
      )

    if dim % num_heads != 0:
      raise ValueError(
        f"dim ({dim}) must be divisible by "
        f"num_heads ({num_heads})"
      )

    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.qkv = nn.Linear(
      dim, 3 * dim, bias=bias
    )

    self.proj = nn.Linear(
      dim, dim, bias=bias
    )

  def forward(self, x: Tensor) -> Tensor:
    B, N, V, D = x.shape

    x = x.permute(0, 2, 1, 3)

    x = x.reshape(B * N, V, D)

    qkv = self.qkv(x)

    qkv = qkv.reshape(B*N, V, 3, self.num_heads, self.head_dim)

    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv.unbind(dim=0)

    out = scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2)
    out = out.reshape(B, N, V, D)
    out = out.permute(0, 2, 1, 3)

    return out
