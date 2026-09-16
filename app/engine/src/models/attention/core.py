from __future__ import annotations

# Native Import(s)
import math

# Third Party Import(s)
import torch
from torch import Tensor, nn

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
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
    raise ValueError(f"Expected: [B, H, Nq, Dh], Got: {tuple(q.shape)}")
  
  if k.ndim != 4:
    raise ValueError(f"Expected: [B, H, Nk, Dh], Got: {tuple(k.shape)}")

  if v.ndim != 4:
    raise ValueError(
      f"v must have shape [B,H,Nk,Dh], "
      f"got {tuple(v.shape)}"
    )

  if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
    raise ValueError(
      "q, k, and v must have the same batch size"
    )

  if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
    raise ValueError(
      "q, k, and v must have the same number of heads"
    )

  if q.shape[-1] != k.shape[-1]:
    raise ValueError(
      "q and k must have the same head dimension"
    )

  if k.shape[-2] != v.shape[-2]:
    raise ValueError(
      "k and v must have the same sequence length"
    )

  scale = 1.0 / math.sqrt(q.shape[-1])  # 1/sqrt(Dk)
  
  # z = (Q @ K^T) / sqrt(Dk)
  scores = torch.matmul(
    q, k.transpose(-2, -1),
  ) * scale

  # A = softmax(z)
  attention_weights = torch.softmax(scores, dim=-1)

  return torch.matmul(attention_weights, v)


class MultiHeadAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, *, bias: bool = True) -> None:
    super().__init__()

    if dim <= 0: 
      raise ValueError(f"dim must be positive, Got: {dim}")

    if num_heads <= 0: 
      raise ValueError(f"num_heads must be positive, Got: {num_heads}")

    if dim % num_heads != 0:
      raise ValueError(
        f"dim ({dim}) must be divisible by "
        f"num_heads ({num_heads})"
      )
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.qkv = nn.Linear(dim, 3 * dim, bias=bias)

    self.proj = nn.Linear(dim, dim, bias=bias)

  def forward(self, x: Tensor) -> Tensor:
    if x.ndim != 3:
      raise ValueError(
        f"x must have shape [B,N,D], "
        f"got {tuple(x.shape)}"
      )

    B, N, D = x.shape # [B, N, D]
    if D != self.dim:
      raise ValueError(
        f"Expected embedding dim: {self.dim}"
        f" Got: {D}"
      )

    qkv = self.qkv(x)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv.unbind(dim=0)

    out = scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2)
    out = out.reshape(B, N, D)

    return self.proj(out)