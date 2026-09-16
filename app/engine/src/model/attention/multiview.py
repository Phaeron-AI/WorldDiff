from __future__ import annotations

# Third Party Import(s)
from torch import nn, Tensor

# Local Import(s)
from src.model.attention.core import MultiHeadAttention


class PerViewAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, *, bias: bool = True) -> None:
    super().__init__()

    self.mha = MultiHeadAttention(
      dim=dim, 
      num_heads=num_heads,
      bias=bias
    )

  def forward(self, x: Tensor) -> Tensor:
    if x.ndim != 4:
      raise ValueError(
        f"x must have shape [B, V, N, D]",
        f"Got: {tuple(x.shape)}"
      )

    B, V, N, D = x.shape

    x = x.reshape(B*V, N, D)

    x = self.mha(x)

    return x.reshape(B, V, N, D)

class CrossViewAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, *, bias: bool = True) -> None:
    super().__init__()

    self.mha = MultiHeadAttention(
      dim=dim,
      num_heads=num_heads,
      bias=bias
    )

  def forward(self, x: Tensor) -> Tensor:
    if x.ndim != 4:
      raise ValueError(
        f"x must have shape [B,V,N,D], "
        f"got {tuple(x.shape)}"
      )

    B, V, N, D = x.shape

    x = x.reshape(B, V*N, D)
    x = self.mha(x)

    return x.reshape(B, V, N, D)