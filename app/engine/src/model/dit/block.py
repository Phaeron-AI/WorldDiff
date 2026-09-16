from __future__ import annotations

# Third Party Import(s)
from torch import nn, Tensor

# Local Import(s)
from src.model.attention.multiview import (
  PerViewAttention, CrossViewAttention
)

class FFN(nn.Module):
  def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
    super().__init__()

    hidden_dim = int(dim * mlp_ratio)

    self.net = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, dim)
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)

class DiTBlock(nn.Module):
  def __init__(
    self, 
    dim: int, 
    num_heads: int, 
    cond_dim: int,
    mlp_ratio: float = 4.0 
  ) -> None:
    super().__init__()

    self.dim = dim
    self.num_heads = num_heads
    self.mlp_ratio = mlp_ratio
    self.cond_dim = cond_dim 

    self.normal1 = nn.LayerNorm(dim, elementwise_affine=False)
    self.normal2 = nn.LayerNorm(dim, elementwise_affine=False)
    self.normal3 = nn.LayerNorm(dim, elementwise_affine=False)

    self.per_view_attention = PerViewAttention(
      dim=dim, 
      num_heads=num_heads
    )
    self.cross_view_attention = CrossViewAttention(
      dim=dim, 
      num_heads=num_heads
    )

    self.ffn = FFN(dim=dim, mlp_ratio=mlp_ratio)

    self.modulation = nn.Sequential(
      nn.SiLU(),
      nn.Linear(cond_dim, 9 * dim)  # type: ignore
    )

    nn.init.zeros_(self.modulation[-1].weight)  # type: ignore
    nn.init.zeros_(self.modulation[-1].bias)  # type: ignore

  def forward(self, tokens: Tensor, c: Tensor) -> Tensor:
    B, _, _, D = tokens.shape # [B, V, N, D]

    assert D == self.dim, (
      f"Expected token dimension {self.dim}, got {D}"
    )

    assert c.shape == (B, self.cond_dim), (
      f"Expected c shape {(B, self.cond_dim)}, got {c.shape}"
    )

    modulation = self.modulation(c)
    (
      gamma1, beta1, alpha1,
      gamma2, beta2, alpha2,
      gamma3, beta3, alpha3
    ) = modulation.chunk(9, dim=-1)

    gamma1 = gamma1[:, None, None, :]
    beta1 = beta1[:, None, None, :]
    alpha1 = alpha1[:, None, None, :]

    gamma2 = gamma2[:, None, None, :]
    beta2  = beta2[:, None, None, :]
    alpha2 = alpha2[:, None, None, :]

    gamma3 = gamma3[:, None, None, :]
    beta3  = beta3[:, None, None, :]
    alpha3 = alpha3[:, None, None, :]

    x = tokens

    h = self.normal1(x)
    h = (1 + gamma1) * h + beta1
    h = self.per_view_attention(h)

    x = x + alpha1 * h

    h = self.normal2(x)
    h = (1 + gamma2) * h + beta2
    h = self.cross_view_attention(h)

    x = x + alpha2 * h

    h = self.normal3(x)
    h = (1 + gamma3) * h + beta3
    h = self.ffn(h)

    x = x + alpha3 * h

    return x