from __future__ import annotations

# Third Party Imports
import torch
from torch import Tensor, nn

# Local Imports
from src.models.dit.embeddings import (
  TimeStepEmbedding,
  PatchEmbedding,
  ViewConditionEmbedder,
  unpatchify,
  get_2d_sincos_pos_embedding,
)
from src.models.dit.block import DiTBlock


class MultiViewDiT(nn.Module):
  def __init__(
    self,
    in_channels: int,
    dim: int,
    num_heads: int,
    cond_dim: int,
    num_layers: int,
    patch_size: int = 2,
    mlp_ratio: float = 4.0,
    frequency_embedding_size: int = 256,
  ) -> None:
    super().__init__()

    assert dim % num_heads == 0, "dim must be divisible by num_heads"

    self.in_channels = in_channels
    self.dim = dim
    self.num_heads = num_heads
    self.cond_dim = cond_dim
    self.num_layers = num_layers
    self.patch_size = patch_size

    # x:    [B, V, C,   H, W]
    # rays: [B, V, 6,   H, W]
    # input to PatchEmbedding: [B, V, C+6, H, W]
    self.patch_embed = PatchEmbedding(
      in_channels=in_channels + 6,
      embed_dim=dim,
      patch_size=patch_size,
    )

    # t: [B]
    # -> Fourier features: [B, frequency_embedding_size]
    # -> c: [B, cond_dim]
    self.time_step_embedding = TimeStepEmbedding(
      cond_dim=cond_dim,
      frequency_embedding_size=frequency_embedding_size,
    )

    # cond_mask: [B, V]
    # -> [B, V, D]
    self.view_cond_embedding = ViewConditionEmbedder(
      embed_dim=dim,
    )

    self.multi_view_dit = nn.ModuleList(
      [
        DiTBlock(
          dim=dim,
          num_heads=num_heads,
          cond_dim=cond_dim,
          mlp_ratio=mlp_ratio,
        )
        for _ in range(num_layers)
      ]
    )

    self.output_layer = nn.Linear(
      dim,
      patch_size * patch_size * in_channels,
    )

    nn.init.zeros_(self.output_layer.weight)
    nn.init.zeros_(self.output_layer.bias)

  def forward(
    self,
    x: Tensor,
    rays: Tensor,
    t: Tensor,
    cond_mask: Tensor,
  ) -> Tensor:

    B, V, C, H, W = x.shape

    assert C == self.in_channels
    assert rays.shape == (B, V, 6, H, W)

    x = torch.cat([x, rays], dim=2) # [B, V, C+6, H, W]

    tokens = self.patch_embed(x) # [B, V, N, D]

    H_p = H // self.patch_size
    W_p = W // self.patch_size

    pos_embed = get_2d_sincos_pos_embedding(
      embed_dim=self.dim,
      grid_h=H_p,
      grid_w=W_p,
    ).to(tokens) # [N, D]

    tokens = tokens + pos_embed # [B, V, N, D]

    view_cond_emb = self.view_cond_embedding(cond_mask) # [B, V, D]

    tokens = tokens + view_cond_emb.unsqueeze(2) # [B, V, N, D]

    c = self.time_step_embedding(t) # [B, cond_dim]

    for block in self.multi_view_dit:
      tokens = block(tokens, c) # [B, V, N, D]

    tokens = self.output_layer(tokens) # [B, V, N, p²C]

    velocity = unpatchify(
      tokens=tokens,
      patch_size=self.patch_size,
      height=H,
      width=W,
      out_channels=self.in_channels,
    ) # [B, V, C, H, W]

    return velocity