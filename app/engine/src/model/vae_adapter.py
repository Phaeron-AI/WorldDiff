from __future__ import annotations

# Third-Party Import(s)
import torch
from torch import Tensor, nn, Generator

class VAEAdapter(nn.Module):
  def __init__(
    self, 
    model_name: str = "stabilityai/sd-vae-ft-mse",
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None
  ) -> None:
    super().__init__()

    from diffusers import AutoencoderKL as AE # type: ignore[import]

    vae = AE.from_pretrained(model_name)
    vae.to(device=device, dtype=dtype).eval().requires_grad_(False)  # type: ignore[call-overload]

    self.vae = vae

    self._scaling = float(vae.config.scaling_factor)
    self._latent_channels = int(vae.config.latent_channels) 
    self._downsample = 2 ** (len(vae.config.block_out_channels) - 1)

  @property
  def downsample(self) -> int:
    return self._downsample

  @property
  def latent_channels(self) -> int:
    return self._latent_channels

  @property
  def scaling_factor(self) -> float:
    return self._scaling

  @torch.no_grad()
  def encode(self, images: Tensor, *, sample: bool = False, generator: Generator | None = None) -> Tensor:
    lead = images.shape[:-3]  # arbitrary leading dims (B, or B,V, ...)
    x = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).to(self.vae.dtype) # [-1, 3, H, W]
    x = 2.0 * x - 1.0

    dist = self.vae.encode(x).latent_dist # type: ignore[union-attr]

    z = dist.sample(generator=generator) if sample else dist.mode() # [-1, C, H', W']
    z = z * self._scaling

    return z.reshape(*lead, self._latent_channels, z.shape[-2], z.shape[-1])

  @torch.no_grad()
  def decode(self, latents: Tensor) -> Tensor:
    lead = latents.shape[:-3]  # arbitrary leading dims (B, or B,V, ...)
    z = latents.reshape(-1, self.latent_channels, latents.shape[-2], latents.shape[-1]).to(self.vae.dtype) / self._scaling

    x = self.vae.decode(z).sample # type: ignore[union-attr]

    x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    return x.reshape(*lead, 3, x.shape[-2], x.shape[-1])

