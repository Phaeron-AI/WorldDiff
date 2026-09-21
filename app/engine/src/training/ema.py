from __future__ import annotations

# Native Import(s)
from collections import OrderedDict

# Third Party Import(s)
import torch
from torch import nn

class EMA:
  def __init__(self, model: nn.Module, *, beta: float = 0.9999) -> None:
    if not 0.0 <= beta < 1.0:
      raise ValueError(f"Expected Value: [0,1), Got: {beta}")

    self.beta = beta

    self.shadow = OrderedDict(
      (name, param.detach().float().clone()) for name, param in model.named_parameters()
    )

  @torch.no_grad()
  def update(self, model: nn.Module) -> None:
    beta = self.beta
    one_minus_beta = 1.0 - beta

    current_names = (name for name, _ in model.named_parameters())

    if current_names != set(self.shadow.keys()):
      raise ValueError("Model Params do not match EMA params")

    for name, param in model.named_parameters():
      shadow = self.shadow[name]

      if shadow.device != param.device:
        shadow = shadow.to(
          device=param.device
        )
        self.shadow[name] = shadow

      shadow.mul_(beta)
      shadow.add_(
        param.detach().float(),
        alpha=one_minus_beta,
      )

  @torch.no_grad()
  def copy_to(self, model: nn.Module) -> None:
    current_names = {
      name
      for name, _ in model.named_parameters()
    }

    if current_names != set(self.shadow.keys()):
      raise ValueError(
        "Model parameters do not match EMA parameters"
      )

    for name, param in model.named_parameters():
      param.copy_(
        self.shadow[name].to(
          device=param.device,
          dtype=param.dtype,
        )
      )

  def state_dict(self) -> dict:
    return {
      "beta": self.beta,
      "shadow": OrderedDict(
        (name, value.clone()) for name, value in self.shadow.items()
      ),
    }

  def load_state_dict(
    self,
    state_dict: dict,
  ) -> None:
    if "beta" not in state_dict:
      raise ValueError(
        "EMA state_dict is missing 'beta'"
      )

    if "shadow" not in state_dict:
      raise ValueError(
        "EMA state_dict is missing 'shadow'"
      )

    beta = float(state_dict["beta"])

    if not 0.0 <= beta < 1.0:
      raise ValueError(
        f"beta must satisfy 0 <= beta < 1, got {beta}"
      )

    shadow = state_dict["shadow"]

    if set(shadow.keys()) != set(self.shadow.keys()):
      raise ValueError(
        "EMA parameter names do not match"
      )

    self.beta = beta
    self.shadow = OrderedDict(
      (
        name,
        value.detach().float().clone()
      )
      for name, value in shadow.items()
    )