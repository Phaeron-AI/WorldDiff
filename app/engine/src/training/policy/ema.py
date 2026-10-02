from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class EMA:
  def __init__(
    self,
    model: nn.Module,
    *,
    beta: float = 0.9999,
  ) -> None:
    if not 0.0 <= beta < 1.0:
      raise ValueError(
        f"beta must satisfy 0 <= beta < 1, got {beta}"
      )

    self.beta = beta

    self.shadow = OrderedDict(
      (
        name,
        param.detach().float().clone()
      )
      for name, param in model.named_parameters()
    )

    self._stored = None

  def _validate_model(
    self,
    model: nn.Module,
  ) -> None:
    model_names = {
      name
      for name, _ in model.named_parameters()
    }

    ema_names = set(self.shadow.keys())

    if model_names != ema_names:
      raise ValueError(
        "Model Params do not match EMA params"
      )

  @torch.no_grad()
  def update(
    self,
    model: nn.Module,
  ) -> None:
    self._validate_model(model)

    beta = self.beta
    one_minus_beta = 1.0 - beta

    for name, param in model.named_parameters():
      shadow = self.shadow[name]

      if shadow.device != param.device:
        shadow = shadow.to(
          device=param.device,
        )
        self.shadow[name] = shadow

      shadow.mul_(beta)
      shadow.add_(
        param.detach().float(),
        alpha=one_minus_beta,
      )

  @torch.no_grad()
  def store(
    self,
    model: nn.Module,
  ) -> None:
    """Store the current raw model parameters for later restoration."""
    self._validate_model(model)

    if self._stored is not None:
      raise RuntimeError(
        "Model parameters are already stored. "
        "Call restore(model) before store(model) again."
      )

    self._stored = OrderedDict(
      (
        name,
        param.detach().clone()
      )
      for name, param in model.named_parameters()
    )

  @torch.no_grad()
  def copy_to(
    self,
    model: nn.Module,
  ) -> None:
    """Copy EMA parameters into the model for evaluation."""
    self._validate_model(model)

    for name, param in model.named_parameters():
      param.copy_(
        self.shadow[name].to(
          device=param.device,
          dtype=param.dtype,
        )
      )

  @torch.no_grad()
  def restore(
    self,
    model: nn.Module,
  ) -> None:
    """Restore the raw parameters saved by store()."""
    self._validate_model(model)

    if self._stored is None:
      raise RuntimeError(
        "No stored model parameters. "
        "Call store(model) before restore(model)."
      )

    for name, param in model.named_parameters():
      param.copy_(
        self._stored[name].to(
          device=param.device,
          dtype=param.dtype,
        )
      )

    self._stored = None

  def state_dict(self) -> dict:
    """Return a detached copy of the EMA state."""
    return {
      "beta": self.beta,
      "shadow": OrderedDict(
        (
          name,
          value.detach().clone()
        )
        for name, value in self.shadow.items()
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

    if self._stored is not None:
      raise RuntimeError(
        "Cannot load EMA state while model parameters "
        "are stored for evaluation."
      )

    self.beta = beta

    self.shadow = OrderedDict(
      (
        name,
        value.detach().float().clone()
      )
      for name, value in shadow.items()
    )