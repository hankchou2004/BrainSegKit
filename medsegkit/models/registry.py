"""Central model registry — maps string names to nn.Module classes."""

from __future__ import annotations

from typing import Any, Type
import torch.nn as nn

_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator: @register_model('unet')"""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs: Any) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available: {list_models()}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
