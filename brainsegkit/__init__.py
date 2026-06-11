"""BrainSegKit — brain segmentation research framework."""

__version__ = "0.1.0"

from brainsegkit.models.registry import build_model, list_models, register_model
from brainsegkit.models import wrappers  # triggers model registration

__all__ = ["build_model", "list_models", "register_model"]
