"""MedSegKit — brain segmentation research framework."""

__version__ = "0.1.0"

from medsegkit.models.registry import build_model, list_models, register_model
from medsegkit.models import wrappers  # triggers model registration

__all__ = ["build_model", "list_models", "register_model"]
