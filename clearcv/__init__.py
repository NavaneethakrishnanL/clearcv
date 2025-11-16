"""
ClearCV — a lightweight, dependency-free computer vision toolkit.
"""

__version__ = "0.1.1"
__author__ = "Navaneetha Krishnan"

# Import submodules so users can access clearcv.io, clearcv.color, etc.
from . import io, color, filters, transform, utils, ops, metrics

# Public API — functions visible at top-level
from .io import imread, imwrite
from .color import rgb2gray, gray2rgb
from .ops import resize, rotate, blur, crop, pad, flip
from .metrics.ssim import ssim

__all__ = [
    # I/O
    "imread",
    "imwrite",

    # Color
    "rgb2gray",
    "gray2rgb",

    # Ops
    "resize",
    "rotate",
    "blur",
    "crop",
    "pad",
    "flip",

    # Metrics
    "ssim",

    # Modules
    "io",
    "color",
    "filters",
    "transform",
    "utils",
    "ops",
    "metrics",
]
