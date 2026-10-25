"""motiontrack — Lucas–Kanade optical flow, motion saliency and vehicle tracking.

The algorithmic core is a cleaned-up, extended version of the original
Colab script (``legacy/untitled0.py``) and the reference repository
ShreyaPI/Lucas-Kanade-with-Motion-Saliency.
"""

from .config import PipelineConfig, Layers
from .calibration import GroundCalibration
from .pipeline import run_pipeline, PipelineCancelled

__all__ = [
    "PipelineConfig",
    "Layers",
    "GroundCalibration",
    "run_pipeline",
    "PipelineCancelled",
]

__version__ = "2.0.0"
