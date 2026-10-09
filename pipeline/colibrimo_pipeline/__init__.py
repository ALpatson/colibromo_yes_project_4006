"""Colibrimo reconstruction pipeline: phone video of a room -> Gaussian Splatting scene.

Use it from the command line (``cpipe run ...``) or as a library::

    from colibrimo_pipeline import load_config, run_pipeline

    config = load_config("balanced")
    manifest = run_pipeline("room.mp4", "data/runs/room1", config)
"""

from colibrimo_pipeline.config import PipelineConfig, load_config
from colibrimo_pipeline.errors import PipelineError, StepError
from colibrimo_pipeline.pipeline import STEP_NAMES, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "STEP_NAMES",
    "PipelineConfig",
    "PipelineError",
    "StepError",
    "__version__",
    "load_config",
    "run_pipeline",
]
