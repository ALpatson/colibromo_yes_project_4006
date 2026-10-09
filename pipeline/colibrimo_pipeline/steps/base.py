"""Types shared by all steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colibrimo_pipeline.config import PipelineConfig
from colibrimo_pipeline.manifest import Manifest

# Shown whenever a failure is most likely caused by how the video was filmed.
CAPTURE_TIPS = (
    "This usually comes from the capture: move slowly, walk around the room "
    "(don't just turn on the spot), film at 2-3 heights, use good lighting and "
    "avoid mirrors and large windows. See docs/capture-guide.md."
)


@dataclass
class StepContext:
    """Everything a step needs to do its work."""

    run_dir: Path
    video: Path
    config: PipelineConfig
    manifest: Manifest
    logger: logging.Logger


@dataclass
class StepResult:
    """What a successful step produced.

    ``outputs``: files/folders created (checked to decide whether the step can be skipped).
    ``info``: measured results stored in the manifest.
    ``versions``: tool versions used, merged into the manifest's ``tool_versions``.
    """

    outputs: dict[str, Path]
    info: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
