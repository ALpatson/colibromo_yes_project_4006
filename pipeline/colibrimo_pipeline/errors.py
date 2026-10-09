"""Exceptions raised by the pipeline. Messages are meant to be read by humans."""


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(PipelineError):
    """Invalid preset, override or configuration value."""


class ToolNotFoundError(PipelineError):
    """An external program (ffmpeg, ffprobe, colmap, ...) could not be found."""


class StepError(PipelineError):
    """A pipeline step failed. ``step`` names the failing step.

    ``info`` optionally carries what the step measured before failing (e.g. the
    share of frames COLMAP registered), so it ends up in the manifest too.
    """

    def __init__(self, step: str, message: str, info: dict | None = None):
        super().__init__(message)
        self.step = step
        self.message = message
        self.info = info or {}

    def __str__(self) -> str:
        return f"Step '{self.step}' failed: {self.message}"
