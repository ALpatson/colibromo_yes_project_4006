"""Run the pipeline steps in order, with logging, manifest and resumability.

Resumability: a step is skipped when the manifest says it succeeded and all its
outputs still exist. Once a step actually runs, every later step runs too (its
inputs may have changed). ``force`` re-runs everything; ``from_step`` re-runs
from that step onwards; ``until`` stops after the given step (e.g. ``poses`` on
a laptop without GPU).
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from types import ModuleType

from colibrimo_pipeline.config import PipelineConfig
from colibrimo_pipeline.errors import ConfigError, PipelineError, StepError
from colibrimo_pipeline.manifest import Manifest
from colibrimo_pipeline.steps import export, extract_frames, poses, train, validate
from colibrimo_pipeline.steps.base import StepContext
from colibrimo_pipeline.utils.logs import close_logging, setup_logging, step_log_file
from colibrimo_pipeline.utils.versions import git_commit, pipeline_version, system_info

STEPS: list[ModuleType] = [validate, extract_frames, poses, train, export]
STEP_NAMES: list[str] = [step.NAME for step in STEPS]


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def _warn_if_inputs_changed(
    manifest: Manifest, video: Path, config: PipelineConfig, logger
):
    previous_config = manifest.data.get("config")
    previous_video = manifest.data.get("input", {}).get("video")
    if previous_config and previous_config != config.model_dump(mode="json"):
        logger.warning(
            "Warning: the configuration differs from the previous run in this folder. "
            "Completed steps are reused; use --force or --from-step to recompute them."
        )
    if previous_video and Path(previous_video).name != video.name:
        logger.warning(
            "Warning: this folder was used for '%s' before; now '%s'. "
            "Use a new output folder or --force.",
            Path(previous_video).name,
            video.name,
        )


def _run_step(step: ModuleType, ctx: StepContext, position: str) -> None:
    name, manifest, logger = step.NAME, ctx.manifest, ctx.logger
    logger.info("[%s] %s: starting", position, name)
    manifest.mark_started(name)
    manifest.save()

    start = time.monotonic()
    error: StepError | None = None
    with step_log_file(logger, ctx.run_dir / "logs" / f"{name}.log"):
        try:
            result = step.run(ctx)
        except StepError as exc:
            error = exc
        except PipelineError as exc:
            error = StepError(name, str(exc))
        except Exception as exc:  # unexpected bug: keep the traceback in the logs
            logger.debug("%s", traceback.format_exc())
            error = StepError(name, f"Unexpected error: {exc!r}. See logs/{name}.log.")
        duration = time.monotonic() - start

        if error is not None:
            manifest.mark_failed(name, error.message, duration)
            if error.info:
                manifest.step(name)["info"] = error.info
            manifest.set_status("failed", str(error))
            manifest.save()
            logger.error("[%s] FAILED: %s", position, error)
            raise error

    manifest.mark_succeeded(name, result.outputs, result.info, duration)
    manifest.add_tool_versions(result.versions)
    manifest.save()
    logger.info("[%s] %s: done in %s", position, name, _format_duration(duration))


def run_pipeline(
    video: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    *,
    force: bool = False,
    from_step: str | None = None,
    until: str | None = None,
    verbose: bool = False,
    steps: list[ModuleType] | None = None,
) -> Manifest:
    """Run the pipeline on ``video`` into ``output_dir`` and return the manifest.

    Raises StepError (with ``.step``) if a step fails. ``steps`` is only meant
    for tests (to swap in fake steps).
    """
    steps = steps or STEPS
    names = [step.NAME for step in steps]
    for option, value in (("from_step", from_step), ("until", until)):
        if value is not None and value not in names:
            raise ConfigError(f"Unknown step '{value}' for {option}. Steps: {names}")

    video = Path(video).resolve()
    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir / "logs", verbose)
    try:
        manifest = Manifest.load_or_create(run_dir)
        _warn_if_inputs_changed(manifest, video, config, logger)
        manifest.set_run_info(
            pipeline_version=pipeline_version(),
            git_commit=git_commit(),
            input={"video": str(video)},
            config=config.model_dump(mode="json"),
            system=system_info(),
        )
        manifest.set_status("running")
        manifest.save()
        logger.info("Run folder: %s (preset '%s')", run_dir, config.preset)

        start_index = names.index(from_step) if from_step else None
        last_index = names.index(until) if until else len(steps) - 1
        ctx = StepContext(run_dir, video, config, manifest, logger)
        rerun = force
        total_start = time.monotonic()

        for index, step in enumerate(steps[: last_index + 1]):
            position = f"{index + 1}/{len(steps)}"
            if start_index is not None and index >= start_index:
                rerun = True
            if not rerun and manifest.is_step_done(step.NAME):
                logger.info(
                    "[%s] %s: already done, skipping (use --force to re-run)",
                    position,
                    step.NAME,
                )
                continue
            rerun = True  # later steps depend on this one: recompute them too
            _run_step(step, ctx, position)

        finished_all = last_index == len(steps) - 1
        manifest.set_status(
            "succeeded" if finished_all else f"stopped after '{names[last_index]}'"
        )
        manifest.data["total_duration_s"] = round(time.monotonic() - total_start, 2)
        manifest.save()
        logger.info(
            "Pipeline %s in %s.",
            "finished" if finished_all else f"stopped after '{names[last_index]}'",
            _format_duration(manifest.data["total_duration_s"]),
        )
        return manifest
    finally:
        close_logging(logger)
