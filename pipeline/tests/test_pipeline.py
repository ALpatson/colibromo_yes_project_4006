"""Runner behaviour (skip / force / from-step / until / failures) with fake steps."""

import json
from types import SimpleNamespace

import pytest

from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.errors import ConfigError, StepError
from colibrimo_pipeline.pipeline import STEP_NAMES, run_pipeline
from colibrimo_pipeline.steps.base import StepResult


def test_real_step_order():
    assert STEP_NAMES == ["validate", "extract_frames", "poses", "train", "export"]


class Recorder:
    def __init__(self):
        self.calls = []


def fake_step(name, recorder, fail=False):
    def run(ctx):
        recorder.calls.append(name)
        if fail:
            raise StepError(name, "deliberate failure", info={"measured": 1})
        out = ctx.run_dir / f"{name}.out"
        out.write_text(name)
        return StepResult(
            outputs={"out": out}, info={"ok": True}, versions={name: "1.0"}
        )

    return SimpleNamespace(NAME=name, run=run)


@pytest.fixture
def setup(tmp_path):
    video = tmp_path / "room.mp4"
    video.write_text("fake")
    recorder = Recorder()
    steps = [fake_step(n, recorder) for n in ("a", "b", "c")]
    run_dir = tmp_path / "run"

    def run(**kwargs):
        return run_pipeline(video, run_dir, load_config("fast"), steps=steps, **kwargs)

    return SimpleNamespace(run=run, recorder=recorder, steps=steps, run_dir=run_dir)


def test_first_run_executes_all_steps_and_writes_manifest(setup):
    manifest = setup.run()
    assert setup.recorder.calls == ["a", "b", "c"]
    data = json.loads((setup.run_dir / "manifest.json").read_text())
    assert data["status"] == "succeeded"
    assert data["config"]["preset"] == "fast"
    assert data["tool_versions"] == {"a": "1.0", "b": "1.0", "c": "1.0"}
    assert all(data["steps"][n]["status"] == "succeeded" for n in "abc")
    assert manifest.data["pipeline_version"]
    for name in ("pipeline", "a", "b", "c"):
        assert (setup.run_dir / "logs" / f"{name}.log").is_file()


def test_rerun_skips_completed_steps(setup):
    setup.run()
    setup.recorder.calls.clear()
    setup.run()
    assert setup.recorder.calls == []


def test_force_reruns_everything(setup):
    setup.run()
    setup.recorder.calls.clear()
    setup.run(force=True)
    assert setup.recorder.calls == ["a", "b", "c"]


def test_from_step_reruns_that_step_and_later_ones(setup):
    setup.run()
    setup.recorder.calls.clear()
    setup.run(from_step="b")
    assert setup.recorder.calls == ["b", "c"]


def test_missing_output_reruns_that_step_and_later_ones(setup):
    setup.run()
    setup.recorder.calls.clear()
    (setup.run_dir / "b.out").unlink()
    setup.run()
    assert setup.recorder.calls == ["b", "c"]


def test_until_stops_early_and_resume_continues(setup):
    manifest = setup.run(until="a")
    assert setup.recorder.calls == ["a"]
    assert manifest.data["status"] == "stopped after 'a'"

    setup.recorder.calls.clear()
    manifest = setup.run()
    assert setup.recorder.calls == ["b", "c"]
    assert manifest.data["status"] == "succeeded"


def test_failure_names_step_and_is_recorded(setup):
    setup.steps[1] = fake_step("b", setup.recorder, fail=True)
    with pytest.raises(StepError) as excinfo:
        setup.run()
    assert excinfo.value.step == "b"
    assert "Step 'b' failed: deliberate failure" in str(excinfo.value)
    assert setup.recorder.calls == ["a", "b"]  # 'c' never ran

    data = json.loads((setup.run_dir / "manifest.json").read_text())
    assert data["status"] == "failed"
    assert data["steps"]["b"]["status"] == "failed"
    assert data["steps"]["b"]["info"] == {"measured": 1}
    assert "deliberate failure" in (setup.run_dir / "logs" / "b.log").read_text()


def test_unexpected_exception_becomes_step_error(setup):
    def broken(ctx):
        raise ZeroDivisionError("oops")

    setup.steps[0] = SimpleNamespace(NAME="a", run=broken)
    with pytest.raises(StepError, match="Unexpected error"):
        setup.run()
    assert "ZeroDivisionError" in (setup.run_dir / "logs" / "a.log").read_text()


def test_unknown_step_name_is_rejected(setup):
    with pytest.raises(ConfigError, match="Unknown step"):
        setup.run(from_step="nope")
