"""cpipe ui and Colab (D-020): stop after COLMAP, export the zip, import the result."""

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from test_colab import laptop_run_after_poses, trained_on_colab  # noqa: F401 (fixture)
from test_ui import FakePipeline, request, runner_with_videos, server  # noqa: F401

from colibrimo_pipeline.ui.jobs import Runner, UserError, read_progress


def test_stop_after_colmap_is_remembered_by_resume(tmp_path):
    fake = FakePipeline()
    runner = runner_with_videos(tmp_path, fake)
    runner.start("kitchen", "balanced", until="poses")
    runner.wait(30)
    runner.resume("kitchen")
    runner.wait(30)
    runner.resume("kitchen", train_here=True)  # "Train on this laptop instead"
    runner.wait(30)
    assert fake.untils == ["poses", "poses", None]
    assert read_progress(runner.run_dir("kitchen"))["until"] is None


def test_unknown_until_is_refused(tmp_path):
    runner = runner_with_videos(tmp_path, FakePipeline())
    with pytest.raises(UserError, match="Unknown step"):
        runner.start("kitchen", "balanced", until="everything")


def test_export_for_colab_from_the_page(tmp_path):
    runner = Runner(tmp_path, run_fn=FakePipeline())
    with pytest.raises(UserError, match="no manifest.json"):
        runner.export_for_colab("kitchen")
    laptop_run_after_poses(runner.run_dir("kitchen"))
    assert read_progress(runner.run_dir("kitchen"))["poses_done"] is True
    assert runner.colab_export_info("kitchen") is None

    info = runner.export_for_colab("kitchen")
    assert info["path"] == str((tmp_path / "colab" / "kitchen_for_colab.zip").resolve())
    assert info["size_bytes"] > 0
    assert runner.colab_export_info("kitchen") == info


def runner_with_laptop_run(tmp_path, laptop):
    runner = Runner(tmp_path / "data", run_fn=FakePipeline())
    shutil.copytree(laptop, runner.run_dir("kitchen"))
    return runner


def test_import_result_zip(trained_on_colab, tmp_path):  # noqa: F811
    laptop, _, packed = trained_on_colab
    runner = runner_with_laptop_run(tmp_path, laptop)
    data = Path(packed["path"]).read_bytes()
    answer = runner.import_colab_upload(
        "kitchen", "kitchen_colab_result.zip", io.BytesIO(data), len(data)
    )
    assert answer["imported"] is True
    assert answer["check"][-1].startswith("Overall: PASS")
    progress = read_progress(runner.run_dir("kitchen"))
    assert progress["status"] == "succeeded"
    assert progress["result"]["imported_at"]
    assert progress["result"]["trainer"] == "gsplat simple_trainer.py"
    assert progress["result"]["gpu"] == "Tesla T4"


def test_import_three_files_one_by_one(trained_on_colab, tmp_path):  # noqa: F811
    _, colab_run, _ = trained_on_colab
    runner = runner_with_laptop_run(tmp_path, trained_on_colab[0])
    answers = []
    for name in ("scene.ply", "cameras.json", "manifest.json"):
        data = (colab_run / name).read_bytes()
        answers.append(
            runner.import_colab_upload("kitchen", name, io.BytesIO(data), len(data))
        )
    assert answers[0] == {
        "imported": False,
        "waiting_for": ["cameras.json", "manifest.json"],
    }
    assert answers[-1]["imported"] is True


def test_refused_import_changes_nothing(trained_on_colab, tmp_path):  # noqa: F811
    laptop, _, packed = trained_on_colab
    runner = runner_with_laptop_run(tmp_path, laptop)
    before = (runner.run_dir("kitchen") / "manifest.json").read_text()
    with zipfile.ZipFile(packed["path"]) as z:
        manifest = json.loads(z.read("manifest.json"))
    manifest["steps"]["poses"]["finished_at"] = "2020-01-01T00:00:00+00:00"
    data = json.dumps(manifest).encode()
    with pytest.raises(UserError, match="not trained from this room's current"):
        for name in ("scene.ply", "cameras.json"):
            raw = (trained_on_colab[1] / name).read_bytes()
            runner.import_colab_upload("kitchen", name, io.BytesIO(raw), len(raw))
        runner.import_colab_upload(
            "kitchen", "manifest.json", io.BytesIO(data), len(data)
        )
    assert (runner.run_dir("kitchen") / "manifest.json").read_text() == before
    assert not (runner.run_dir("kitchen") / "scene.ply").exists()
    assert not (runner.colab_dir / "import" / "kitchen").exists()  # retry starts clean


def test_import_accepts_only_result_files(tmp_path):
    runner = Runner(tmp_path, run_fn=FakePipeline())
    with pytest.raises(UserError, match="choose the result zip"):
        runner.import_colab_upload("kitchen", "evil.exe", io.BytesIO(b"x"), 1)


def test_server_colab_routes(server):  # noqa: F811
    port, _ = server
    status, body = request(
        port, "PUT", "/api/import-colab?room=kitchen&file=notes.txt", body=b"x"
    )
    assert status == 400 and b"choose the result zip" in body
    status, body = request(
        port,
        "POST",
        "/api/export-colab",
        body=json.dumps({"room": "kitchen"}),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400 and b"no manifest.json" in body
