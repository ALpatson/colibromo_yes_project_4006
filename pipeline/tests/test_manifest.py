import json

from colibrimo_pipeline.manifest import Manifest


def test_new_manifest_has_expected_structure(tmp_path):
    manifest = Manifest.load_or_create(tmp_path)
    assert manifest.data["status"] == "new"
    assert manifest.data["steps"] == {}
    assert not manifest.path.exists()  # nothing written until save()


def test_step_lifecycle_and_reload(tmp_path):
    out_file = tmp_path / "out" / "result.txt"
    out_file.parent.mkdir()
    out_file.write_text("x")

    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_started("demo")
    manifest.mark_succeeded(
        "demo", {"result": out_file}, {"answer": 42}, duration_s=1.234
    )
    manifest.add_tool_versions({"ffmpeg": "9.0"})
    manifest.save()

    data = json.loads((tmp_path / "manifest.json").read_text())
    step = data["steps"]["demo"]
    assert step["status"] == "succeeded"
    assert step["outputs"] == {"result": "out/result.txt"}  # relative, forward slashes
    assert step["info"] == {"answer": 42}
    assert step["duration_s"] == 1.23
    assert data["tool_versions"] == {"ffmpeg": "9.0"}

    reloaded = Manifest.load_or_create(tmp_path)
    assert reloaded.is_step_done("demo")
    assert reloaded.step_output("demo", "result") == tmp_path / "out" / "result.txt"
    assert reloaded.step_info("demo") == {"answer": 42}


def test_step_not_done_when_output_was_deleted(tmp_path):
    out_file = tmp_path / "result.txt"
    out_file.write_text("x")
    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_succeeded("demo", {"result": out_file}, {}, duration_s=0)
    assert manifest.is_step_done("demo")
    out_file.unlink()
    assert not manifest.is_step_done("demo")


def test_failed_step_is_not_done(tmp_path):
    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_started("demo")
    manifest.mark_failed("demo", "boom", duration_s=0.5)
    assert manifest.step("demo")["error"] == "boom"
    assert not manifest.is_step_done("demo")
    assert not manifest.is_step_done("never-ran")


def test_run_folder_can_be_moved(tmp_path):
    """Relative paths keep a run usable after copying it elsewhere (laptop -> Colab)."""
    first = tmp_path / "first"
    (first / "colmap").mkdir(parents=True)
    manifest = Manifest.load_or_create(first)
    manifest.mark_succeeded("demo", {"folder": first / "colmap"}, {}, duration_s=0)
    manifest.save()

    moved = tmp_path / "moved"
    first.rename(moved)
    assert Manifest.load_or_create(moved).is_step_done("demo")


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    manifest = Manifest.load_or_create(tmp_path)
    manifest.save()
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]
