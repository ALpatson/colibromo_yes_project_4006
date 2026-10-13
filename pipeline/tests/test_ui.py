"""cpipe ui: uploads, progress reading, runs/resume, viewer copy, local-only HTTP server."""

import http.client
import io
import json
import threading
from pathlib import Path

import pytest

from colibrimo_pipeline.errors import StepError
from colibrimo_pipeline.ui.jobs import (
    Busy,
    Runner,
    UserError,
    clean_room_name,
    clean_video_filename,
    copy_to_viewer,
    list_rooms,
    list_videos,
    read_progress,
    save_upload,
    ui_overrides,
)
from colibrimo_pipeline.ui.server import App, make_server

# ---- names and uploads --------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [("kitchen", "kitchen"), ("Living room 2", "Living_room_2"), ("  a-b_c ", "a-b_c")],
)
def test_room_names(name, expected):
    assert clean_room_name(name) == expected


@pytest.mark.parametrize("name", ["", "../etc", "a/b", "_hidden", "x" * 65, "piÃ¨ce"])
def test_bad_room_names(name):
    with pytest.raises(UserError):
        clean_room_name(name)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("bottom.mp4", "bottom.mp4"),
        ("IMG 0042.MOV", "IMG_0042.mov"),
        ("../../evil.mp4", "evil.mp4"),  # no way out of the room folder
        ("C:\\Users\\x\\up.m4v", "up.m4v"),
    ],
)
def test_video_filenames(name, expected):
    assert clean_video_filename(name) == expected


@pytest.mark.parametrize("name", ["notes.txt", "run.exe", "video", ""])
def test_non_videos_are_refused(name):
    with pytest.raises(UserError, match="not a video"):
        clean_video_filename(name)


def test_save_upload_streams_to_room_folder(tmp_path):
    data = b"x" * 3_000_000
    path = save_upload(
        tmp_path / "uploads",
        "kitchen",
        "middle.MP4",
        io.BytesIO(data),
        len(data),
        1 << 20,
    )
    assert path == tmp_path / "uploads" / "kitchen" / "middle.mp4"
    assert path.read_bytes() == data
    assert [p.name for p in list_videos(tmp_path / "uploads", "kitchen")] == [
        "middle.mp4"
    ]


def test_interrupted_upload_leaves_nothing(tmp_path):
    with pytest.raises(UserError, match="interrupted"):
        save_upload(tmp_path, "kitchen", "up.mp4", io.BytesIO(b"short"), 100)
    assert list((tmp_path / "kitchen").iterdir()) == []


def test_list_rooms_from_uploads_and_runs(tmp_path):
    (tmp_path / "uploads" / "a").mkdir(parents=True)
    (tmp_path / "runs" / "b").mkdir(parents=True)
    (tmp_path / "runs" / "a").mkdir()
    assert list_rooms(tmp_path) == ["a", "b"]


# ---- progress ------------------------------------------------------------------------


def write_manifest(
    run_dir: Path, steps: dict, status="running", preset="fast", **extra
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "status": status,
        "config": {"preset": preset, "train": {"max_steps": 30000}},
        "steps": steps,
        **extra,
    }
    (run_dir / "manifest.json").write_text(json.dumps(data))


def test_no_manifest_no_progress(tmp_path):
    assert read_progress(tmp_path) is None


def test_progress_during_brush_training(tmp_path):
    write_manifest(
        tmp_path,
        {
            "validate": {"status": "succeeded", "duration_s": 0.3},
            "extract_frames": {"status": "succeeded", "duration_s": 60},
            "poses": {"status": "succeeded", "duration_s": 1200},
            "train": {"status": "running"},
        },
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "train.log").write_text(
        "2026 DEBUG [.. INFO  brush_cli] Refine iter 5801, 106302 splats.\n"
        "2026 DEBUG [.. INFO  cubecl_runtime::tune::tuner] Tuning ...\n"
        "2026 DEBUG [.. INFO  brush_cli] Refine iter 6001, 110507 splats.\n"
    )
    progress = read_progress(tmp_path)
    assert [s["status"] for s in progress["steps"]] == [
        "succeeded",
        "succeeded",
        "succeeded",
        "running",
        "pending",
    ]
    assert progress["brush"] == {"step": 6001, "total": 30000, "splats": 110507}
    assert progress["result"] is None


def test_progress_failed_step_shows_error(tmp_path):
    write_manifest(
        tmp_path,
        {"validate": {"status": "failed", "error": "The video is 5.0 s long"}},
        status="failed",
    )
    progress = read_progress(tmp_path)
    assert progress["steps"][0] == {
        "name": "validate",
        "status": "failed",
        "duration_s": None,
        "error": "The video is 5.0 s long",
    }
    assert progress["brush"] is None


def test_progress_result_after_export(tmp_path):
    write_manifest(
        tmp_path,
        {
            "train": {
                "status": "succeeded",
                "duration_s": 3530,
                "info": {"train_seconds": 3529.4, "gpu": {"name": "NVIDIA T500"}},
            },
            "export": {
                "status": "succeeded",
                "info": {
                    "num_gaussians": 282405,
                    "file_size_bytes": 66649131,
                    "num_cameras": 300,
                },
            },
        },
        status="succeeded",
    )
    assert read_progress(tmp_path)["result"] == {
        "train_seconds": 3529.4,
        "num_gaussians": 282405,
        "file_size_bytes": 66649131,
        "num_cameras": 300,
        "gpu": "NVIDIA T500",
    }


# ---- runs and resume ----------------------------------------------------------------


class FakePipeline:
    """Stands in for run_pipeline(): records calls, can block or fail once."""

    def __init__(self, fail_first=False):
        self.calls = []
        self.release = threading.Event()
        self.release.set()
        self.fail_first = fail_first

    def __call__(self, videos, run_dir, config):
        self.calls.append((list(videos), Path(run_dir), config.preset))
        self.release.wait(30)
        if self.fail_first and len(self.calls) == 1:
            write_manifest(
                run_dir,
                {"poses": {"status": "failed", "error": "COLMAP placed 10%"}},
                status="failed",
                preset=config.preset,  # like run_pipeline, which records the config
            )
            raise StepError("poses", "COLMAP placed 10%")
        write_manifest(
            run_dir,
            {"export": {"status": "succeeded"}},
            status="succeeded",
            preset=config.preset,
        )


def runner_with_videos(tmp_path, fake, room="kitchen"):
    folder = tmp_path / "uploads" / room
    folder.mkdir(parents=True)
    for name in ("bottom.mp4", "middle.mp4", "notes.txt"):
        (folder / name).write_bytes(b"v")
    return Runner(tmp_path, {"train": {"trainer": "brush"}}, run_fn=fake)


def test_start_runs_pipeline_with_room_videos(tmp_path):
    fake = FakePipeline()
    runner = runner_with_videos(tmp_path, fake)
    runner.start("kitchen", "fast")
    runner.wait(30)
    videos, run_dir, preset = fake.calls[0]
    assert [v.name for v in videos] == ["bottom.mp4", "middle.mp4"]
    assert run_dir == tmp_path / "runs" / "kitchen"
    assert preset == "fast"
    assert read_progress(run_dir)["status"] == "succeeded"


def test_only_one_run_at_a_time(tmp_path):
    fake = FakePipeline()
    fake.release.clear()  # keep the first run going
    runner = runner_with_videos(tmp_path, fake)
    runner.start("kitchen", "balanced")
    assert runner.running
    with pytest.raises(Busy, match="still running"):
        runner.start("kitchen", "balanced")
    fake.release.set()
    runner.wait(30)
    assert not runner.running


def test_resume_reruns_the_same_room_and_preset(tmp_path):
    fake = FakePipeline(fail_first=True)
    runner = runner_with_videos(tmp_path, fake)
    runner.start("kitchen", "quality")
    runner.wait(30)
    progress = read_progress(tmp_path / "runs" / "kitchen")
    assert progress["status"] == "failed"
    assert progress["steps"][2]["error"] == "COLMAP placed 10%"

    runner.resume("kitchen")  # preset read back from the manifest
    runner.wait(30)
    assert [c[2] for c in fake.calls] == ["quality", "quality"]
    assert (
        fake.calls[1][1] == fake.calls[0][1]
    )  # same run folder: completed steps reused
    assert read_progress(tmp_path / "runs" / "kitchen")["status"] == "succeeded"


def test_start_without_videos_or_bad_preset(tmp_path):
    runner = Runner(tmp_path, run_fn=FakePipeline())
    with pytest.raises(UserError, match="No videos"):
        runner.start("empty", "fast")
    runner_with_videos(tmp_path, FakePipeline(), room="k2")
    with pytest.raises(UserError, match="Unknown preset"):
        runner.start("k2", "ultra")


def test_unexpected_error_is_kept_for_the_page(tmp_path):
    def broken(videos, run_dir, config):
        raise RuntimeError("colmap not found")

    runner = runner_with_videos(tmp_path, broken)
    runner.start("kitchen", "fast")
    runner.wait(30)
    assert runner.errors["kitchen"] == "colmap not found"


def test_ui_overrides():
    overrides = ui_overrides({"train": {"max_splats": 500000}}, gpu_colmap=True)
    assert overrides["train"] == {
        "trainer": "brush",
        "max_splats": 500000,
        "max_resolution": 1280,
    }
    assert overrides["poses"] == {"use_gpu": True}
    assert "poses" not in ui_overrides(gpu_colmap=False)


# ---- viewer ---------------------------------------------------------------------------


def test_copy_to_viewer(tmp_path):
    run_dir = tmp_path / "runs" / "kitchen"
    run_dir.mkdir(parents=True)
    (run_dir / "scene.ply").write_bytes(b"ply")
    (run_dir / "cameras.json").write_text("{}")
    viewer = tmp_path / "3DViewing"
    (viewer / "public").mkdir(parents=True)

    url = copy_to_viewer(run_dir, viewer, "kitchen")
    assert url == "http://localhost:5173/?run=runs/kitchen"
    assert (viewer / "public" / "runs" / "kitchen" / "scene.ply").read_bytes() == b"ply"
    assert (viewer / "public" / "runs" / "kitchen" / "cameras.json").is_file()


def test_copy_to_viewer_needs_results_and_viewer(tmp_path):
    with pytest.raises(UserError, match="no scene.ply"):
        copy_to_viewer(tmp_path, tmp_path, "kitchen")
    (tmp_path / "scene.ply").write_bytes(b"ply")
    (tmp_path / "cameras.json").write_text("{}")
    with pytest.raises(UserError, match="No viewer"):
        copy_to_viewer(tmp_path, tmp_path / "nothing", "kitchen")


# ---- HTTP server -----------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    app = App(Runner(tmp_path, run_fn=FakePipeline()), None, port=0)
    httpd = make_server(app, 0)
    port = httpd.server_address[1]
    app.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    app.viewer_running = lambda: False  # don't probe port 5173 in tests
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield port, tmp_path
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_server_listens_on_localhost_only(server):
    port, _ = server
    status, page = request(port, "GET", "/")
    assert status == 200 and b"Room pipeline" in page
    from colibrimo_pipeline.ui.server import HOST

    assert HOST == "127.0.0.1"


def test_server_rejects_other_hosts_and_origins(server):
    port, _ = server
    assert (
        request(port, "GET", "/api/state", headers={"Host": "evil.example:80"})[0]
        == 403
    )
    status, _ = request(
        port,
        "POST",
        "/api/start",
        body=b"{}",
        headers={"Origin": "http://evil.example", "Content-Type": "application/json"},
    )
    assert status == 403


def test_upload_then_state_then_start(server):
    port, data_dir = server
    status, body = request(
        port, "PUT", "/api/upload?room=my%20room&file=bottom.mp4", body=b"x" * 1000
    )
    assert status == 201 and json.loads(body) == {"saved": "bottom.mp4", "size": 1000}
    assert (data_dir / "uploads" / "my_room" / "bottom.mp4").stat().st_size == 1000

    status, body = request(port, "GET", "/api/state?room=my_room")
    state = json.loads(body)
    assert state["rooms"] == ["my_room"]
    assert state["room"]["videos"] == [{"name": "bottom.mp4", "size": 1000}]
    assert state["room"]["progress"] is None

    status, body = request(
        port,
        "POST",
        "/api/start",
        body=json.dumps({"room": "my_room", "preset": "fast"}),
    )
    assert status == 200


def test_upload_refuses_non_videos(server):
    port, data_dir = server
    status, body = request(
        port, "PUT", "/api/upload?room=k&file=..%2F..%2Frun.exe", body=b"x"
    )
    assert status == 400 and "not a video" in json.loads(body)["error"]
    assert not (data_dir / "uploads").exists()
