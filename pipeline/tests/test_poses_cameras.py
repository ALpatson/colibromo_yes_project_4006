"""poses: one shared camera for same-looking videos, one camera per video otherwise (D-017)."""

import json
import struct
from pathlib import Path

import pytest

from colibrimo_pipeline.config import load_config
from colibrimo_pipeline.manifest import Manifest
from colibrimo_pipeline.steps import poses
from colibrimo_pipeline.steps.base import StepContext
from colibrimo_pipeline.utils.logs import setup_logging


def make_run(tmp_path, same_camera):
    """A run folder after extract_frames: 2 portrait + 1 landscape video's frames."""
    images = tmp_path / "colmap" / "images"
    images.mkdir(parents=True)
    frames = []
    for video, count in (("bottom", 3), ("middle", 2), ("up", 2)):
        for i in range(count):
            name = f"{video}_{i:05d}.jpg"
            (images / name).write_bytes(b"jpg")
            frames.append({"file": name, "video": video, "index": i})
    (tmp_path / "frames.json").write_text(json.dumps({"frames": frames}))
    manifest = Manifest.load_or_create(tmp_path)
    manifest.mark_succeeded(
        "validate", {}, {"num_videos": 3, "same_camera": same_camera}, 0
    )
    manifest.mark_succeeded("extract_frames", {"images_dir": images}, {}, 0)
    return StepContext(
        tmp_path, [], load_config("fast"), manifest, setup_logging(tmp_path / "logs")
    )


class FakeColmap:
    """Records COLMAP calls; the mapper writes one model holding every frame."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(poses, "find_tool", lambda name: "colmap")
        monkeypatch.setattr(poses, "colmap_version", lambda exe: "COLMAP test")
        monkeypatch.setattr(poses.Colmap, "options", lambda s, command: self.OPTIONS)
        # A plain function on the class receives the Colmap object as first argument.
        monkeypatch.setattr(
            poses.Colmap, "run", lambda colmap, command, args: self.run(command, args)
        )

    OPTIONS = {
        "image_list_path",
        "SequentialMatching.loop_detection",
        "SequentialMatching.overlap",
    }

    def run(self, command, args):
        self.calls.append((command, list(args)))
        if command == "mapper":
            out = args[args.index("--output_path") + 1]
            model = Path(out) / "0"
            model.mkdir(parents=True)
            images = args[args.index("--image_path") + 1]
            count = len(list(Path(images).glob("*.jpg")))
            (model / "images.bin").write_bytes(struct.pack("<Q", count))
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 100))

    def extractor_calls(self):
        return [args for command, args in self.calls if command == "feature_extractor"]


def option(args, name):
    return args[args.index(f"--{name}") + 1] if f"--{name}" in args else None


def test_mixed_videos_get_one_camera_per_video(tmp_path, monkeypatch):
    colmap = FakeColmap(monkeypatch)
    ctx = make_run(tmp_path, same_camera=False)
    result = poses.run(ctx)

    calls = colmap.extractor_calls()
    assert len(calls) == 3  # one feature extraction (= one camera) per video
    lists = [option(args, "image_list_path") for args in calls]
    assert [p.split("\\")[-1].split("/")[-1] for p in lists] == [
        "bottom.txt",
        "middle.txt",
        "up.txt",
    ]
    middle_list = (
        (tmp_path / "colmap" / "image_lists" / "middle.txt").read_text().split()
    )
    assert middle_list == ["middle_00000.jpg", "middle_00001.jpg"]
    assert all(option(args, "ImageReader.single_camera") == "1" for args in calls)
    assert result.info["num_cameras"] == 3
    assert result.info["single_camera"] is False


def test_same_videos_share_one_camera(tmp_path, monkeypatch):
    colmap = FakeColmap(monkeypatch)
    result = poses.run(make_run(tmp_path, same_camera=True))
    (args,) = colmap.extractor_calls()
    assert option(args, "image_list_path") is None  # all frames at once
    assert option(args, "ImageReader.single_camera") == "1"
    assert result.info["num_cameras"] == 1


def test_never_one_camera_per_frame(tmp_path, monkeypatch):
    colmap = FakeColmap(monkeypatch)
    poses.run(make_run(tmp_path, same_camera=False))
    assert all(
        option(args, "ImageReader.single_camera") != "0"
        for args in colmap.extractor_calls()
    )


def test_frames_by_video_without_frames_json(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a_00000.jpg", "a_00001.jpg", "b_c_00000.jpg"):
        (images / name).write_bytes(b"")
    assert poses.frames_by_video(tmp_path, images) == {
        "a": ["a_00000.jpg", "a_00001.jpg"],
        "b_c": ["b_c_00000.jpg"],
    }


@pytest.fixture(autouse=True)
def _close_logs():
    yield
    import logging

    from colibrimo_pipeline.utils.logs import LOGGER_NAME, close_logging

    close_logging(logging.getLogger(LOGGER_NAME))
