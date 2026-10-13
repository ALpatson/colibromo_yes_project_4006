"""cameras.json: COLMAP poses in scene.ply's coordinates (identity or gsplat transform)."""

import struct

import numpy as np
import pytest

from colibrimo_pipeline.utils.cameras import build_cameras, split_similarity
from colibrimo_pipeline.utils.colmap_model import (
    qvec_to_rotation,
    read_cameras,
    read_images,
    rotation_to_qvec,
)


def rotation(axis, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def write_model(folder, poses):
    """COLMAP binary model (format: colmap.github.io/format.html).

    ``poses``: list of (name, world_from_cam 3x3, centre 3) for one SIMPLE_RADIAL camera.
    """
    folder.mkdir(parents=True)
    cams = struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 2, 1600, 900)
    cams += struct.pack("<4d", 785.3, 800.0, 450.0, -0.0024)
    (folder / "cameras.bin").write_bytes(cams)

    images = struct.pack("<Q", len(poses))
    for i, (name, world_from_cam, centre) in enumerate(poses, start=1):
        r = world_from_cam.T  # camera-from-world
        t = -r @ centre
        images += struct.pack("<I7dI", i, *rotation_to_qvec(r), *t, 1)
        images += name.encode() + b"\0"
        images += struct.pack("<Q", 2) + struct.pack(
            "<ddqddq", 1.0, 2.0, -1, 3.0, 4.0, 7
        )
    (folder / "images.bin").write_bytes(images)


# Camera looking along +x (forward), upright (COLMAP up = -y), at (1, 2, 3).
# Columns = camera x (right), y (down), z (forward) in world coordinates.
LOOK_X = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)
assert np.isclose(np.linalg.det(LOOK_X), 1)  # a rotation, not a reflection
POSES = [
    ("middle_00000.jpg", LOOK_X, np.array([1.0, 2.0, 3.0])),
    ("bottom_00000.jpg", rotation("y", 30) @ LOOK_X, np.array([-1.0, 2.5, 0.0])),
    ("up_00001.jpg", rotation("x", 20) @ LOOK_X, np.array([0.5, 1.0, -2.0])),
]


def test_readers(tmp_path):
    write_model(tmp_path / "0", POSES)
    cameras = read_cameras(tmp_path / "0" / "cameras.bin")
    assert cameras[1]["model"] == "SIMPLE_RADIAL"
    assert (cameras[1]["width"], cameras[1]["height"]) == (1600, 900)
    assert cameras[1]["params"][0] == pytest.approx(785.3)
    images = read_images(tmp_path / "0" / "images.bin")
    assert [i["name"] for i in images] == [p[0] for p in POSES]


@pytest.mark.parametrize("seed", range(5))
def test_quaternion_round_trip(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    q = q if q[0] >= 0 else -q
    np.testing.assert_allclose(rotation_to_qvec(qvec_to_rotation(q)), q, atol=1e-9)


def test_quaternion_half_turn():
    r = rotation("x", 180)
    np.testing.assert_allclose(qvec_to_rotation(rotation_to_qvec(r)), r, atol=1e-9)


def test_identity_transform_keeps_colmap_poses(tmp_path):
    write_model(tmp_path / "0", POSES)
    frames = [
        {"file": "middle_00000.jpg", "video": "middle", "index": 0, "time_s": 1.5}
    ]
    data = build_cameras(tmp_path / "0", np.eye(4), frames)

    assert data["num_cameras"] == 3
    assert data["videos"] == {"bottom": 1, "middle": 1, "up": 1}
    assert data["intrinsics"]["1"]["model"] == "SIMPLE_RADIAL"
    by_name = {c["name"]: c for c in data["cameras"]}
    middle = by_name["middle_00000.jpg"]
    assert middle["position"] == pytest.approx([1, 2, 3])
    assert middle["forward"] == pytest.approx([1, 0, 0])
    assert middle["up"] == pytest.approx([0, 1, 0])
    assert (middle["video"], middle["frame_index"], middle["time_s"]) == (
        "middle",
        0,
        1.5,
    )
    # No frames.json entry: video taken from the file name.
    assert (by_name["up_00001.jpg"]["video"], by_name["up_00001.jpg"]["time_s"]) == (
        "up",
        None,
    )
    np.testing.assert_allclose(
        qvec_to_rotation(middle["rotation_wxyz"]), LOOK_X, atol=1e-6
    )


def test_gsplat_transform_is_applied_consistently(tmp_path):
    """A point seen by a camera must stay in front of it at the scaled distance."""
    write_model(tmp_path / "0", POSES)
    scale, rot, shift = (
        0.25,
        rotation("z", 90) @ rotation("x", -30),
        np.array([3.0, -1, 2]),
    )
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = shift
    data = build_cameras(tmp_path / "0", matrix)

    for (name, world_from_cam, centre), cam in zip(
        sorted(POSES, key=lambda p: p[0]), data["cameras"], strict=True
    ):
        assert cam["name"] == name
        point = centre + 2.0 * world_from_cam[:, 2]  # 2 units in front of the camera
        point_scene = matrix[:3, :3] @ point + shift
        position = np.array(cam["position"])
        np.testing.assert_allclose(position, matrix[:3, :3] @ centre + shift, atol=1e-6)
        np.testing.assert_allclose(
            point_scene, position + 2.0 * scale * np.array(cam["forward"]), atol=1e-5
        )
        np.testing.assert_allclose(np.linalg.norm(cam["up"]), 1, atol=1e-6)
        r = qvec_to_rotation(cam["rotation_wxyz"])
        np.testing.assert_allclose(r[:, 2], cam["forward"], atol=1e-6)
        np.testing.assert_allclose(-r[:, 1], cam["up"], atol=1e-6)


def test_split_similarity_rejects_non_uniform_scale():
    with pytest.raises(ValueError, match="rotation \\+ uniform scale"):
        split_similarity(np.diag([1.0, 2.0, 1.0, 1.0]))
    scale, rot, shift = split_similarity(np.diag([2.0, 2.0, 2.0, 1.0]))
    assert scale == pytest.approx(2) and np.allclose(rot, np.eye(3))
