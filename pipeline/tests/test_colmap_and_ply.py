import struct

import pytest

from colibrimo_pipeline.utils.colmap_model import (
    find_models,
    parse_help_options,
    pick_option,
    promote_best_model,
    read_entry_count,
)
from colibrimo_pipeline.utils.ply import read_vertex_count


def write_fake_model(folder, num_images, num_points=10):
    folder.mkdir(parents=True)
    (folder / "images.bin").write_bytes(struct.pack("<Q", num_images) + b"rest")
    (folder / "points3D.bin").write_bytes(struct.pack("<Q", num_points))


def test_read_entry_count(tmp_path):
    write_fake_model(tmp_path / "0", num_images=123, num_points=4567)
    assert read_entry_count(tmp_path / "0" / "images.bin") == 123
    assert read_entry_count(tmp_path / "0" / "points3D.bin") == 4567


def test_read_entry_count_rejects_truncated_file(tmp_path):
    bad = tmp_path / "images.bin"
    bad.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError):
        read_entry_count(bad)


def test_promote_best_model_makes_largest_model_zero(tmp_path):
    sparse = tmp_path / "sparse"
    write_fake_model(sparse / "0", num_images=20)
    write_fake_model(sparse / "1", num_images=250)
    write_fake_model(sparse / "2", num_images=5)

    models = promote_best_model(sparse)

    assert [(p.name, n) for p, n in models] == [
        ("0", 250),
        ("unused_1", 20),
        ("unused_2", 5),
    ]
    assert read_entry_count(sparse / "0" / "images.bin") == 250


def test_promote_best_model_no_models(tmp_path):
    (tmp_path / "sparse").mkdir()
    assert promote_best_model(tmp_path / "sparse") == []
    assert find_models(tmp_path / "missing") == []


COLMAP_4_HELP = """
  --database_path arg
  --FeatureExtraction.use_gpu arg (=1)
  --SiftExtraction.max_num_features arg (=8192)
"""
COLMAP_3_HELP = """
  --database_path arg
  --SiftExtraction.use_gpu arg (=1)
  --SiftExtraction.max_num_features arg (=8192)
"""


@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        (COLMAP_4_HELP, "FeatureExtraction.use_gpu"),
        (COLMAP_3_HELP, "SiftExtraction.use_gpu"),
    ],
)
def test_option_names_are_picked_per_colmap_version(help_text, expected):
    options = parse_help_options(help_text)
    assert (
        pick_option(options, ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"])
        == expected
    )
    assert pick_option(options, ["Does.not_exist"]) is None


def test_read_vertex_count(tmp_path):
    ply = tmp_path / "scene.ply"
    ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1234\n"
        b"property float x\nend_header\n\x00\x01"
    )
    assert read_vertex_count(ply) == 1234


def test_read_vertex_count_rejects_non_ply(tmp_path):
    other = tmp_path / "x.ply"
    other.write_bytes(b"not a ply")
    with pytest.raises(ValueError):
        read_vertex_count(other)
