import pytest

from colibrimo_pipeline.config import (
    available_presets,
    deep_merge,
    load_config,
    parse_set_overrides,
)
from colibrimo_pipeline.errors import ConfigError


def test_builtin_presets_exist_and_load():
    assert available_presets() == ["balanced", "fast", "quality"]
    for name in available_presets():
        config = load_config(name)
        assert config.preset == name


def test_presets_differ_in_frames_and_steps():
    fast, balanced, quality = (load_config(n) for n in ("fast", "balanced", "quality"))
    assert (
        fast.frames.target_count
        < balanced.frames.target_count
        < quality.frames.target_count
    )
    assert fast.train.max_steps < balanced.train.max_steps


def test_overrides_are_applied_on_top_of_preset():
    config = load_config(
        "fast", {"frames": {"target_count": 42}, "poses": {"matcher": "exhaustive"}}
    )
    assert config.frames.target_count == 42
    assert config.frames.max_long_side == 1280  # untouched value from the preset
    assert config.poses.matcher == "exhaustive"


def test_custom_preset_file(tmp_path):
    preset = tmp_path / "mine.yaml"
    preset.write_text("train:\n  max_steps: 500\n", encoding="utf-8")
    config = load_config(preset)
    assert config.preset == "mine"
    assert config.train.max_steps == 500


def test_unknown_preset_is_rejected():
    with pytest.raises(ConfigError, match="Unknown preset"):
        load_config("does-not-exist")


@pytest.mark.parametrize(
    "overrides",
    [
        {"frames": {"target_count": 1}},  # below minimum
        {"poses": {"matcher": "magic"}},  # not an allowed value
        {"frames": {"typo_field": 3}},  # unknown key
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config("balanced", overrides)


def test_parse_set_overrides_types_and_nesting():
    result = parse_set_overrides(
        [
            "frames.target_count=200",
            "frames.blur_filter=false",
            "poses.matcher=exhaustive",
        ]
    )
    assert result == {
        "frames": {"target_count": 200, "blur_filter": False},
        "poses": {"matcher": "exhaustive"},
    }


def test_parse_set_overrides_rejects_missing_equals():
    with pytest.raises(ConfigError):
        parse_set_overrides(["frames.target_count"])


def test_deep_merge_does_not_modify_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 10}})
    assert merged == {"a": {"b": 10, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
