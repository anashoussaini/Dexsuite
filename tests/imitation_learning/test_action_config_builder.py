"""Test robomimic action config generation from DexSuite metadata."""

import json

import pytest

from scripts.imitation_learning.robomimic.utils.config_builder import (
    build_action_settings_from_json,
)


def _write_action_config(tmp_path, payload):
    path = tmp_path / "action_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _action_payload():
    return {
        "action_keys": [
            "action/eef_pos",
            "action/eef_quat",
            "action/gripper",
        ],
        "segments": [
            {
                "parts": [
                    {
                        "robomimic_id": "action/eef_pos",
                        "normalized_in_env": False,
                        "recommended_normalization": "min_max",
                    },
                    {
                        "robomimic_id": "action/eef_quat",
                        "normalized_in_env": False,
                        "recommended_normalization": "none",
                        "rot_format": "quat_wxyz",
                    },
                    {
                        "robomimic_id": "action/gripper",
                        "normalized_in_env": False,
                        "recommended_normalization": "min_max",
                    },
                ],
            },
        ],
    }


def test_build_action_settings_uses_dataset_normalization(tmp_path):
    path = _write_action_config(tmp_path, _action_payload())

    action_keys, action_config = build_action_settings_from_json(str(path))

    assert action_keys == ["action/eef_pos", "action/eef_quat", "action/gripper"]
    assert action_config == {
        "action/eef_pos": {"normalization": "min_max", "rot_conversion": None},
        "action/eef_quat": {"normalization": None, "rot_conversion": None},
        "action/gripper": {"normalization": "min_max", "rot_conversion": None},
    }


def test_build_action_settings_can_disable_normalization(tmp_path):
    path = _write_action_config(tmp_path, _action_payload())

    _action_keys, action_config = build_action_settings_from_json(
        str(path),
        action_normalization="none",
    )

    assert action_config == {
        "action/eef_pos": {"normalization": None, "rot_conversion": None},
        "action/eef_quat": {"normalization": None, "rot_conversion": None},
        "action/gripper": {"normalization": None, "rot_conversion": None},
    }


def test_build_action_settings_prefers_exact_robomimic_config(tmp_path):
    payload = _action_payload()
    payload["robomimic_action_config"] = {
        "action/eef_pos": {"normalization": "gaussian"},
        "action/eef_quat": {"normalization": None},
        "action/gripper": {"normalization": "min_max"},
    }
    path = _write_action_config(tmp_path, payload)

    _action_keys, action_config = build_action_settings_from_json(str(path))

    assert action_config["action/eef_pos"]["normalization"] == "gaussian"
    assert action_config["action/eef_quat"]["normalization"] is None
    assert action_config["action/gripper"]["normalization"] == "min_max"


def test_build_action_settings_rejects_unknown_normalization(tmp_path):
    payload = _action_payload()
    payload["segments"][0]["parts"][0]["recommended_normalization"] = "bad"
    path = _write_action_config(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported action normalization"):
        build_action_settings_from_json(str(path))
