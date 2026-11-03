from __future__ import annotations

import pytest
import yaml

from scripts.imitation_learning.robomimic.utils.obs_profiles import (
    ObservationProfile,
    default_camera_names,
    hdf5_has_visual_observations,
    infer_observation_profile,
    resolve_observation_yaml,
    task_low_dim_keys,
    write_observation_profile_yaml,
)

h5py = pytest.importorskip("h5py")


def _write_hdf5(path, obs_shapes) -> None:
    with h5py.File(path, "w") as file:
        obs = file.create_group("data/demo_0/obs")
        for key, shape, dtype in obs_shapes:
            obs.create_dataset(key, shape=shape, dtype=dtype)


def _read_yaml(path):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_lowdim_profile_excludes_camera_and_depth_keys(tmp_path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("joint_matrix", (5, 2, 2), "float32"),
            ("front_rgb", (5, 64, 64, 3), "uint8"),
            ("front_depth", (5, 64, 64), "float32"),
        ],
    )

    profile = infer_observation_profile(hdf5_path, "lowdim")

    assert profile.low_dim == ("eef_pos", "joint_matrix")
    assert profile.rgb == ()
    assert profile.depth == ()
    assert profile.proprio == ("eef_pos", "joint_matrix")
    assert profile.state == ()


def test_lowdim_profile_excludes_configured_keys(tmp_path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("runtime_state", (5, 4), "float32"),
            ("logged_only_state", (5, 2), "float32"),
        ],
    )

    profile = infer_observation_profile(
        hdf5_path,
        "lowdim",
        exclude_keys={"logged_only_state"},
    )

    assert profile.low_dim == ("runtime_state",)
    assert profile.state == ("runtime_state",)


def test_cameras_profile_groups_obs_by_modality(tmp_path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("front_depth", (5, 64, 64), "float32"),
            ("wrist_image", (5, 64, 64, 3), "uint8"),
        ],
    )

    profile = infer_observation_profile(hdf5_path, "cameras")

    assert profile.low_dim == ("eef_pos",)
    assert profile.rgb == ("wrist_image",)
    assert profile.depth == ("front_depth",)
    assert profile.proprio == ("eef_pos",)
    assert profile.state == ()


def test_hdf5_visual_detection_distinguishes_lowdim_and_camera_data(tmp_path) -> None:
    lowdim_path = tmp_path / "lowdim.hdf5"
    camera_path = tmp_path / "camera.hdf5"
    _write_hdf5(lowdim_path, [("eef_pos", (5, 3), "float32")])
    _write_hdf5(
        camera_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("front_rgb", (5, 32, 32, 3), "uint8"),
        ],
    )

    assert hdf5_has_visual_observations(lowdim_path) is False
    assert hdf5_has_visual_observations(camera_path) is True


def test_cameras_profile_filters_requested_cameras(tmp_path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("front_rgb", (5, 64, 64, 3), "uint8"),
            ("wrist_rgb", (5, 64, 64, 3), "uint8"),
        ],
    )

    profile = infer_observation_profile(
        hdf5_path,
        "cameras",
        camera_names=["front"],
    )

    assert profile.low_dim == ("eef_pos",)
    assert profile.rgb == ("front_rgb",)
    assert profile.depth == ()


def test_cameras_profile_requires_visual_obs(tmp_path) -> None:
    hdf5_path = tmp_path / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("state_matrix", (5, 2, 2), "float32"),
        ],
    )

    with pytest.raises(ValueError, match="contains no image or depth"):
        infer_observation_profile(hdf5_path, "cameras")


def test_resolve_observation_yaml_generates_dataset_local_profile(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("front_rgb", (5, 32, 32, 3), "uint8"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="cameras",
    )

    assert obs_yaml == hdf5_dir / "dataset_obs_cameras.yaml"
    assert _read_yaml(obs_yaml) == {
        "obs": {
            "low_dim": ["eef_pos"],
            "rgb": ["front_rgb"],
            "depth": [],
        },
        "groups": {
            "proprio": ["eef_pos"],
            "state": [],
        },
    }


def test_resolve_observation_yaml_can_select_proprio_low_dim(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("object_pose", (5, 7), "float32"),
            ("front_rgb", (5, 32, 32, 3), "uint8"),
            ("wrist_rgb", (5, 32, 32, 3), "uint8"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="cameras",
        camera_names=["front"],
        low_dim_selection="proprio",
    )

    assert obs_yaml == hdf5_dir / "dataset_obs_cameras_proprio_front.yaml"
    assert _read_yaml(obs_yaml) == {
        "obs": {
            "low_dim": ["eef_pos"],
            "rgb": ["front_rgb"],
            "depth": [],
        },
        "groups": {
            "proprio": ["eef_pos"],
            "state": ["object_pose"],
        },
    }


def test_resolve_lowdim_yaml_ignores_camera_names_in_filename(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("eef_pos", (5, 3), "float32"),
            ("front_rgb", (5, 32, 32, 3), "uint8"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="lowdim",
        camera_names=["front", "wrist"],
    )

    assert obs_yaml == hdf5_dir / "dataset_obs_lowdim.yaml"


def test_state_lowdim_selection_keeps_policy_state_only(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("state_gripper_qpos", (5, 2), "float32"),
            ("state_gripper_qvel", (5, 2), "float32"),
            ("state_gripper_tactile_tactile_points", (5, 768, 4), "float32"),
            ("state_manipulator_eef_pos", (5, 1, 3), "float32"),
            ("state_manipulator_eef_quat", (5, 1, 4), "float32"),
            ("state_manipulator_qpos", (5, 7), "float32"),
            ("state_other_action", (5, 9), "float32"),
            ("state_other_cube_pos", (5, 3), "float32"),
            ("state_other_cube_quat", (5, 4), "float32"),
            ("state_other_last_action", (5, 9), "float32"),
            ("state_other_target_pos", (5, 3), "float32"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="lowdim",
        low_dim_selection="state",
    )

    assert obs_yaml == hdf5_dir / "dataset_obs_lowdim_state.yaml"
    assert _read_yaml(obs_yaml)["obs"]["low_dim"] == [
        "state_manipulator_eef_pos",
        "state_gripper_qpos",
        "state_other_cube_pos",
        "state_other_target_pos",
    ]


def test_task_lowdim_selection_uses_paper_lift_profile(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("state_gripper_qpos", (5, 2), "float32"),
            ("state_gripper_tactile_tactile_points", (5, 768, 4), "float32"),
            ("state_manipulator_eef_pos", (5, 1, 3), "float32"),
            ("state_manipulator_eef_quat", (5, 1, 4), "float32"),
            ("state_manipulator_qpos", (5, 7), "float32"),
            ("state_other_action", (5, 9), "float32"),
            ("state_other_cube_pos", (5, 3), "float32"),
            ("state_other_cube_quat", (5, 4), "float32"),
            ("state_other_last_action", (5, 9), "float32"),
            ("state_other_target_pos", (5, 3), "float32"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="lowdim",
        low_dim_selection="state",
        task="lift",
    )

    assert _read_yaml(obs_yaml)["obs"]["low_dim"] == [
        "state_manipulator_eef_pos",
        "state_gripper_qpos",
        "state_other_cube_pos",
        "state_other_target_pos",
    ]


def test_camera_profile_uses_robot_subset_of_task_profile(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("front_rgb", (5, 32, 32, 3), "uint8"),
            ("state_gripper_qpos", (5, 2), "float32"),
            ("state_gripper_tactile_tactile_points", (5, 768, 4), "float32"),
            ("state_manipulator_eef_pos", (5, 1, 3), "float32"),
            ("state_manipulator_eef_quat", (5, 1, 4), "float32"),
            ("state_other_cube_pos", (5, 3), "float32"),
            ("state_other_cube_quat", (5, 4), "float32"),
            ("state_other_target_pos", (5, 3), "float32"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="cameras",
        camera_names=["front"],
        low_dim_selection="proprio",
        task="lift",
    )

    assert _read_yaml(obs_yaml)["obs"] == {
        "low_dim": [
            "state_manipulator_eef_pos",
            "state_gripper_qpos",
        ],
        "rgb": ["front_rgb"],
        "depth": [],
    }


def test_task_lowdim_selection_uses_paper_pan_profile(tmp_path) -> None:
    hdf5_dir = tmp_path / "hdf5"
    hdf5_dir.mkdir()
    hdf5_path = hdf5_dir / "dataset.hdf5"
    _write_hdf5(
        hdf5_path,
        [
            ("state_gripper_qpos", (5, 16), "float32"),
            ("state_gripper_tactile_tactile_points", (5, 768, 4), "float32"),
            ("state_manipulator_eef_pos", (5, 1, 3), "float32"),
            ("state_manipulator_eef_quat", (5, 1, 4), "float32"),
            ("state_manipulator_qvel", (5, 7), "float32"),
            ("state_other_action", (5, 23), "float32"),
            ("state_other_mat_inner_hi", (5, 3), "float32"),
            ("state_other_mat_inner_lo", (5, 3), "float32"),
            ("state_other_mat_top_z", (5, 1), "float32"),
            ("state_other_pan_ang", (5, 3), "float32"),
            ("state_other_pan_pos", (5, 3), "float32"),
            ("state_other_pan_quat", (5, 4), "float32"),
            ("state_other_pan_vel", (5, 3), "float32"),
        ],
    )

    obs_yaml = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=None,
        obs_profile="lowdim",
        low_dim_selection="state",
        task="pick_place_pan",
    )

    assert _read_yaml(obs_yaml)["obs"]["low_dim"] == [
        "state_manipulator_eef_pos",
        "state_gripper_qpos",
        "state_other_pan_pos",
        "state_other_pan_vel",
        "state_other_mat_top_z",
    ]


def test_observation_config_exposes_default_cameras_and_task_keys() -> None:
    assert default_camera_names() == ("front", "wrist")
    assert task_low_dim_keys("lift") == (
        "state_manipulator_eef_pos",
        "state_gripper_qpos",
        "state_other_cube_pos",
        "state_other_target_pos",
    )


def test_write_observation_profile_yaml_preserves_schema(tmp_path) -> None:
    out_path = write_observation_profile_yaml(
        ObservationProfile(
            low_dim=("eef_pos",),
            rgb=("front_rgb",),
            depth=("front_depth",),
            proprio=("eef_pos",),
            state=(),
        ),
        tmp_path / "obs.yaml",
    )

    assert _read_yaml(out_path) == {
        "obs": {
            "low_dim": ["eef_pos"],
            "rgb": ["front_rgb"],
            "depth": ["front_depth"],
        },
        "groups": {
            "proprio": ["eef_pos"],
            "state": [],
        },
    }
