from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.imitation_learning.robomimic.utils import converter

h5py = pytest.importorskip("h5py")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_lerobot_root(root) -> None:
    _write_json(
        root / "meta" / "env_config.json",
        {
            "task": "lift",
            "options": {
                "robot": {
                    "single": {
                        "gripper_controller": {
                            "name": "joint_position",
                            "config": {"normalized": False},
                        },
                    },
                },
            },
        },
    )
    _write_json(
        root / "meta" / "action_layout.json",
        {
            "total_dim": 2,
            "segments": [
                {
                    "name": "gripper",
                    "start": 0,
                    "stop": 2,
                    "width": 2,
                    "ctrl_type": "joint_position",
                },
            ],
        },
    )
    _write_json(
        root / "meta" / "info.json",
        {
            "features": {
                "obs.state.other.cube_pos": {"dtype": "float32", "shape": [3]},
                "obs.cameras.front.depth": {"dtype": "float32", "shape": [2, 2]},
            },
        },
    )

    table = pa.table(
        {
            "episode_index": [0, 0, 1],
            "action": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "reward": [0.0, 1.0, 1.0],
            "success": [False, True, True],
            "truncated": [False, False, False],
            "obs.state.other.cube_pos": [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.2, 0.3, 0.4], dtype=np.float32),
                np.array([0.3, 0.4, 0.5], dtype=np.float32),
            ],
            "obs.cameras.front.depth": [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
        },
    )
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_dir / "file-000.parquet")


def test_local_no_image_conversion_streams_parquet_without_lerobot(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path / "owner__lift_auto"
    output_dir = tmp_path / "hdf5"
    _write_lerobot_root(root)

    def fail_if_lerobot_dataset_is_used(*_args, **_kwargs):
        raise AssertionError("LeRobotDataset should not be used for local no-image conversion.")

    monkeypatch.setattr(
        converter,
        "_build_dataset_with_fallback",
        fail_if_lerobot_dataset_is_used,
    )

    converter.convert_lerobot_to_hdf5(
        repo_or_root=str(root),
        output_path=str(output_dir),
        h5_basename="lift_auto",
        no_images=True,
        block_size=2,
    )

    with h5py.File(output_dir / "lift_auto.hdf5", "r") as file:
        assert set(file["data"].keys()) == {"demo_0", "demo_1"}
        assert "state_other_cube_pos" in file["data/demo_0/obs"]
        assert "cameras_front_depth" not in file["data/demo_0/obs"]
        assert file["data/demo_0/obs/state_other_cube_pos"].dtype == np.float32
        np.testing.assert_allclose(
            file["data/demo_0/actions"][:],
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        )
