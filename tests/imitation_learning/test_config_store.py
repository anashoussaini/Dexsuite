"""Test DexSuite robomimic preset loading."""

from types import SimpleNamespace

from scripts.imitation_learning.robomimic.utils import config_store


def test_dataset_registry_contains_integrated_dexsuite_datasets():
    registry = config_store.load_dataset_registry()

    assert list(registry) == [
        "reach_franka_robotiq",
        "pick_and_place_auto",
        "lift_auto",
        "pick_place_pan_franka_allegro",
    ]


def test_dataset_registry_contains_lift_auto_camera_dataset():
    registry = config_store.load_dataset_registry()

    spec = registry["lift_auto"]

    assert spec["repo_id"] == "dexsuite/lift_auto"
    assert spec["task"] == "lift"
    assert spec["default_sensors"] == "state"
    assert "camera_names" not in spec
    assert spec["no_images"] is True


def test_policy_preset_reports_algorithm():
    preset = config_store.load_policy_preset("bc")

    assert config_store.policy_algo_name(preset) == "bc"
    assert config_store.policy_no_images(preset) is True


def test_legacy_policy_names_resolve_to_generic_recipe():
    preset = config_store.load_policy_preset("bc_resnet_cameras")

    assert config_store.policy_algo_name(preset) == "bc"


def test_rollout_preset_exposes_standalone_defaults():
    preset = config_store.load_rollout_preset("smoke")

    defaults = config_store.rollout_arg_defaults(preset)

    assert defaults["n_rollouts"] == 1
    assert defaults["horizon"] == 10
    assert defaults["camera_names"] == ["front"]


def test_rollout_presets_are_minimal():
    assert config_store.list_rollout_presets() == ["off", "smoke", "standard"]


def test_checked_in_robomimic_base_configs_are_discoverable():
    configs = config_store.list_robomimic_base_configs()

    assert "bc.json" in configs
    assert "bc_transformer.json" in configs
    assert "diffusion_policy.json" in configs
    assert config_store.robomimic_base_config_path("bc.json").name == "bc.json"


def test_apply_preset_config_updates_nested_objects():
    config = SimpleNamespace(
        train=SimpleNamespace(batch_size=1),
        algo=SimpleNamespace(
            transformer=SimpleNamespace(enabled=False),
            gmm=SimpleNamespace(enabled=False),
        ),
    )
    preset = {
        "config": {
            "train": {"batch_size": 32},
            "algo": {
                "transformer": {"enabled": True},
                "gmm": {"enabled": True},
            },
        },
    }

    config_store.apply_preset_config(config, preset)

    assert config.train.batch_size == 32
    assert config.algo.transformer.enabled is True
    assert config.algo.gmm.enabled is True
