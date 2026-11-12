"""Test DexSuite robomimic preset loading."""

from types import SimpleNamespace

from scripts.imitation_learning.robomimic.utils import config_store

BENCHMARK_PAIRS = [  # the 14 human-demonstration pairs of the benchmark (Spray and Toast, the other two, are scripted)
    "reach_franka_robotiq", "reach_franka_allegro",
    "lift_franka_franka_umi_tacticle", "lift_franka_allegro",
    "push_franka_franka_umi_tacticle", "push_franka_allegro",
    "pick_place_franka_franka_umi_tacticle", "pick_place_franka_allegro",
    "stack_franka_franka_umi_tacticle", "stack_franka_allegro",
    "pick_place_mug_franka_allegro", "pick_place_pan_franka_allegro",
    "drill_to_point_franka_allegro", "pick_place_fruit_franka_allegro",
]


def test_dataset_registry_contains_every_dexsuite_dataset():
    registry = config_store.load_dataset_registry()

    expected = (
        BENCHMARK_PAIRS
        + [name + "_mg" for name in BENCHMARK_PAIRS]
        + [f"{task}_franka_franka_umi_tacticle_scripted" for task in ("lift", "push", "pick_place", "stack", "reach")]
        + [f"{task}_floating_sharpa_scripted" for task in ("spray_graffiti", "put_bread_toaster", "close_window_spray")]
        + ["lift_auto", "pick_and_place_auto"]
    )
    assert sorted(registry) == sorted(expected)
    for name, spec in registry.items():
        assert spec["repo_id"] == f"dexsuite/{name}"
        assert spec["task"] and spec["h5_basename"] == name


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
