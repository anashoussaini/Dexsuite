"""Test robomimic training CLI design helpers."""

from types import SimpleNamespace

import pytest

from scripts.imitation_learning.robomimic import train


def _args(**overrides):
    args = SimpleNamespace(
        action_normalization=None,
        algo=None,
        architecture=None,
        act_dropout=None,
        act_embed_dim=None,
        act_lr=None,
        act_num_heads=None,
        act_num_layers=None,
        action_horizon=None,
        cameras=None,
        config="bc",
        dataset=None,
        dataset_name=None,
        debug=False,
        h5_basename=None,
        no_images=False,
        name=None,
        obs=None,
        obs_horizon=None,
        obs_profile=None,
        output_dir=None,
        policy_preset=None,
        repo_id=None,
        revision=None,
        sensors=None,
        task=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_config_arg_selects_generic_bc_recipe():
    args = _args(config="bc")

    preset = train._load_policy_preset(args)

    assert args.policy_preset == "bc"
    assert preset["algo_name"] == "bc"


def test_dataset_arg_can_select_registry_alias():
    args = _args(dataset="lift_auto")
    preset = train._load_policy_preset(args)

    spec = train._spec_from_args(args, preset)

    assert spec.name == "lift_auto"
    assert spec.repo_id == "dexsuite/lift_auto"
    assert spec.obs_profile == "lowdim"


def test_dataset_arg_can_use_direct_huggingface_repo_id():
    args = _args(
        dataset="Helium-bre/pick_place_pan_franka_allegro",
        sensors="state",
    )
    preset = train._load_policy_preset(args)

    spec = train._spec_from_args(args, preset)

    assert spec.name == "Helium-bre/pick_place_pan_franka_allegro"
    assert spec.repo_id == "Helium-bre/pick_place_pan_franka_allegro"
    assert spec.obs_profile == "lowdim"


def test_lift_auto_alias_selects_camera_dataset_defaults():
    args = _args(dataset="lift_auto")
    preset = train._load_policy_preset(args)

    spec = train._spec_from_args(args, preset)

    assert spec.name == "lift_auto"
    assert spec.repo_id == "dexsuite/lift_auto"
    assert spec.task == "lift"
    assert spec.obs_profile == "lowdim"
    assert spec.camera_names == ()
    assert train._camera_names_for_profile(args, spec.camera_names, spec.obs_profile) is None
    assert spec.no_images is True


def test_dataset_arg_treats_hdf5_file_as_converted_dataset(tmp_path):
    hdf5_path = tmp_path / "demo.hdf5"
    hdf5_path.touch()
    args = _args(dataset=str(hdf5_path))

    assert train._spec_from_args(args, None) is None
    assert train._dataset_file_path(args.dataset) == hdf5_path.resolve()


def test_sensors_override_selects_camera_profile():
    args = _args(sensors="cameras")

    obs_profile = train._dataset_obs_profile(args, {}, None)

    assert obs_profile == "cameras"


def test_camera_sensors_select_robot_proprio_low_dim():
    args = _args(sensors="cameras")

    assert train._low_dim_selection_from_sensors(args, "cameras") == "proprio"


def test_camera_names_default_to_shared_observation_config():
    args = _args(sensors="cameras")

    assert train._camera_names_for_profile(args, (), "cameras") == ("front", "wrist")


def test_camera_names_accept_comma_separated_cli_values():
    args = _args(sensors="cameras", cameras=["front,wrist"])

    assert train._camera_names_for_profile(args, (), "cameras") == ("front", "wrist")


def test_state_sensors_select_task_state_low_dim():
    args = _args(sensors="state")

    assert train._low_dim_selection_from_sensors(args, "lowdim") == "state"


def test_legacy_config_name_selects_architecture():
    assert train._legacy_architecture_name("bc_rnn") == "rnn"
    assert train._legacy_architecture_name("bc_transformer") == "transformer"
    assert train._legacy_architecture_name("bc_mlp") == "mlp"


def test_json_config_name_resolves_checked_in_base_template():
    path = train._config_file_path("bc.json")

    assert path.name == "bc.json"
    assert path.parent.name == "base_configs"


def test_act_recipe_applies_horizon_and_transformer_overrides():
    args = _args(
        config="act",
        obs_horizon=4,
        action_horizon=3,
        act_embed_dim=128,
        act_num_layers=2,
        act_num_heads=4,
        act_dropout=0.05,
        act_lr=1e-5,
    )
    preset = train._load_policy_preset(args)

    config, generated_path = train.build_config(args, None, preset, None)

    assert generated_path is None
    assert config.algo_name == "act"
    assert config.train.frame_stack == 4
    assert config.train.seq_length == 3
    assert config.train.hdf5_load_next_obs is False
    assert config.train.dataset_keys == ["actions"]
    assert config.algo.horizon.observation_horizon == 4
    assert config.algo.horizon.action_horizon == 3
    assert config.algo.transformer.embed_dim == 128
    assert config.algo.transformer.num_layers == 2
    assert config.algo.transformer.num_heads == 4
    assert config.algo.transformer.emb_dropout == 0.05
    assert config.algo.optim_params.policy.learning_rate.initial == 1e-5


def test_bc_recipe_applies_training_and_rollout_overrides():
    args = _args(
        config="bc",
        bc_gaussian=True,
        batch_size=100,
        epoch_every_n_steps=100,
        hdf5_cache_mode="all",
        hdf5_normalize_obs=True,
        lr=1e-4,
        num_epochs=200,
        render_video=True,
        rollout_enabled=True,
        rollout_horizon=400,
        rollout_n=5,
        rollout_rate=25,
        rollout_terminate_on_success=True,
        rollout_warmstart=0,
        save_models=False,
        train_dataset_keys=["actions", "rewards", "dones"],
        validation_epoch_every_n_steps=10,
        video_skip=5,
    )
    preset = train._load_policy_preset(args)

    config, generated_path = train.build_config(args, None, preset, None)

    assert generated_path is None
    assert config.algo_name == "bc"
    assert config.algo.gaussian.enabled is True
    assert config.algo.gmm.enabled is False
    assert config.train.num_epochs == 200
    assert config.train.batch_size == 100
    assert config.train.hdf5_cache_mode == "all"
    assert config.train.hdf5_normalize_obs is True
    assert config.train.dataset_keys == ["actions", "rewards", "dones"]
    assert config.algo.optim_params.policy.learning_rate.initial == 1e-4
    assert config.experiment.save.enabled is False
    assert config.experiment.epoch_every_n_steps == 100
    assert config.experiment.validation_epoch_every_n_steps == 10
    assert config.experiment.render_video is True
    assert config.experiment.video_skip == 5
    assert config.experiment.rollout.enabled is True
    assert config.experiment.rollout.n == 5
    assert config.experiment.rollout.horizon == 400
    assert config.experiment.rollout.rate == 25
    assert config.experiment.rollout.warmstart == 0
    assert config.experiment.rollout.terminate_on_success is True


def test_checkpoint_saving_patch_skips_latest_checkpoint_backup(tmp_path):
    calls = []

    def save_model(*_args, **kwargs):
        calls.append(("save", kwargs.get("ckpt_path")))

    def copyfile(src, dst, *_args, **_kwargs):
        calls.append(("copy", src, dst))
        return dst

    train_module = SimpleNamespace(
        TrainUtils=SimpleNamespace(save_model=save_model),
        shutil=SimpleNamespace(copyfile=copyfile),
    )
    latest_path = tmp_path / "model_latest.pth"
    backup_path = tmp_path / "model_latest_backup.pth"
    robomimic_latest_path = tmp_path / "last.pth"
    robomimic_backup_path = tmp_path / "last_backup.pth"

    with train._checkpoint_saving_patch(enabled=False, train_module=train_module):
        train_module.TrainUtils.save_model(ckpt_path=str(latest_path))
        assert train_module.shutil.copyfile(str(latest_path), str(backup_path)) == str(
            backup_path,
        )
        assert train_module.shutil.copyfile(
            str(robomimic_latest_path),
            str(robomimic_backup_path),
        ) == str(robomimic_backup_path)

    assert calls == []


def test_dataloader_cpu_generator_patch_handles_cuda_default_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to reproduce the DataLoader generator mismatch.")

    from torch.utils.data import DataLoader, TensorDataset

    torch.set_default_device("cuda")
    try:
        with train._dataloader_cpu_generator_patch():
            generator = torch.Generator(device="cuda")
            dataset = TensorDataset(torch.arange(4, device="cpu"))
            loader = DataLoader(
                dataset, batch_size=2, generator=generator, num_workers=0
            )
            batch = next(iter(loader))
    finally:
        torch.set_default_device("cpu")

    assert generator.device.type == "cpu"
    assert batch[0].tolist() == [0, 1]
