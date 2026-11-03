"""Test the robomimic ACT against the original ACT's loss and temporal ensembling."""

import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("robomimic")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_act(action_horizon=5, ac_dim=2):
    from robomimic.algo import algo_factory
    from robomimic.config import config_factory
    from robomimic.utils import obs_utils

    from scripts.imitation_learning.robomimic.custom_algorithms import (
        register_custom_algorithms,
    )

    register_custom_algorithms()
    config = config_factory("act")
    with config.values_unlocked():
        config.observation.modalities.obs.low_dim = ["qpos"]
        config.observation.modalities.obs.rgb = []
        config.algo.horizon.action_horizon = action_horizon
        config.train.seq_length = action_horizon
        config.algo.transformer.embed_dim = 32
        config.algo.transformer.num_layers = 1
        config.algo.transformer.feedforward_dim = 64
        config.algo.cvae.latent_dim = 4
        config.algo.cvae.num_encoder_layers = 1
    obs_utils.initialize_obs_utils_with_config(config)
    return algo_factory(
        "act",
        config,
        obs_key_shapes=OrderedDict(qpos=[3]),
        ac_dim=ac_dim,
        device=torch.device("cpu"),
    )


def test_temporal_ensembling_matches_original_act():
    """get_action blends chunks as imitate_episodes.py does (k = 0.01, oldest first)."""
    horizon, ac_dim, steps = 5, 2, 12
    algo = _tiny_act(horizon, ac_dim)
    algo.set_eval()
    algo.reset()
    generator = torch.Generator().manual_seed(0)
    chunks = [
        torch.rand(1, horizon, ac_dim, generator=generator) + 0.1 for _ in range(steps)
    ]
    feed = iter(chunks)
    algo.nets["policy"].forward = lambda *_args, **_kwargs: next(feed)

    all_time_actions = torch.zeros(steps, steps + horizon, ac_dim)
    for t in range(steps):
        action = algo.get_action({"qpos": torch.zeros(1, 3)})

        all_time_actions[[t], t : t + horizon] = chunks[t]
        current = all_time_actions[:, t]
        current = current[torch.all(current != 0, axis=1)]
        weights = np.exp(-0.01 * np.arange(len(current)))
        weights = torch.from_numpy(weights / weights.sum()).float().unsqueeze(1)
        expected = (current * weights).sum(dim=0, keepdim=True)

        torch.testing.assert_close(action, expected, rtol=0, atol=1e-6)


def test_loss_is_l1_plus_weighted_kl_as_in_original_act():
    """L1 on the chunk plus kl_weight x KL(q(z) || N(0, I)), summed over z and averaged over the batch."""
    algo = _tiny_act()
    generator = torch.Generator().manual_seed(0)
    a_hat, actions = torch.randn(2, 4, 5, 2, generator=generator)
    mu, logvar = torch.randn(2, 4, 4, generator=generator)

    losses = algo._compute_losses(
        OrderedDict(actions=a_hat, mu=mu, logvar=logvar),
        {"actions": actions},
    )

    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0)
    l1 = torch.nn.functional.l1_loss(actions, a_hat)
    assert algo.algo_config.cvae.kl_weight == 10.0
    torch.testing.assert_close(losses["kl_loss"], kl)
    torch.testing.assert_close(losses["action_loss"], l1 + 10.0 * kl)


def test_inference_uses_zero_latent_and_is_deterministic():
    algo = _tiny_act()
    algo.set_eval()
    net = algo.nets["policy"]
    obs = {"qpos": torch.randn(3, 1, 3)}
    with torch.no_grad():
        z_embed = net.nets["cvae_latent_out_proj"](torch.zeros(3, net.latent_dim))
        torch.testing.assert_close(net(obs), net._decode(obs, z_embed))
        torch.testing.assert_close(net(obs), net(obs))


def test_local_act_replaces_an_upstream_robomimic_act():
    """A robomimic release with its own "act" must neither crash nor shadow the local ACT."""
    code = "\n".join(
        [
            "from robomimic.algo.algo import REGISTERED_ALGO_FACTORY_FUNCS, register_algo_factory_func",
            "from robomimic.config.base_config import REGISTERED_CONFIGS, BaseConfig",
            "class UpstreamACTConfig(BaseConfig):",
            "    ALGO_NAME = 'act'",
            "register_algo_factory_func('act')(lambda algo_config: (None, {}))",
            "from scripts.imitation_learning.robomimic.custom_algorithms import register_custom_algorithms",
            "register_custom_algorithms()",
            "register_custom_algorithms()",
            "print(REGISTERED_CONFIGS['act'].__module__, REGISTERED_ALGO_FACTORY_FUNCS['act'].__module__)",
        ]
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
    )
    result = subprocess.run(  # noqa: S603, fixed code run by this interpreter
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    package = "scripts.imitation_learning.robomimic.custom_algorithms"
    assert lines[-1] == f"{package}.act_config {package}.act"
    assert sum("DexSuite's own act" in line for line in lines) == 1
