"""SKRL model definitions used by DexSuite training scripts."""

from __future__ import annotations

import torch
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch import nn


class PPOPolicy(GaussianMixin, Model):
    """Small stochastic MLP policy for PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: int = -20,
        max_log_std: int = 2,
        reduction: str = "sum",
    ) -> None:
        """Initialize the policy network."""
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            reduction,
        )
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):  # noqa: ARG002
        """Compute mean action and log standard deviation."""
        return self.net(inputs["states"]), self.log_std_parameter, {}


class PPOValue(DeterministicMixin, Model):
    """State-value MLP for PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
    ) -> None:
        """Initialize the value network."""
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):  # noqa: ARG002
        """Compute state values."""
        return self.net(inputs["states"]), {}


class TD3Policy(DeterministicMixin, Model):
    """Deterministic actor MLP for TD3."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
    ) -> None:
        """Initialize the actor network."""
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role):  # noqa: ARG002
        """Compute bounded deterministic actions."""
        return self.net(inputs["states"]), {}


class TD3Value(DeterministicMixin, Model):
    """Q-value critic MLP for TD3."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
    ) -> None:
        """Initialize the critic network."""
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def compute(self, inputs, role):  # noqa: ARG002
        """Compute Q-values for state-action pairs."""
        return self.net(
            torch.cat([inputs["states"], inputs["taken_actions"]], dim=1),
        ), {}
