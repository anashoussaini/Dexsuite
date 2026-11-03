"""Action-change gate for interactive data collection scripts."""

import numpy as np


class AutoPauser:
    """Gate repeated actions until they move beyond a configured threshold."""

    def __init__(
        self,
        action_space: int = 0,
        trigger_range: list[float] | None = None,
        initial_action=None,
    ):
        """Initialize the pauser state and per-dimension movement thresholds."""
        self.action_space = action_space
        if trigger_range is None:
            trigger_range = [0.0] * action_space
        self.trigger_range = np.asarray(trigger_range, dtype=np.float32)
        if self.trigger_range.size != self.action_space:
            raise ValueError("trigger_range does not match action space")
        if initial_action is None or len(initial_action) == self.action_space:
            self.action = initial_action
        else:
            raise ValueError("new action does not match action space")

    def set_action(self, new_action):
        """Update the action unconditionally, without checking the trigger range."""
        if new_action is None or len(new_action) == self.action_space:
            self.action = new_action
        else:
            raise ValueError("new action does not match action space")

    def check_movement(self, new_action):
        """Check whether the action moved beyond the trigger.

        If so, update the stored action and return True.
        """
        if len(new_action) != self.action_space:
            raise ValueError("new action does not match action space")

        if self.action is None:
            self.action = new_action
            return True

        diff = np.abs(
            [self.action[i] - new_action[i] for i in range(self.action_space)],
        )
        trigger = diff > self.trigger_range
        if np.any(trigger):
            self.action = new_action
            return True
        return False
