"""Monkeypatch a stock-robomimic bug that breaks per-component action keys.

``get_dataset_for_ep`` assumes every slash-separated key nests under
``obs``/``next_obs``, but only those two actually do -- ``action/eef_pos``-style
keys are stored flat, so any config using action components (which ACT needs
for per-component normalization) hits a bare ``AssertionError``. Invisible with
DexSuite's flat ``action_keys: ["actions"]`` alias, which is why only ACT hits it.

Auto-imported by registry.py's ``_import_local_modules()``, so it's applied
before any dataset loads.
"""

from __future__ import annotations

import robomimic.utils.dataset as dataset_module


def _get_dataset_for_ep(self, ep, key):
    key1, key2 = key.split("/", 1) if "/" in key else (None, None)
    is_nested_obs = key1 in ("obs", "next_obs")

    key_should_be_in_memory = self.hdf5_cache_mode in ["all", "low_dim"]
    if key_should_be_in_memory and is_nested_obs and key2 not in self.obs_keys_in_memory:
        key_should_be_in_memory = False

    if key_should_be_in_memory:
        if is_nested_obs:
            return self.hdf5_cache[ep][key1][key2]
        return self.hdf5_cache[ep][key]
    return self.hdf5_file["data/{}/{}".format(ep, key)]


dataset_module.SequenceDataset.get_dataset_for_ep = _get_dataset_for_ep
