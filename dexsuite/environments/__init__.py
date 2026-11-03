"""Import every concrete environment module in this package.

Purpose
-------
Running this file has a single goal: make sure that each environment
module executes its own register_env decorator so that the
dexsuite.core.registry.ENV_REG table is populated.

Rules applied by the loader
---------------------------
- Only Python files that live directly in dexsuite/environments are
  considered.
- A module is skipped when its filename starts with an underscore.  This
  avoids importing helpers such as _utils.py.
- No symbols from the sub-modules are re-exported. Importing the module
  is done solely for the registration side effect.
"""

from __future__ import annotations

import importlib
import pkgutil

# --------------------------------------------------------------------- #
# walk the package tree recursively
# --------------------------------------------------------------------- #
for spec in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    leaf = spec.name.rsplit(".", 1)[-1]
    if leaf.startswith("_"):
        continue  # skip helpers
    importlib.import_module(spec.name)
