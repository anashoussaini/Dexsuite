"""Shared helpers for DexSuite SKRL scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_checkpoint_path(path: str) -> str:
    """Accept a checkpoint file or run directory and return a concrete file."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return str(checkpoint_path)

    ckpt_dir = checkpoint_path / "checkpoints"
    if not ckpt_dir.is_dir():
        ckpt_dir = checkpoint_path

    best = ckpt_dir / "best_agent.pt"
    if best.exists():
        return str(best)

    candidates = list(ckpt_dir.glob("agent_*.pt"))
    if candidates:
        return str(max(candidates, key=_checkpoint_step))

    for name in ("best_agent.pt", "agent.pt"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Could not find a checkpoint under: {path}")


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"agent_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def infer_policy_action_dim_from_checkpoint(path: str) -> int | None:
    """Try to read a PPO policy action dimension from a checkpoint."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError):
        return None
    return _scan_policy_action_dim(data)


def _scan_policy_action_dim(node: Any) -> int | None:
    if not isinstance(node, dict):
        return None
    log_std = node.get("log_std_parameter")
    if isinstance(log_std, torch.Tensor) and log_std.ndim == 1:
        return int(log_std.shape[0])
    for key, value in node.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2 and key.endswith(
            ".weight",
        ):
            return int(value.shape[0])
    for value in node.values():
        out = _scan_policy_action_dim(value)
        if out is not None:
            return out
    return None


def map_actions_tensor(
    actions: torch.Tensor,
    mode: str,
    src_dim: int,
    dst_dim: int,
) -> torch.Tensor:
    """Map action vectors between compatible action dimensions."""
    if src_dim == dst_dim or mode == "none":
        return actions

    if mode == "auto":
        if src_dim == dst_dim + 1:
            mode = "trim_last"
        elif dst_dim == src_dim + 1:
            mode = "pad_zero_last"
        else:
            raise RuntimeError(
                f"Cannot auto map actions: ckpt_dim={src_dim} env_dim={dst_dim}",
            )

    if mode == "trim_last":
        if src_dim <= dst_dim:
            raise RuntimeError("trim_last requires src_dim > dst_dim")
        return actions[..., :dst_dim]

    if mode == "pad_zero_last":
        if src_dim >= dst_dim:
            raise RuntimeError("pad_zero_last requires src_dim < dst_dim")
        pad = torch.zeros(
            (*actions.shape[:-1], dst_dim - src_dim),
            device=actions.device,
            dtype=actions.dtype,
        )
        return torch.cat([actions, pad], dim=-1)

    raise RuntimeError(f"Unknown action_map: {mode}")


def as_action_tensor(actions_out: Any, device: torch.device) -> torch.Tensor:
    """Extract an action tensor from SKRL `agent.act` output."""
    if isinstance(actions_out, tuple) and len(actions_out) >= 1:
        actions_out = actions_out[0]
    if isinstance(actions_out, dict):
        for key in ("actions", "action", "act"):
            if key in actions_out:
                actions_out = actions_out[key]
                break
    if isinstance(actions_out, np.ndarray):
        return torch.as_tensor(actions_out, device=device, dtype=torch.float32)
    if isinstance(actions_out, torch.Tensor):
        return actions_out.to(device=device, dtype=torch.float32)
    raise TypeError(f"Unsupported action output type: {type(actions_out)}")
