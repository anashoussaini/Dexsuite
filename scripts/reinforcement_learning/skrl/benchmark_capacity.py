"""Benchmark DexSuite SKRL vectorized env capacity."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.reinforcement_learning.skrl.configs import ENV_CONFIGS
    from scripts.reinforcement_learning.skrl.envs import make_skrl_env
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.reinforcement_learning"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs import ENV_CONFIGS
    from envs import make_skrl_env


def _rss_mb() -> float:
    """Return process max RSS in MB on Linux/macOS."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        rss /= 1024.0 * 1024.0
    else:
        rss /= 1024.0
    return rss


def _cuda_stats() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_allocated_mb": None,
            "cuda_reserved_mb": None,
            "cuda_free_mb": None,
            "cuda_total_mb": None,
        }
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "cuda_free_mb": round(free / 1024**2, 2),
        "cuda_total_mb": round(total / 1024**2, 2),
    }


def _cube_positions(raw_env: Any) -> torch.Tensor | None:
    cube = getattr(raw_env, "cube", None)
    if cube is None or not hasattr(cube, "get_pos"):
        return None
    pos = cube.get_pos()
    tensor = torch.as_tensor(pos, dtype=torch.float32).detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _randomization_stats(before: torch.Tensor | None, after: torch.Tensor | None) -> dict[str, Any]:
    if before is None:
        return {
            "randomization_checked": False,
            "cube_xy_std": None,
            "reset_idx_cube_changed": None,
        }

    xy = before[:, :2].reshape(before.shape[0], 2)
    cube_xy_std = float(torch.std(xy, dim=0).mean().item()) if xy.shape[0] > 1 else 0.0
    changed = None
    if after is not None and before.shape == after.shape:
        delta = torch.linalg.norm(before - after, dim=-1)
        changed = bool(torch.any(delta > 1e-5).item())
    return {
        "randomization_checked": True,
        "cube_xy_std": round(cube_xy_std, 6),
        "reset_idx_cube_changed": changed,
    }


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """Run one benchmark size in a fresh process."""
    device_label = "cuda" if torch.cuda.is_available() else "cpu"
    result: dict[str, Any] = {
        "env": args.env,
        "robot_preset": args.robot_preset or ENV_CONFIGS[args.env].default_robot_preset,
        "n_envs": args.n_envs,
        "steps": args.steps,
        "device": device_label,
        "ok": False,
    }
    t0 = time.perf_counter()
    env = None
    try:
        env = make_skrl_env(
            args.env,
            n_envs=args.n_envs,
            render_mode=None,
            robot_preset=args.robot_preset,
        )
        build_s = time.perf_counter() - t0

        reset_t0 = time.perf_counter()
        obs, _info = env.reset()
        reset_s = time.perf_counter() - reset_t0
        before_cube = _cube_positions(env._env)  # noqa: SLF001

        action_dim = int(env.action_space.shape[-1])
        step_t0 = time.perf_counter()
        for _ in range(args.steps):
            actions = torch.empty(
                (args.n_envs, action_dim),
                device=env.device,
                dtype=torch.float32,
            ).uniform_(-0.25, 0.25)
            obs, reward, terminated, truncated, _info = env.step(actions)
        step_total_s = time.perf_counter() - step_t0

        reset_idx_s = None
        after_cube = None
        if args.n_envs > 1:
            idx = torch.unique(
                torch.tensor(
                    [0, args.n_envs // 2, args.n_envs - 1],
                    dtype=torch.long,
                ),
            )
            reset_idx_t0 = time.perf_counter()
            reset_obs, _reset_info = env.reset_by_idx(idx)
            reset_idx_s = time.perf_counter() - reset_idx_t0
            if reset_obs.shape[0] not in {args.n_envs, int(idx.numel())}:
                raise RuntimeError(
                    f"reset_by_idx returned unexpected shape {tuple(reset_obs.shape)}",
                )
            after_cube = _cube_positions(env._env)  # noqa: SLF001

        result.update(
            {
                "ok": True,
                "obs_shape": list(obs.shape),
                "reward_shape": list(reward.shape),
                "terminated_shape": list(terminated.shape),
                "truncated_shape": list(truncated.shape),
                "build_s": round(build_s, 4),
                "reset_s": round(reset_s, 4),
                "step_total_s": round(step_total_s, 4),
                "step_s": round(step_total_s / max(args.steps, 1), 4),
                "reset_idx_s": None if reset_idx_s is None else round(reset_idx_s, 4),
                "rss_mb": round(_rss_mb(), 2),
            },
        )
        result.update(_cuda_stats())
        result.update(_randomization_stats(before_cube, after_cube))
        return result
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "rss_mb": round(_rss_mb(), 2),
            },
        )
        result.update(_cuda_stats())
        return result
    finally:
        if env is not None:
            env.close()


def _powers(max_envs: int, min_envs: int = 1) -> list[int]:
    values = []
    n = 1
    while n <= max_envs:
        if n >= min_envs:
            values.append(n)
        n *= 2
    return values


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run one subprocess per vectorization size."""
    results = []
    script = Path(__file__).resolve()
    for n_envs in _powers(args.max_envs, args.min_envs):
        cmd = [
            sys.executable,
            str(script),
            "--single",
            "--env",
            args.env,
            "--n-envs",
            str(n_envs),
            "--steps",
            str(args.steps),
        ]
        if args.robot_preset is not None:
            cmd += ["--robot-preset", args.robot_preset]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        try:
            result = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            result = {
                "env": args.env,
                "robot_preset": args.robot_preset or ENV_CONFIGS[args.env].default_robot_preset,
                "n_envs": n_envs,
                "steps": args.steps,
                "ok": False,
                "error": proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}",
            }
        results.append(result)
        print(_format_row(result), flush=True)
        if args.stop_on_fail and not result.get("ok", False):
            break
    if args.output:
        _write_csv(Path(args.output), results)
    return results


def _format_row(row: dict[str, Any]) -> str:
    if not row.get("ok", False):
        return f"{row.get('n_envs'):>5} FAIL {row.get('error')}"
    return (
        f"{row['n_envs']:>5} ok "
        f"reset={row['reset_s']:.3f}s "
        f"step={row['step_s']:.3f}s "
        f"rss={row['rss_mb']:.1f}MB "
        f"cuda={row.get('cuda_available')} "
        f"xy_std={row.get('cube_xy_std')}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="lift", choices=sorted(ENV_CONFIGS))
    parser.add_argument("--robot-preset", default=None)
    parser.add_argument("--min-envs", type=int, default=1)
    parser.add_argument("--max-envs", type=int, default=2048)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", default="/tmp/dexsuite_skrl_capacity.csv")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.single:
        print(json.dumps(run_single(args), sort_keys=True))
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
