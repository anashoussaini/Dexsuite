"""Push a local LeRobot dataset run folder to the Hugging Face Hub.

This script uploads a local LeRobot dataset run folder to the Hugging Face Hub.
The target repository name is derived from the run's metadata (meta/env_config.json).
An optional integer variation suffix can be appended for convenience.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HfHubHTTPError


def _slug(s: Any | None) -> str | None:
    """Normalize a string into a repo-friendly slug.

    Args:
        s: Input that should represent a string-like value.

    Returns:
        A lowercase, underscore-separated, safe string or None.
    """
    if s is None:
        return None
    s = str(s).strip().lower().replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    s = s.strip("_-")
    return s or None


def derive_repo_id(
    run_path: Path,
    *,
    owner: str = "anashoussaini",
    variation: int | None = None,
) -> str:
    """Derive a Hugging Face datasets repository ID from meta/env_config.json.

    Repo naming rules:
      - Single-arm: "{task}_{manipulator}" plus optional "_{gripper}".
      - Bimanual:
        - If left and right are identical: "{task}_{manipulator}" plus optional
          "_{gripper}".
        - Otherwise: "{task}_left_{l_manip}[_l_grip]_right_{r_manip}[_r_grip]".
      - If variation is provided, append "_{variation}".

    Args:
        run_path: Path to the dataset root (must contain meta/).
        owner: HF Hub namespace/owner (for example, "lerobot" or your username).
        variation: Optional integer appended to the repo name.

    Returns:
        Fully qualified repo ID "{owner}/{repo_name}".

    Raises:
        FileNotFoundError: If meta/env_config.json is missing.
        KeyError: If required fields are missing.
    """
    cfg_path = run_path / "meta" / "env_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"env_config.json not found at: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    task = _slug(cfg.get("task"))
    if not task:
        raise KeyError("Missing 'task' at top-level of env_config.json")

    options = cfg.get("options") or {}
    robot = options.get("robot") or {}
    robot_type = robot.get("type_of_robot")

    if robot_type == "single":
        single = robot.get("single") or {}
        manip = _slug(single.get("manipulator"))
        grip = _slug(single.get("gripper"))
        if not manip:
            raise KeyError("single.manipulator missing in env_config.json")
        repo_name = f"{task}_{manip}" + (f"_{grip}" if grip else "")
    else:
        left = robot.get("left") or {}
        right = robot.get("right") or {}
        l_manip, l_grip = _slug(left.get("manipulator")), _slug(left.get("gripper"))
        r_manip, r_grip = _slug(right.get("manipulator")), _slug(right.get("gripper"))
        if not l_manip or not r_manip:
            raise KeyError(
                "bimanual requires left.manipulator and right.manipulator in env_config.json",
            )
        if l_manip == r_manip and l_grip == r_grip:
            repo_name = f"{task}_{l_manip}" + (f"_{l_grip}" if l_grip else "")
        else:
            left_part = f"left_{l_manip}" + (f"_{l_grip}" if l_grip else "")
            right_part = f"right_{r_manip}" + (f"_{r_grip}" if r_grip else "")
            repo_name = f"{task}_{left_part}_{right_part}"

    if variation is not None:
        repo_name = f"{repo_name}_{int(variation)}"

    return f"{owner}/{repo_name}"


def push_folder_to_hub(
    run_dir: Path,
    repo_id: str,
    *,
    branch: str | None = None,
    private: bool = False,
    push_videos: bool = True,
    allow_patterns: Iterable[str] | str | None = None,
    upload_large_folder: bool = False,
    **card_kwargs: Any,
) -> None:
    """Push the given run folder to the HF Hub as a datasets repository.

    Args:
        run_dir: Local dataset directory.
        repo_id: Target "owner/repo" on the Hub.
        branch: Optional branch name.
        private: Whether to create/update as private.
        push_videos: Include the videos/ folder in the upload.
        allow_patterns: Optional upload filters (glob or list of globs).
        upload_large_folder: Use the large-folder API (more robust).
        card_kwargs: Extra dataset card fields (for example, pretty_name).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=str(run_dir))
    ds.push_to_hub(
        branch=branch,
        private=private,
        push_videos=push_videos,
        allow_patterns=allow_patterns,
        upload_large_folder=upload_large_folder,
        **card_kwargs,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Push a local LeRobot dataset run folder to the Hugging Face Hub.",
    )
    p.add_argument(
        "--path",
        required=True,
        help="Path to the run folder (must contain data/, meta/ ...).",
    )
    p.add_argument(
        "--owner",
        default="anashoussaini",
        help='Hub owner/namespace, e.g. "lerobot" or your username.',
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help='Exact Hub dataset repo ID, e.g. "my-org/lift_auto".',
    )
    p.add_argument(
        "--variation",
        "--var",
        type=int,
        default=None,
        help="Optional integer suffix appended to the repo name (e.g., _3).",
    )
    p.add_argument(
        "--branch",
        default=None,
        help="Optional branch name to push to (created if missing).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/update the dataset repository as private.",
    )
    push_videos = p.add_mutually_exclusive_group()
    push_videos.add_argument(
        "--push-videos",
        dest="push_videos",
        action="store_true",
        default=True,
        help="Include videos/ in the upload (default).",
    )
    push_videos.add_argument(
        "--no-push-videos",
        dest="push_videos",
        action="store_false",
        help="Skip uploading videos/ (faster; only metadata + parquet).",
    )
    p.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use large-folder upload API (slower, more robust).",
    )
    p.add_argument(
        "--pretty-name",
        default=None,
        help="Optional dataset card pretty_name.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for temporary Hugging Face dataset caches.",
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for Python temporary files during upload.",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point for the script.

    Parses arguments, derives the repository ID, and calls the function
    to upload the dataset, handling any potential errors.
    """
    args = parse_args()
    run_dir = Path(args.path).resolve()

    if not run_dir.exists():
        print(f"[ERROR] Path does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        _configure_scratch_dirs(cache_dir=args.cache_dir, tmp_dir=args.tmp_dir)
        repo_id = args.repo_id or derive_repo_id(
            run_dir,
            owner=args.owner,
            variation=args.variation,
        )
        _validate_repo_id(repo_id)
        print(f"[INFO] Repo ID: {repo_id}")
        print(f"[INFO] Local path: {run_dir}")

        card_kwargs = {}
        if args.pretty_name:
            card_kwargs["pretty_name"] = args.pretty_name

        push_folder_to_hub(
            run_dir,
            repo_id,
            branch=args.branch,
            private=args.private,
            push_videos=args.push_videos,
            upload_large_folder=args.upload_large_folder,
            **card_kwargs,
        )
        print(f"[OK] Uploaded to https://huggingface.co/datasets/{repo_id}")

    except (FileNotFoundError, KeyError) as e:
        print(f"[ERROR] Failed to derive repo_id: {e}", file=sys.stderr)
        sys.exit(1)
    except (HfHubHTTPError, OSError, RuntimeError, ValueError) as e:
        print(f"[ERROR] Upload failed: {e}", file=sys.stderr)
        sys.exit(1)


def _configure_scratch_dirs(cache_dir: str | None, tmp_dir: str | None) -> None:
    if cache_dir is not None:
        cache_root = Path(cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
        os.environ["HF_HOME"] = str(cache_root / "home")
        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    if tmp_dir is not None:
        tmp_root = Path(tmp_dir).expanduser().resolve()
        tmp_root.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_root)
        tempfile.tempdir = str(tmp_root)


def _validate_repo_id(repo_id: str) -> None:
    if "your_hf_username_or_org" in repo_id:
        raise ValueError(
            "Replace the placeholder repo id with your real Hugging Face owner, "
            'for example "--repo-id anashoussaini/lift_auto".',
        )
    if "/" not in repo_id:
        raise ValueError(
            'Repo id must include an owner, for example "anashoussaini/lift_auto".',
        )


if __name__ == "__main__":
    main()
