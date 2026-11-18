"""Shared command-line plumbing for the scripts in this directory."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skillseg.config import Config, load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.json"),
                        help="protocol configuration (default: configs/default.json)")
    parser.add_argument("--dataset-root", default=None,
                        help="REASSEMBLE root; overrides dataset.root and SKILLSEG_DATASET_ROOT")
    return parser


def resolve_config(args: argparse.Namespace, require_root: bool = True) -> Config:
    config = load_config(args.config)
    root = args.dataset_root or os.environ.get("SKILLSEG_DATASET_ROOT")
    if root:
        from dataclasses import replace
        from skillseg.config import DatasetConfig

        config = replace(config, dataset=replace(config.dataset, root=Path(root)))
    if require_root and not str(config.dataset.root):
        raise SystemExit(
            "dataset root is not set. Pass --dataset-root /path/to/REASSEMBLE "
            "or export SKILLSEG_DATASET_ROOT."
        )
    return config


def artifacts_dir(args: argparse.Namespace) -> Path:
    path = Path(getattr(args, "artifacts", REPO_ROOT / "artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path
