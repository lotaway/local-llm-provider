"""Verify the host PyTorch build and pin it for subsequent pip installs."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict

PROTECTED_PACKAGES = ("torch", "torchaudio", "torchvision")
ROCM_HINT = "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html"
PYTORCH_HINT = "https://pytorch.org/get-started/locally/"


def emit(message: str, error: bool = False) -> None:
    print(message, file=sys.stderr if error else sys.stdout)


def _is_installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def installed_versions() -> Dict[str, str]:
    return {name: metadata.version(name) for name in PROTECTED_PACKAGES if _is_installed(name)}


def _accelerator(torch_module) -> str:
    version = getattr(torch_module, "version", None)
    if getattr(version, "hip", None):
        return "rocm"
    if getattr(version, "cuda", None):
        return "cuda"
    return "cpu"


def build_report() -> Dict[str, str]:
    import torch

    return {
        "version": str(getattr(torch, "__version__", "unknown")),
        "accelerator": _accelerator(torch),
    }


def write_constraints(path: Path, versions: Dict[str, str]) -> None:
    lines = [f"{name}=={version}" for name, version in sorted(versions.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify host PyTorch build and emit pip constraints")
    parser.add_argument("--constraints", type=Path, default=None)
    parser.add_argument("--require-accelerator", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    versions = installed_versions()
    if "torch" not in versions:
        emit(f"[torch] torch is missing; install the host build first: {ROCM_HINT} or {PYTORCH_HINT}", True)
        return 1
    report = build_report()
    if args.require_accelerator and report["accelerator"] == "cpu":
        emit("[torch] CPU build detected but an accelerator build is required", True)
        return 1
    emit(f"[torch] {report['version']} ({report['accelerator']}) pinned: {versions}")
    if args.constraints:
        write_constraints(args.constraints, versions)
        emit(f"[torch] constraints written to {args.constraints}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
