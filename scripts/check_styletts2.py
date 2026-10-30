#!/usr/bin/env python3
"""Verify that the vendored StyleTTS2 engine can be imported."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tts_runtime import configure_phonemizer_espeak, get_styletts2_path

# Imported after the project root is available.  ``importlib`` is used below so
# a missing optional dependency is reported instead of crashing the checker.
REQUIRED_MODULES = (
    "torch",
    "torchaudio",
    "soundfile",
    "librosa",
    "munch",
    "pydub",
    "nltk",
    "matplotlib",
    "accelerate",
    "transformers",
    "einops",
    "einops_exts",
    "phonemizer",
    "monotonic_align",
)


def _import_result(module_name: str) -> Dict[str, str | bool]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on host packages
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "version": str(getattr(module, "__version__", "unknown"))}


def _find_checkpoint() -> str | None:
    explicit = os.getenv("TTS_MODEL_CHECKPOINT")
    if explicit and Path(explicit).is_file():
        return explicit

    search_roots = [
        Path(os.getenv("TTS_MODEL_DIR", PROJECT_ROOT / "models" / "tts")),
        PROJECT_ROOT / "third_party" / "StyleTTS2",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for pattern in ("**/epoch_2nd_*.pth", "**/styletts2*.pth", "**/*LJSpeech*.pth"):
            matches = sorted(root.glob(pattern))
            if matches:
                return str(matches[0])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    vendor_path = get_styletts2_path(PROJECT_ROOT)
    vendor_ready = (vendor_path / "models.py").is_file()

    report: Dict[str, object] = {
        "vendor_path": str(vendor_path),
        "vendor_ready": vendor_ready,
        "espeak_configured": configure_phonemizer_espeak(),
        "modules": {},
    }
    errors: List[str] = []

    if not vendor_ready:
        errors.append(
            "StyleTTS2 submodule is missing or empty; run "
            "`git submodule update --init --recursive third_party/StyleTTS2`"
        )

    modules: Dict[str, Dict[str, object]] = {}
    for module_name in REQUIRED_MODULES:
        result = _import_result(module_name)
        modules[module_name] = result
        if not result["ok"]:
            errors.append(f"{module_name}: {result['error']}")
    report["modules"] = modules

    # Importing the vendor package catches missing transitive dependencies
    # (einops_exts, transformers, torchaudio, ...) that the adapter would need.
    # Load the files under private module names because the project itself has
    # a ``utils`` package that would otherwise shadow the vendor's ``utils``.
    if vendor_ready:
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))
        vendor_modules: Dict[str, Dict[str, object]] = {}
        for module_name in ("utils", "text_utils", "models"):
            module_path = vendor_path / f"{module_name}.py"
            private_name = f"_styletts2_vendor_{module_name}"
            try:
                spec = importlib.util.spec_from_file_location(private_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[private_name] = module
                spec.loader.exec_module(module)
                vendor_modules[module_name] = {"ok": True}
            except Exception as exc:  # pragma: no cover - host specific
                vendor_modules[module_name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(f"vendor {module_name}: {type(exc).__name__}: {exc}")
        report["vendor_modules"] = vendor_modules

    # Exercise the exact Phonemizer backend used by StyleTTS2.  This fails
    # loudly when espeak-ng (or espeakng-loader) is unavailable.
    if modules["phonemizer"]["ok"]:
        try:
            from phonemizer.backend import EspeakBackend

            backend = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
            phones = backend.phonemize(["Style TTS two is ready."])
            report["phonemizer_sample"] = phones[0] if phones else ""
        except Exception as exc:  # pragma: no cover - host specific
            errors.append(f"phonemizer backend: {type(exc).__name__}: {exc}")

    checkpoint = _find_checkpoint()
    report["checkpoint"] = checkpoint
    if not checkpoint:
        report["checkpoint_hint"] = (
            "No StyleTTS2 checkpoint found. Download LJSpeech/LibriTTS weights into "
            "models/tts/ before generating audio."
        )

    report["ok"] = not errors
    report["errors"] = errors

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"StyleTTS2 vendor: {vendor_path}")
        print(f"Vendor source:    {'OK' if vendor_ready else 'MISSING'}")
        for module_name, result in modules.items():
            state = "OK" if result["ok"] else f"FAIL ({result.get('error')})"
            print(f"  {module_name:<20} {state}")
        if checkpoint:
            print(f"Checkpoint:       {checkpoint}")
        else:
            print("Checkpoint:       not downloaded (runtime audio generation will be unavailable)")
        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\nAll StyleTTS2 runtime checks passed.")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
