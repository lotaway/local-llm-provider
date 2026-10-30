"""Runtime helpers for the vendored StyleTTS2 voice engine.

The vendor code depends on a few process-wide environment variables (most
notably for Phonemizer/espeak-ng).  Keeping the setup here means the adapter,
the setup script and diagnostics all behave the same way.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_VENDOR_RELATIVE_PATH = Path("third_party") / "StyleTTS2"


def get_styletts2_path(project_root: Optional[os.PathLike] = None) -> Path:
    """Return the absolute path to the vendored StyleTTS2 checkout."""

    if project_root is None:
        # Import lazily so this helper stays usable by standalone scripts that
        # do not have the project root on sys.path yet.
        from constants import PROJECT_ROOT

        project_root = PROJECT_ROOT
    return Path(project_root).resolve() / _VENDOR_RELATIVE_PATH


def configure_phonemizer_espeak() -> bool:
    """Point Phonemizer at the bundled espeak-ng library when available.

    ``espeakng-loader`` ships libespeak-ng and its data files in a wheel, which
    avoids requiring a system-wide ``apt install espeak-ng``.  Existing user
    overrides are preserved.
    """

    if os.getenv("PHONEMIZER_ESPEAK_LIBRARY") and os.getenv("ESPEAK_DATA_PATH"):
        return True

    try:
        import espeakng_loader
    except ImportError:
        logger.debug("espeakng-loader is not installed; falling back to system espeak-ng")
        return False

    try:
        library_path = str(espeakng_loader.get_library_path())
        data_path = str(espeakng_loader.get_data_path())
    except Exception as exc:  # pragma: no cover - platform specific loader
        logger.warning("Unable to locate bundled espeak-ng data: %s", exc)
        return False

    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", library_path)
    os.environ.setdefault("ESPEAK_DATA_PATH", data_path)
    return True


def styletts2_dependency_report(project_root: Optional[os.PathLike] = None) -> Dict[str, object]:
    """Collect lightweight diagnostics used by setup/health checks."""

    vendor_path = get_styletts2_path(project_root)
    return {
        "vendor_path": str(vendor_path),
        "vendor_present": (vendor_path / "models.py").is_file(),
        "phonemizer_available": _can_import("phonemizer"),
        "espeak_configured": configure_phonemizer_espeak(),
        "torch_available": _can_import("torch"),
        "torchaudio_available": _can_import("torchaudio"),
        "monotonic_align_available": _can_import("monotonic_align"),
    }


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True
