#!/usr/bin/env bash
# Initialize the StyleTTS2 submodule and install its runtime dependencies.
#
# Usage:
#   mamba activate py313
#   bash scripts/setup_styletts2.sh
#   bash scripts/setup_styletts2.sh --check-only
#
# Environment:
#   PYTHON  Python interpreter to use (default: python from the active env).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
VENDOR_DIR="$ROOT_DIR/third_party/StyleTTS2"
CHECK_ONLY=0
FORCE_MONOTONIC_ALIGN=0

usage() {
    cat <<'EOF'
Usage:
  mamba activate py313
  bash scripts/setup_styletts2.sh
  bash scripts/setup_styletts2.sh --check-only
  bash scripts/setup_styletts2.sh --force-monotonic-align

Environment:
  PYTHON  Python interpreter to use (default: python from the active env).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only)
            CHECK_ONLY=1
            ;;
        --force-monotonic-align)
            FORCE_MONOTONIC_ALIGN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

log() {
    printf '[styletts2] %s\n' "$*"
}

die() {
    printf '[styletts2] ERROR: %s\n' "$*" >&2
    exit 1
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    die "Python interpreter '$PYTHON' not found. Activate the project environment first (e.g. 'mamba activate py313')."
fi

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("StyleTTS2 requires Python >= 3.10; found %s" % sys.version.split()[0])
print("[styletts2] Python %s (%s)" % (sys.version.split()[0], sys.executable))
PY

# ---------------------------------------------------------------------------
# 1. Ensure the vendor checkout exists.
# ---------------------------------------------------------------------------
if [[ ! -f "$VENDOR_DIR/models.py" ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        die "StyleTTS2 submodule is not initialized at $VENDOR_DIR"
    fi

    if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log "Initializing third_party/StyleTTS2 submodule"
        git -C "$ROOT_DIR" submodule update --init --recursive third_party/StyleTTS2
    else
        die "StyleTTS2 checkout is missing and this is not a Git worktree; vendor it at $VENDOR_DIR"
    fi
fi

if [[ ! -f "$VENDOR_DIR/models.py" ]]; then
    die "StyleTTS2 checkout is still incomplete after submodule init"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    exec "$PYTHON" "$ROOT_DIR/scripts/check_styletts2.py"
fi

# ---------------------------------------------------------------------------
# 2. Require an existing host-specific PyTorch build.  Never install or
#    upgrade torch here: doing so can replace a ROCm/CUDA wheel with the
#    default PyPI build.
# ---------------------------------------------------------------------------
"$PYTHON" - <<'PY'
import importlib

for name in ("torch", "torchaudio"):
    try:
        module = importlib.import_module(name)
    except Exception as error:
        raise SystemExit(
            "%s is required before installing the TTS extras (%s).\n"
            "Install the ROCm/CUDA/CPU build for this host first, then re-run.\n"
            "ROCm: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html"
            % (name, error)
        )
    print("[styletts2] %s %s (host build preserved)" % (name, getattr(module, "__version__", "unknown")))
PY

# ---------------------------------------------------------------------------
# 3. Install only the [tts] extras declared in pyproject.toml.  The list is
#    read from pyproject so there is a single source of truth.  The project
#    itself is intentionally not installed here, so unrelated base
#    dependencies cannot trigger a torch re-resolution.
# ---------------------------------------------------------------------------
log "Installing [tts] dependencies declared in pyproject.toml"
mapfile -t TTS_DEPENDENCIES < <("$PYTHON" - <<'PY'
import tomllib
from pathlib import Path

with Path("pyproject.toml").open("rb") as handle:
    project = tomllib.load(handle)
for dependency in project["project"]["optional-dependencies"]["tts"]:
    # monotonic-align is handled in step 4, where it is skipped when already
    # importable so a normal setup run does not rebuild the extension.
    if dependency.startswith("monotonic-align"):
        continue
    print(dependency)
PY
)
if [[ ${#TTS_DEPENDENCIES[@]} -eq 0 ]]; then
    die "No [tts] dependencies found in pyproject.toml"
fi

# Pin torch/torchaudio to the already-installed host build for the duration of
# this install.  Some extras (vector-quantize-pytorch) depend on torch, and
# without a constraint pip could upgrade it to the default PyPI wheel.
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
"$PYTHON" - "$CONSTRAINTS_FILE" <<'PY'
import importlib.metadata as metadata
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    for name in ("torch", "torchaudio"):
        handle.write(f"{name}=={metadata.version(name)}\n")
PY

"$PYTHON" -m pip install \
    --upgrade-strategy only-if-needed \
    --constraint "$CONSTRAINTS_FILE" \
    "${TTS_DEPENDENCIES[@]}"

# ---------------------------------------------------------------------------
# 4. Repair monotonic_align.  The PyPI sdist pins Cython 0.29 and fails on
#    Python >= 3.12, so use the actively maintained upstream repository.
# ---------------------------------------------------------------------------
if [[ "$FORCE_MONOTONIC_ALIGN" -eq 1 ]] || ! "$PYTHON" -c 'import monotonic_align' >/dev/null 2>&1; then
    log "Installing monotonic_align from upstream"
    "$PYTHON" -m pip install "git+https://github.com/resemble-ai/monotonic_align.git"
else
    log "monotonic_align already installed"
fi

# ---------------------------------------------------------------------------
# 5. Verify the vendor import path and the Phonemizer/espeak backend.
# ---------------------------------------------------------------------------
"$PYTHON" "$ROOT_DIR/scripts/check_styletts2.py"
