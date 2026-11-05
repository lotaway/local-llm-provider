#!/usr/bin/env bash
# Install the project without replacing the host-specific PyTorch build.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
EXTRAS="${EXTRAS:-}"
REQUIRE_ACCELERATOR="${REQUIRE_ACCELERATOR:-0}"
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf '[install] Python interpreter not found: %s\n' "$PYTHON" >&2
    exit 1
fi

VERIFY_ARGS=()
if [[ "$REQUIRE_ACCELERATOR" == "1" ]]; then
    VERIFY_ARGS+=("--require-accelerator")
fi
"$PYTHON" scripts/check_torch_build.py --constraints "$CONSTRAINTS_FILE" "${VERIFY_ARGS[@]}"

if [[ -n "$EXTRAS" ]]; then
    INSTALL_TARGET=(-e ".[$EXTRAS]")
else
    INSTALL_TARGET=(-e ".")
fi

"$PYTHON" -m pip install \
    --upgrade-strategy only-if-needed \
    --constraint "$CONSTRAINTS_FILE" \
    "${INSTALL_TARGET[@]}"

"$PYTHON" scripts/check_torch_build.py "${VERIFY_ARGS[@]}"
