"""Resolve evaluation model files on demand from the project models directory.

A model file that already exists is always used as-is; a missing or corrupted
file is downloaded from its pinned official source, verified against the
recorded sha256 and installed atomically. A failed download rolls the previous
file back so the deployment stays in its last known state.
"""

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class ModelAsset:
    name: str
    url: str
    sha256: str
    version: str


DNSMOS_ASSET = ModelAsset(
    name="dnsmos",
    url="https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
    sha256="269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    version="DNSMOS P.835 sig_bak_ovr",
)


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_local_model(
    path: Path,
    asset: ModelAsset,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> Path:
    """Return ``path`` ready for use, downloading the asset only when needed."""
    target = Path(path)
    if file_sha256(target) == asset.sha256:
        _log("model_asset_ready", asset, target)
        return target
    aside = _quarantine_existing(target, asset)
    try:
        _download_verified(target, asset, opener)
    except Exception:
        _rollback_quarantined(aside, asset, target)
        raise
    _discard_quarantined(aside)
    _log("model_asset_installed", asset, target)
    return target


def _log(event: str, asset: ModelAsset, target: Path) -> None:
    logger.info(
        event,
        extra={"model": asset.name, "path": str(target), "version": asset.version},
    )


def _rollback_quarantined(aside: Optional[Path], asset: ModelAsset, target: Path) -> None:
    if aside is None:
        return
    os.replace(aside, target)
    logger.warning(
        "model_asset_rollback",
        extra={"model": asset.name, "path": str(target)},
    )


def _discard_quarantined(aside: Optional[Path]) -> None:
    if aside is not None:
        aside.unlink(missing_ok=True)


def _quarantine_existing(target: Path, asset: ModelAsset) -> Optional[Path]:
    if not target.exists():
        return None
    aside = target.with_name(f"{target.name}.rejected-{asset.sha256[:8]}")
    logger.warning(
        "model_asset_checksum_mismatch",
        extra={"model": asset.name, "path": str(target)},
    )
    os.replace(target, aside)
    return aside


def _download_verified(target: Path, asset: ModelAsset, opener: Callable[..., object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _fetch_to_temporary(target, asset, opener)
    try:
        if file_sha256(Path(temporary)) != asset.sha256:
            raise ValueError(f"downloaded {asset.name} does not match the pinned checksum")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fetch_to_temporary(target: Path, asset: ModelAsset, opener: Callable[..., object]) -> str:
    descriptor, name = tempfile.mkstemp(dir=str(target.parent), prefix=f"{target.name}.", suffix=".part")
    os.close(descriptor)
    try:
        with opener(asset.url, timeout=300) as response, open(name, "wb") as stream:
            shutil.copyfileobj(response, stream, CHUNK_BYTES)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
    return name
