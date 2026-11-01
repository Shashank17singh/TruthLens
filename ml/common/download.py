"""Resumable downloads and safe archive extraction."""

import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from ml.common.checksums import ChecksumError, verify_file

CHUNK_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 60


def download(url: str, destination: Path, checksum: str | None = None) -> Path:
    """Download to `destination`, resuming a partial file and verifying the result."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checksum:
            verify_file(destination, checksum)
        return destination

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT_SECONDS) as response:
        if response.status_code == 416:
            pass
        else:
            response.raise_for_status()
            resumed = response.status_code == 206
            remaining = int(response.headers.get("Content-Length", 0))
            with partial.open("ab" if resumed else "wb") as stream, tqdm(
                total=remaining + (offset if resumed else 0),
                initial=offset if resumed else 0,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(CHUNK_BYTES):
                    stream.write(chunk)
                    progress.update(len(chunk))

    if checksum:
        try:
            verify_file(partial, checksum)
        except ChecksumError:
            partial.unlink()
            raise
    partial.replace(destination)
    return destination


def extract(archive: Path, destination: Path) -> None:
    """Unpack once. Tar's data filter and zipfile both refuse paths outside `destination`."""
    marker = destination / f".{archive.name}.extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    marker.touch()
