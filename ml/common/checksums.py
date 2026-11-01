"""File hashing and verification.

Published checksums are written as "<algorithm>:<hex>". A file without a
published checksum is recorded in a lock file on first use, and every later
machine must reproduce that value. The lock is committed, so every run
trains on identical bytes.
"""

import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
LOCK_PATH = Path(__file__).with_name("checksums.lock.json")


class ChecksumError(ValueError):
    """A file's bytes do not match the expected checksum."""


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    algorithm, _, value = expected.partition(":")
    if not value:
        raise ValueError(f"checksum must look like 'sha256:<hex>', got {expected!r}")
    actual = hash_file(path, algorithm)
    if actual != value.lower():
        raise ChecksumError(f"{path}: expected {expected}, got {algorithm}:{actual}")


def verify_or_record(path: Path, key: str, lock_path: Path = LOCK_PATH) -> bool:
    """Check a file against the lock. Returns True when it was newly recorded."""
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    if key in lock:
        verify_file(path, lock[key])
        return False
    lock[key] = f"sha256:{hash_file(path)}"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True
