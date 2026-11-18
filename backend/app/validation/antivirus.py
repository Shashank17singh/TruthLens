"""Antivirus hook interface."""

EICAR_SIGNATURE = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def scan_bytes(data: bytes) -> bool:
    """Scan file bytes for malicious content.

    Returns True if clean, False if infected.
    """
    if EICAR_SIGNATURE in data:
        return False
    # Additional scanner hooks (e.g. ClamAV, VirusTotal) can be registered here.
    return True
