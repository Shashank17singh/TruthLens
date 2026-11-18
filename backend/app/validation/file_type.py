"""Magic-number file type validation."""

from typing import Literal

ModalityType = Literal["image", "audio", "text"]

IMAGE_MIME_TYPES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],  # And "WEBP" at offset 8
}

AUDIO_MIME_TYPES = {
    "audio/wav": [b"RIFF"],  # And "WAVE" at offset 8
    "audio/x-wav": [b"RIFF"],
    "audio/mpeg": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
    "audio/flac": [b"fLaC"],
    "audio/ogg": [b"OggS"],
    "audio/mp4": [b"ftyp"],  # offset 4
    "audio/x-m4a": [b"ftyp"],  # offset 4
}


def detect_image_type(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes."""
    if len(data) < 12:
        return None
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def detect_audio_type(data: bytes) -> str | None:
    """Detect audio MIME type from magic bytes."""
    if len(data) < 12:
        return None
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"ID3") or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "audio/mp4"
    return None


def validate_file_content(data: bytes, expected_modality: ModalityType) -> str:
    """Validate data magic bytes against expected modality.

    Returns the detected MIME type, or raises ValueError.
    """
    if expected_modality == "image":
        mime = detect_image_type(data)
        if not mime:
            raise ValueError("Unsupported or invalid image format. Supported: JPEG, PNG, WebP.")
        return mime
    if expected_modality == "audio":
        mime = detect_audio_type(data)
        if not mime:
            raise ValueError("Unsupported or invalid audio format. Supported: WAV, MP3, FLAC, OGG, M4A, MP4.")
        return mime
    if expected_modality == "text":
        try:
            data.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            raise ValueError("Text input must be valid UTF-8.")
    raise ValueError(f"Unknown modality: {expected_modality}")
