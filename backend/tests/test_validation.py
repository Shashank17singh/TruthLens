"""Tests for file validation, magic numbers, EXIF stripping, and SSRF prevention."""

import io

import pytest
from PIL import Image

from backend.app.validation.antivirus import scan_bytes
from backend.app.validation.exif import strip_exif
from backend.app.validation.file_type import (
    detect_audio_type,
    detect_image_type,
    validate_file_content,
)
from backend.app.validation.url_fetch import is_public_ip, validate_target_url


def test_magic_number_image_detection():
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    webp_bytes = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 10
    corrupt_bytes = b"NOT_AN_IMAGE_FILE"

    assert detect_image_type(jpeg_bytes) == "image/jpeg"
    assert detect_image_type(png_bytes) == "image/png"
    assert detect_image_type(webp_bytes) == "image/webp"
    assert detect_image_type(corrupt_bytes) is None

    assert validate_file_content(jpeg_bytes, "image") == "image/jpeg"
    with pytest.raises(ValueError):
        validate_file_content(corrupt_bytes, "image")


def test_magic_number_audio_detection():
    wav_bytes = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 10
    mp3_id3 = b"ID3" + b"\x00" * 20
    mp3_frame = b"\xff\xfb" + b"\x00" * 20
    flac_bytes = b"fLaC" + b"\x00" * 20
    ogg_bytes = b"OggS" + b"\x00" * 20

    assert detect_audio_type(wav_bytes) == "audio/wav"
    assert detect_audio_type(mp3_id3) == "audio/mpeg"
    assert detect_audio_type(mp3_frame) == "audio/mpeg"
    assert detect_audio_type(flac_bytes) == "audio/flac"
    assert detect_audio_type(ogg_bytes) == "audio/ogg"


def test_exif_stripping():
    # Create an in-memory image
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    cleaned = strip_exif(raw_bytes, "image/jpeg")
    assert cleaned.startswith(b"\xff\xd8\xff")
    # Verify cleaned decodes cleanly
    opened = Image.open(io.BytesIO(cleaned))
    assert opened.size == (64, 64)


def test_ssrf_ip_filtering():
    # Private / internal IPs must be blocked
    assert not is_public_ip("127.0.0.1")
    assert not is_public_ip("10.0.0.5")
    assert not is_public_ip("192.168.1.1")
    assert not is_public_ip("172.16.0.1")
    assert not is_public_ip("169.254.169.254")  # AWS/Cloud metadata
    assert not is_public_ip("::1")

    # Public IPs must be accepted
    assert is_public_ip("8.8.8.8")
    assert is_public_ip("1.1.1.1")
    assert is_public_ip("93.184.216.34")


def test_url_validation_blocks_private():
    with pytest.raises(ValueError, match="disallowed|private|non-routable|resolve"):
        validate_target_url("http://127.0.0.1/test.png")

    with pytest.raises(ValueError, match="disallowed|private|non-routable|resolve"):
        validate_target_url("http://169.254.169.254/latest/meta-data/")


def test_antivirus_scanner():
    clean_bytes = b"Safe file content with normal bytes."
    infected_bytes = b"prefix" + b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*" + b"suffix"

    assert scan_bytes(clean_bytes) is True
    assert scan_bytes(infected_bytes) is False
