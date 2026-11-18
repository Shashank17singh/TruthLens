"""Strip EXIF metadata from uploaded images."""

import io

from PIL import Image


def strip_exif(data: bytes, mime_type: str) -> bytes:
    """Strip EXIF metadata from an image and return cleaned bytes.

    Raises ValueError if image decoding fails (corrupt media).
    """
    try:
        image = Image.open(io.BytesIO(data))
        # Ensure image is loaded
        image.load()
    except Exception as exc:
        raise ValueError(f"Corrupt or unreadable image: {exc}") from exc

    output = io.BytesIO()
    # Determine format for Pillow save
    format_map = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }
    fmt = format_map.get(mime_type, image.format or "PNG")

    # If saving JPEG, ensure RGB mode
    if fmt == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Create fresh image without exif dict/info
    clean_image = Image.new(image.mode, image.size)
    clean_image.paste(image)
    clean_image.save(output, format=fmt)
    return output.getvalue()
