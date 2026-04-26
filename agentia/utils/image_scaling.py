"""Utility for downscaling images to standard resolutions."""

from typing import Literal
from io import BytesIO

ScaleOption = Literal["360p", "720p", "1080p", "raw"]

_TARGET_PIXELS: dict[ScaleOption, int] = {
    "360p": 640 * 360,  # 230_400
    "720p": 1280 * 720,  # 921_600
    "1080p": 1920 * 1080,  # 2_073_600
}


def downscale_jpeg(data: bytes, scale: ScaleOption) -> bytes:
    """Downscale a JPEG image to roughly match the target pixel count.

    The aspect ratio is preserved. The image is scaled so that
    ``new_w * new_h ≈ target_pixels``. If the image already has fewer
    pixels than the target, it is returned unchanged.

    Args:
        data: JPEG image bytes.
        scale: Target resolution or "raw" to skip scaling.

    Returns:
        JPEG image bytes, possibly resized.
    """
    if scale == "raw":
        return data

    import math
    from PIL import Image

    target_pixels = _TARGET_PIXELS[scale]
    img = Image.open(BytesIO(data))
    orig_w, orig_h = img.size
    orig_pixels = orig_w * orig_h

    if orig_pixels <= target_pixels:
        return data

    ratio = math.sqrt(target_pixels / orig_pixels)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)  # type: ignore[attr-defined]
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
