"""Small JSON/image helpers shared by optional robot SDKs."""

import base64
import binascii
import struct
import zlib


def json_value(value):
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def image_observation(images):
    """Encode named uint8 RGB camera images without importing an image SDK."""
    result = []
    for name, frame in images.items():
        if frame is None:
            continue
        rows = json_value(frame)
        height = len(rows)
        width = len(rows[0]) if height else 0
        if not width or len(rows[0][0]) not in (3, 4):
            raise ValueError(f"camera {name} must be an H x W x RGB/RGBA image")
        channels = len(rows[0][0])

        def chunk(kind, data):
            return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", binascii.crc32(kind + data) & 0xFFFFFFFF)

        pixels = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in rows)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2 if channels == 3 else 6, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
        result.append({"name": name, "mimeType": "image/png", "data": base64.b64encode(png).decode("ascii")})
    return result
