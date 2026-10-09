import base64
import io

from PIL import Image


def decode_image(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Only PNG and JPEG images are supported.")

        image.load()

        return image.convert("RGB")


def read_heatmap(value):
    if not isinstance(value, str) or not value:
        raise ValueError("No Grad-CAM image was returned.")

    encoded = value.split(",", 1)[1] if value.startswith("data:") else value

    return decode_image(base64.b64decode(encoded, validate=True))
