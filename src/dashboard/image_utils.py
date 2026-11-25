import base64
import io

import numpy as np
from PIL import Image

DICOM_MIME = "application/dicom"


def is_dicom(raw, name=""):
    """DICOM Part 10 files have 'DICM' after a 128-byte preamble."""
    return raw[128:132] == b"DICM" or str(name).lower().endswith((".dcm", ".dicom"))


def upload_mime(raw, name=""):
    if is_dicom(raw, name):
        return DICOM_MIME
    return "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"


def decode_dicom(raw):
    """Display-only conversion. The API does its own conversion and removes patient
    identifiers before inference; nothing from the DICOM header is shown here."""
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)
        if "PixelData" not in ds:
            raise ValueError("DICOM file has no image data.")
        pixels = ds.pixel_array
    except ValueError:
        raise
    except Exception as exc:  # invalid file or missing decoder for compressed pixel data
        raise ValueError("This DICOM file cannot be displayed.") from exc
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        raise ValueError("Multi-frame DICOM preview is not supported.")
    if int(getattr(ds, "SamplesPerPixel", 1)) == 3:
        return Image.fromarray(pixels.astype(np.uint8), mode="RGB")
    gray = apply_modality_lut(pixels, ds)
    if "WindowCenter" in ds or "VOILUTSequence" in ds:
        gray = apply_voi_lut(gray, ds)
    gray = np.asarray(gray, dtype=np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        gray = gray.max() - gray
    low, high = float(gray.min()), float(gray.max())
    scaled = np.zeros(gray.shape) if high <= low else (gray - low) / (high - low) * 255
    return Image.fromarray(np.round(scaled).astype(np.uint8), mode="L").convert("RGB")


def decode_image(raw, name=""):
    if is_dicom(raw, name):
        return decode_dicom(raw)

    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Only PNG, JPEG and DICOM images are supported.")

        image.load()

        return image.convert("RGB")


def read_heatmap(value):
    if not isinstance(value, str) or not value:
        raise ValueError("No Grad-CAM image was returned.")

    encoded = value.split(",", 1)[1] if value.startswith("data:") else value

    return decode_image(base64.b64decode(encoded, validate=True))
