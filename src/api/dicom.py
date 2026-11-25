"""DICOM upload support: remove patient identifiers and convert pixels to an 8-bit image."""

import io

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut, apply_voi_lut

# Attributes that identify the patient or the visit. Removed before any further processing.
IDENTIFYING_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "OtherPatientIDs",
    "OtherPatientIDsSequence",
    "OtherPatientNames",
    "PatientBirthName",
    "PatientMotherBirthName",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "MilitaryRank",
    "MedicalRecordLocator",
    "IssuerOfPatientID",
    "AccessionNumber",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "StudyID",
)


class DicomError(ValueError):
    """The upload is DICOM but cannot be used (no pixels, unsupported encoding, ...)."""


def looks_like_dicom(data: bytes, filename: str | None = None) -> bool:
    """DICOM Part 10 files have 'DICM' after a 128-byte preamble."""
    if len(data) >= 132 and data[128:132] == b"DICM":
        return True
    return bool(filename) and str(filename).lower().endswith((".dcm", ".dicom"))


def remove_identifiers(ds: Dataset) -> Dataset:
    """Delete patient identifiers and private tags in place and return the dataset."""
    for keyword in IDENTIFYING_KEYWORDS:
        if keyword in ds:
            delattr(ds, keyword)
    ds.remove_private_tags()
    ds.PatientIdentityRemoved = "YES"
    return ds


def read_dicom(data: bytes) -> Dataset:
    """Parse DICOM bytes and return the dataset with identifiers already removed."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
    except (InvalidDicomError, EOFError, OSError, ValueError, TypeError) as exc:
        raise DicomError("File is not a valid DICOM file.") from exc
    return remove_identifiers(ds)


def to_grayscale_uint8(ds: Dataset) -> np.ndarray:
    """Convert DICOM pixel data to a display-ready 8-bit grayscale array."""
    if "PixelData" not in ds:
        raise DicomError("DICOM file has no image data.")
    try:
        pixels = ds.pixel_array
    except Exception as exc:  # pydicom raises several types for missing codecs
        raise DicomError("This DICOM encoding is not supported. Use uncompressed DICOM.") from exc

    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frames > 1:
        pixels = pixels[0]
        ds = ds.copy()
        ds.NumberOfFrames = 1

    if int(getattr(ds, "SamplesPerPixel", 1)) == 3:
        rgb = pixels.astype(np.float32)
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        gray = apply_modality_lut(pixels, ds)
        if "WindowCenter" in ds or "VOILUTSequence" in ds:
            gray = apply_voi_lut(gray, ds)
        gray = np.asarray(gray, dtype=np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            gray = gray.max() - gray

    if gray.ndim != 2:
        raise DicomError("DICOM image must be a single 2D image.")
    low, high = float(gray.min()), float(gray.max())
    if not np.isfinite([low, high]).all():
        raise DicomError("DICOM image contains invalid values.")
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.round((gray - low) / (high - low) * 255).astype(np.uint8)


def dicom_to_png(data: bytes) -> tuple[bytes, Image.Image]:
    """Return de-identified PNG bytes and the matching PIL image for a DICOM upload."""
    image = Image.fromarray(to_grayscale_uint8(read_dicom(data)), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image
