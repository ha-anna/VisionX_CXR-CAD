"""DICOM upload tests using small synthetic DICOM files (no real patient data)."""

import asyncio
import io

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.api.dicom import (
    DicomError,
    dicom_to_png,
    looks_like_dicom,
    read_dicom,
    to_grayscale_uint8,
)
from src.api.uploads import read_upload


def make_dicom(pixels=None, photometric="MONOCHROME2", with_pixels=True) -> bytes:
    if pixels is None:
        pixels = np.tile(np.arange(0, 4096, 16, dtype=np.uint16), (200, 1))  # 200 x 256 ramp
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.PatientName = "Test^Patient"
    ds.PatientID = "SYNTH-0001"
    ds.PatientBirthDate = "19700101"
    ds.InstitutionName = "Synthetic Hospital"
    ds.AccessionNumber = "ACC123"
    ds.add_new(0x00091001, "LO", "private value")
    if with_pixels:
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = photometric
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PixelData = pixels.tobytes()

    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


def test_detects_dicom_by_preamble_and_extension():
    assert looks_like_dicom(make_dicom(), "scan.bin")
    assert looks_like_dicom(b"no preamble", "scan.DCM")
    assert not looks_like_dicom(b"\x89PNG\r\n\x1a\n", "image.png")


def test_patient_identifiers_are_removed():
    ds = read_dicom(make_dicom())

    for keyword in ("PatientName", "PatientID", "PatientBirthDate", "InstitutionName"):
        assert keyword not in ds
    assert "AccessionNumber" not in ds
    assert (0x0009, 0x1001) not in ds
    assert ds.PatientIdentityRemoved == "YES"


def test_converted_png_contains_no_identifiers():
    png, image = dicom_to_png(make_dicom())

    assert png.startswith(b"\x89PNG")
    assert b"SYNTH-0001" not in png and b"Test^Patient" not in png
    assert image.mode == "L"
    assert image.size == (256, 200)


def test_pixels_are_scaled_to_full_8bit_range():
    gray = to_grayscale_uint8(read_dicom(make_dicom()))

    assert gray.dtype == np.uint8
    assert gray.min() == 0 and gray.max() == 255
    assert gray[0, 0] < gray[0, -1]  # ramp goes dark to bright


def test_monochrome1_is_inverted():
    normal = to_grayscale_uint8(read_dicom(make_dicom()))
    inverted = to_grayscale_uint8(read_dicom(make_dicom(photometric="MONOCHROME1")))

    np.testing.assert_array_equal(inverted, 255 - normal)


def test_dicom_without_pixels_is_rejected():
    with pytest.raises(DicomError):
        dicom_to_png(make_dicom(with_pixels=False))


def test_read_upload_converts_dicom_to_png():
    data, image = asyncio.run(read_upload(upload(make_dicom(), "chest.dcm")))

    assert data.startswith(b"\x89PNG")
    assert image.size == (256, 200)


def test_read_upload_rejects_broken_dicom():
    with pytest.raises(HTTPException) as error:
        asyncio.run(read_upload(upload(make_dicom(with_pixels=False), "empty.dcm")))

    assert error.value.status_code == 400


def test_read_upload_keeps_png_unchanged():
    buffer = io.BytesIO()
    Image.new("L", (64, 48), color=120).save(buffer, format="PNG")
    original = buffer.getvalue()

    data, image = asyncio.run(read_upload(upload(original, "xray.png")))

    assert data == original
    assert image.format == "PNG"
