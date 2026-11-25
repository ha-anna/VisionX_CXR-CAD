import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ALLOWED_FORMATS, MAX_UPLOAD_BYTES
from src.api.dicom import DicomError, dicom_to_png, looks_like_dicom


async def read_upload(file: UploadFile):
    """Return (image_bytes, PIL image) for a PNG, JPEG or DICOM upload.

    DICOM files are de-identified and converted to PNG here, so everything after this
    point receives PNG bytes and never sees patient identifiers.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    if looks_like_dicom(data, file.filename):
        try:
            return dicom_to_png(data)
        except DicomError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Only PNG, JPEG and DICOM are supported.",
            )
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="File is not a valid image.") from exc

    return data, image
