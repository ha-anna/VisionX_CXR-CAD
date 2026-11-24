import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ALLOWED_FORMATS, MAX_UPLOAD_BYTES


async def read_upload(file: UploadFile):
    data = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Only PNG and JPEG are supported.",
            )
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="File is not a valid image.") from exc

    return data, image
