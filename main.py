
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

app = FastAPI(title="MegaAPI")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "all.pt"


@app.get("/")
def home():
    return {"status": "online", "model": "all"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model")
def model_status():
    exists = MODEL_PATH.is_file()

    return {
        "name": "all.pt",
        "exists": exists,
        "size": MODEL_PATH.stat().st_size if exists else 0
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in (
        "image/png",
        "image/jpeg",
        "image/webp"
    ):
        raise HTTPException(400, "Unsupported image format")

    data = await file.read()

    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "Image too large")

    try:
        image = Image.open(BytesIO(data))
        image.verify()
    except Exception:
        raise HTTPException(400, "Invalid image")

    return {
        "status": "received",
        "filename": file.filename,
        "size": len(data),
        "message": "Image uploaded successfully"
    }
