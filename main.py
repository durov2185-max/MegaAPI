import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO

app = FastAPI(title="MegaAPI", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_NAMES = [
    "all.pt",
    "funtime.pt",
    "luckydayz.pt",
    "nighthvh.pt",
    "3b3t.pt",
    "bomjpvp.pt",
    "skytime.pt",
    "limbo.pt",
    "holyworld.pt",
    "sunway.pt",
    "game.pt",
]

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "MegaAPI",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/models")
def models():
    return {
        name: (MODEL_DIR / name).is_file()
        for name in MODEL_NAMES
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
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

@app.get("/models/status")
def model_status():
    return {
        "directory": str(MODEL_DIR),
        "models": {
            name: {
                "exists": (MODEL_DIR / name).is_file(),
                "size": (
                    (MODEL_DIR / name).stat().st_size
                    if (MODEL_DIR / name).is_file()
                    else 0
                )
            }
            for name in MODEL_NAMES
        }
    }