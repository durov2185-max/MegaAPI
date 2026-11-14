from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO  # Библиотека для работы с .pt моделью

app = FastAPI(title="MegaAPI Captcha Solver")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "all.pt"

# Загружаем модель один раз при старте сервера для быстродействия
model = None
if MODEL_PATH.is_file():
    try:
        model = YOLO(str(MODEL_PATH))
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")


@app.get("/")
def home():
    return {
        "status": "online",
        "model": "all",
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model")
def model_status():
    exists = MODEL_PATH.is_file()
    return {
        "name": "all.pt",
        "exists": exists,
        "size": MODEL_PATH.stat().st_size if exists else 0,
        "loaded": model is not None
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(500, "Модель не загружена на сервере")

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
        # Проверяем, что картинка валидная
        image = Image.open(BytesIO(data))
        image.verify()
        
        # Переоткрываем изображение для инференса (verify портит указатель)
        image = Image.open(BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid image")

    try:
        # Запускаем распознавание капчи моделью
        results = model(image)
        
        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])           # ID класса
                conf = float(box.conf[0])          # Уверенность (точность)
                class_name = model.names[cls_id]   # Имя класса (символ / объект)
                
                # Координаты рамки [x1, y1, x2, y2], если нужны для сортировки по порядку (например, слева направо)
                xyxy = box.xyxy[0].tolist()

                predictions.append({
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "box": [round(coord, 2) for coord in xyxy]
                })

        # Если у тебя капча текстовая (символы), их полезно отсортировать по координате X слева направо:
        predictions.sort(key=lambda x: x["box"][0])
        
        # Собираем итоговую строку ответа (если модель распознает отдельные символы)
        solved_text = "".join([p["class"] for p in predictions])

    except Exception as e:
        raise HTTPException(500, f"Ошибка инференса модели: {str(e)}")

    return {
        "status": "success",
        "filename": file.filename,
        "solved_text": solved_text,  # Готовый ответ для бота/прокси
        "predictions": predictions
    }
