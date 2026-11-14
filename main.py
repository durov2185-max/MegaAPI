import base64
import uuid
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="MegaAPI Universal Captcha Solver")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "all.pt"

# Хранилище решенных капч для RuCaptcha эндпоинтов: { "captcha_id": "текст_ответа" }
solved_captchas = {}

model = None
if MODEL_PATH.is_file():
    try:
        model = YOLO(str(MODEL_PATH))
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")


def solve_image_bytes(image_bytes: bytes) -> str:
    """Универсальная функция распознавания капчи из байтов картинки."""
    if model is None:
        raise RuntimeError("Модель не загружена на сервере")

    image = Image.open(BytesIO(image_bytes))
    results = model(image)

    predictions = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            predictions.append({
                "class": class_name,
                "x1": xyxy[0]
            })

    # Сортируем символы слева направо по координате X
    predictions.sort(key=lambda item: item["x1"])
    return "".join([p["class"] for p in predictions])


@app.get("/")
def home():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "endpoints": ["/in.php", "/res.php", "/upload", "/solve"]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ==========================================
# 1. ОБЫЧНЫЕ ЭНДПОИНТЫ (Прямой запрос -> Ответ)
# ==========================================

@app.post("/upload")
@app.post("/solve")
async def direct_solve(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(500, "Модель не загружена на сервере")

    try:
        image_bytes = await file.read()
        
        # Проверяем валидность картинки
        img_check = Image.open(BytesIO(image_bytes))
        img_check.verify()

        # Распознаем
        answer = solve_image_bytes(image_bytes)

        return {
            "status": "success",
            "solution": answer,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(400, f"Ошибка обработки изображения: {str(e)}")


# ==========================================
# 2. RUCAPTCHA ЭНДПОИНТЫ (/in.php и /res.php)
# ==========================================

@app.post("/in.php", response_class=PlainTextResponse)
async def rucaptcha_in(request: Request):
    if model is None:
        return "ERROR_NO_SLOT_AVAILABLE"

    image_bytes = None

    try:
        form = await request.form()
        if "file" in form and hasattr(form["file"], "read"):
            image_bytes = await form["file"].read()
        elif "body" in form:
            b64_str = str(form["body"])
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_str)
    except Exception:
        pass

    if not image_bytes:
        try:
            data = await request.json()
            if "body" in data:
                b64_str = str(data["body"])
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_str)
        except Exception:
            pass

    if not image_bytes:
        return "ERROR_ZERO_CAPTCHA_FILESIZE"

    try:
        answer = solve_image_bytes(image_bytes)
        captcha_id = str(uuid.uuid4().int)[:10]
        solved_captchas[captcha_id] = answer

        # Очищаем старые задачи в памяти
        if len(solved_captchas) > 500:
            oldest_key = next(iter(solved_captchas))
            del solved_captchas[oldest_key]

        return f"OK|{captcha_id}"
    except Exception as e:
        print(f"Ошибка инференса: {e}")
        return "ERROR_CAPTCHA_UNSOLVABLE"


@app.get("/res.php", response_class=PlainTextResponse)
@app.post("/res.php", response_class=PlainTextResponse)
async def rucaptcha_res(request: Request, action: str = "", id: str = ""):
    if not action or not id:
        try:
            form = await request.form()
            action = action or str(form.get("action", ""))
            id = id or str(form.get("id", ""))
        except Exception:
            pass

    if action == "getbalance":
        return "999.00"

    if action == "get":
        if id in solved_captchas:
            answer = solved_captchas.pop(id)
            return f"OK|{answer}"
        else:
            return "ERROR_NO_SUCH_CAPCHA_ID"

    return "ERROR_WRONG_METHOD"
