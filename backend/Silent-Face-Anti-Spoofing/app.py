from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "resources" / "anti_spoof_models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_paths = sorted(MODEL_DIR.glob("*.pth"))
    if not model_paths:
        raise RuntimeError(f"No anti-spoof models found in {MODEL_DIR}")

    predictor = AntiSpoofPredict(device_id=0)
    for model_path in model_paths:
        predictor.load_model(str(model_path))

    app.state.predictor = predictor
    app.state.cropper = CropImage()
    app.state.model_paths = model_paths
    yield


app = FastAPI(title="Face Anti-Spoofing API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file."},
        )

    predictor = app.state.predictor
    image_bbox = predictor.get_bbox(image)
    if image_bbox is None:
        return JSONResponse(
            status_code=422,
            content={"error": "No face detected in the image."},
        )

    prediction = np.zeros((1, 3), dtype=np.float64)
    for model_path in app.state.model_paths:
        h_input, w_input, _, scale = parse_model_name(model_path.name)
        crop = scale is not None
        image_patch = app.state.cropper.crop(
            org_img=image,
            bbox=image_bbox,
            scale=scale,
            out_w=w_input,
            out_h=h_input,
            crop=crop,
        )
        prediction += predictor.predict(image_patch, str(model_path))

    label_index = int(np.argmax(prediction[0]))
    is_real = label_index == 1
    score = float(prediction[0][label_index] / len(app.state.model_paths))
    return {
        "is_real": is_real,
        "label": "REAL" if is_real else "FAKE",
        "score": score,
    }
