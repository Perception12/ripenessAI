from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("model.pt")

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLO detection API!"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    results = model(image)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(coord) for coord in box.xyxy[0]]
            })

    return detections
