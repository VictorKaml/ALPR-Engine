from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# 🔥 Load your YOLO model (supports YOLOv5, v8, v9, v11)
model = YOLO("best.pt")   # <--- put your model path here


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # Read image into memory
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # 🔥 Run YOLO inference
        results = model(img)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
            })

        return JSONResponse({"detections": detections})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {"message": "YOLO Inference API Running"}
