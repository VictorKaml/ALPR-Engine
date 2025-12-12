import io
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from paddleocr import PaddleOCR

# --- INITIALIZATION ---
app = FastAPI()

# Load YOLO model once
# Ensure 'best.pt' is in your root directory or provide the correct path
model = YOLO("best.pt")

# Initialize OCR Engines globally to save memory/time on requests
# We use gpu=False for standard Render web services
EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)
PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)

def clean_plate_result(text: str) -> str:
    """Standardizes the output text from OCR engines."""
    return (
        text.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("\n", "")
        .replace("O", "0") 
        .replace("I", "1")
        .replace("S", "5")
    )

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # 1. Load and Decode Image
        contents = await image.read()
        np_array = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        # 2. YOLO Inference for Bounding Box
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        yolo_results = model(img_pil)[0]

        if not yolo_results.boxes:
            return JSONResponse({
                "plate_text": None, 
                "message": "No objects detected"
            })

        # 3. Filter for best plate detection
        best_box = None
        max_conf = -1.0
        
        for box in yolo_results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()
            
            if "plate" in class_name and conf > max_conf:
                max_conf = conf
                best_box = box.xyxy[0].tolist()

        if not best_box:
            return JSONResponse({
                "plate_text": None, 
                "message": "License plate class not found"
            })

        # 4. Cropping and Padding
        x1, y1, x2, y2 = [int(v) for v in best_box]
        padding = 10
        h, w, _ = img_bgr.shape
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
        
        cropped_plate = img_bgr[y1:y2, x1:x2]

        # 5. Dual OCR Processing
        # EasyOCR
        easy_res = EASY_OCR_READER.readtext(cropped_plate)
        easy_text = clean_plate_result(" ".join([res[1] for res in easy_res])) if easy_res else ""

        # PaddleOCR
        paddle_res = PADDLE_OCR.ocr(cropped_plate, cls=False)
        paddle_text = ""
        if paddle_res and paddle_res[0]:
            paddle_text = clean_plate_result(" ".join([line[1][0] for line in paddle_res[0]]))

        # 6. Logic Comparison
        final_plate = ""
        match_confidence = 0.0

        if easy_text == paddle_text and easy_text != "":
            final_plate = easy_text
            match_confidence = 1.0
        else:
            # Fallback logic: prefer the longer string or PaddleOCR
            final_plate = paddle_text if len(paddle_text) >= len(easy_text) else easy_text
            match_confidence = 0.5 if (easy_text or paddle_text) else 0.0

        return {
            "plate_text": final_plate,
            "ocr_match_confidence": match_confidence,
            "yolo_confidence": max_conf,
            "details": {
                "easyocr": easy_text,
                "paddleocr": paddle_text
            }
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def health_check():
    return {"status": "ALPR Engine Online"}
