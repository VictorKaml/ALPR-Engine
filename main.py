from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np

# --- 🎯 NEW OCR IMPORTS ---
import easyocr
from paddleocr import PaddleOCR

# --- IMPORTANT INITIALIZATION ---
# Initialize OCR Engines once at the start of the application
# This significantly speeds up subsequent requests.

# EasyOCR: Use English and specify GPU=False if running on CPU (common for Render free tier)
# Reader initialization can take 5-10 seconds on first run.
EASY_OCR_READER = easyocr.Reader(['en'], gpu=False)

# PaddleOCR: Use English language and set use_angle_cls=False for speed/ALPR
# lang='en', use_angle_cls=False is often best for structured plates
PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang="en") 
# -------------------------------


app = FastAPI()

# 🔥 Load your YOLO model (supports YOLOv5, v8, v9, v11)
model = YOLO("best.pt")  # <--- put your model path here


def preprocess_for_ocr(cropped_img_bgr):
    """
    Applies common preprocessing steps to enhance OCR accuracy on number plates.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (noise reduction)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding (Binarization) - important for clean OCR input
    # Use OTSU's thresholding for automatic selection
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # PaddleOCR and EasyOCR often perform best on raw, slightly enhanced BGR/RGB input
    # rather than heavily binarized images. We will return the BGR cropped image.
    return cropped_img_bgr 


def clean_plate_result(text: str) -> str:
    """Cleans up common OCR noise for license plates."""
    return (
        text.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("\n", "")
        # Common confusions (O/0, I/1, S/5, etc. - adjust based on your geography)
        .replace("O", "0") 
        .replace("I", "1")
        .replace("S", "5")
    )


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # 1. Read image into memory and convert to OpenCV (BGR format)
        contents = await image.read()
        np_array = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # 2. Run YOLO inference
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model(img_pil)[0]

        if not results.boxes:
            return JSONResponse({"plate_text": None, "message": "No license plate detected."}, status_code=200)

        # 3. Filtering: Find the highest confidence plate detection
        best_box = None
        max_conf = -1.0
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Assuming your license plate class name is "license_plate"
            if class_name in ["License_Plate", "number_plate", "plate"] and conf > max_conf: 
                 max_conf = conf
                 best_box = box.xyxy[0].tolist()
                 
        if not best_box:
             return JSONResponse({"plate_text": None, "message": "No license plate found in the required class."}, status_code=200)


        # 4. Cropping
        x1, y1, x2, y2 = [int(val) for val in best_box]
        padding = 10 
        h, w, _ = img_bgr.shape
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        cropped_img_bgr = img_bgr[y1:y2, x1:x2]
        
        if cropped_img_bgr.size == 0:
             raise Exception("Cropped image is empty.")

        
        # 5. Dual OCR Processing
        
        # --- OCR ENGINE 1: EasyOCR ---
        # EasyOCR expects a path, raw bytes, or a NumPy array
        easyocr_results = EASY_OCR_READER.readtext(cropped_img_bgr, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        easyocr_text = ""
        
        if easyocr_results:
            # Join multiple detected text lines/boxes, if any
            easyocr_text = " ".join([text for (bbox, text, conf) in easyocr_results])
            easyocr_text = clean_plate_result(easyocr_text)

        # --- OCR ENGINE 2: PaddleOCR ---
        # PaddleOCR expects BGR format (NumPy array)
        paddleocr_results = PADDLE_OCR.ocr(cropped_img_bgr, cls=False)
        paddleocr_text = ""
        
        if paddleocr_results and paddleocr_results[0]:
            # PaddleOCR returns a nested list structure: [ [[box], [text, confidence]], ... ]
            # We assume the main result is in the first set of results (results[0])
            paddleocr_text = " ".join([line[1][0] for line in paddleocr_results[0]])
            paddleocr_text = clean_plate_result(paddleocr_text)
            
        
        # 6. Comparison and Final Confidence Assessment
        final_plate_text = ""
        accuracy_confidence = 0.0 # 0.0 means no match, 1.0 means perfect match

        if easyocr_text == paddleocr_text and easyocr_text:
            final_plate_text = easyocr_text
            accuracy_confidence = 1.0 # High confidence due to matching results
        elif easyocr_text or paddleocr_text:
            # If they don't match, pick the one with better structure or length,
            # or simply prefer one engine (e.g., PaddleOCR is often slightly better)
            final_plate_text = paddleocr_text if len(paddleocr_text) > len(easyocr_text) else easyocr_text
            accuracy_confidence = 0.5 # Lower confidence due to disagreement
        else:
             final_plate_text = "OCR Failed"
             accuracy_confidence = 0.0

        
        # 7. Return the final result
        return JSONResponse({
            "plate_text": final_plate_text,
            "yolo_confidence": max_conf,
            "ocr_match_confidence": accuracy_confidence,
            "easyocr_result": easyocr_text,
            "paddleocr_result": paddleocr_text,
            "bounding_box": best_box,
            "message": "Dual-OCR ALPR processing complete."
        })

    except Exception as e:
        print(f"Dual-OCR ALPR Processing Error: {e}")
        return JSONResponse({"plate_text": None, "error": f"Internal server error during ALPR: {str(e)}"}, status_code=500)


