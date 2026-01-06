import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO face detector
FACE_MODEL = YOLO("yolov8n-face.pt")

TARGET_SIZE = 320
PADDING_RATIO = 0.28


# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def b64_to_img(data_uri):
    """Convert base64 to OpenCV BGR image."""
    if data_uri.startswith("data:"):
        data_uri = data_uri.split(",", 1)[1]

    img_bytes = base64.b64decode(data_uri)
    img_array = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def img_to_b64(img):
    """Convert OpenCV BGR image to base64 data URL."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buff = io.BytesIO()
    Image.fromarray(rgb).save(buff, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode()


# ---------------------------------------------------
# Enhancement Pipeline
# ---------------------------------------------------
def enhance_face(img):
    # 1. Mild denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 21)

    # 2. Sharpen
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    img = cv2.filter2D(img, -1, kernel)

    # 3. Contrast enhancement
    img = cv2.convertScaleAbs(img, alpha=1.12, beta=10)

    # 4. Gamma correction
    gamma = 1.10
    img = np.uint8(((img / 255.0) ** gamma) * 255)

    return img


# ---------------------------------------------------
# Crop BEST Face
# ---------------------------------------------------
def crop_face_best(image):
    preds = FACE_MODEL.predict(image, verbose=False)[0]
    boxes = preds.boxes

    if boxes is None or len(boxes) == 0:
        return None

    # Pick biggest face
    biggest = None
    max_area = 0

    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            biggest = (x1, y1, x2, y2)

    x1, y1, x2, y2 = biggest

    # Padding
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * PADDING_RATIO)
    pad_h = int(h * PADDING_RATIO)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(image.shape[1], x2 + pad_w)
    y2 = min(image.shape[0], y2 + pad_h)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Square padding
    h2, w2 = crop.shape[:2]
    size = max(h2, w2)
    square = np.zeros((size, size, 3), dtype=np.uint8)

    oy = (size - h2) // 2
    ox = (size - w2) // 2
    square[oy:oy + h2, ox:ox + w2] = crop

    face_final = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE))

    # Enhance
    face_final = enhance_face(face_final)

    return face_final


# ---------------------------------------------------
# API Endpoint
# ---------------------------------------------------
@app.route("/crop_enhance_face", methods=["POST"])
def crop_endpoint():
    try:
        body = request.json
        if "image_base64" not in body:
            return jsonify({"error": "Missing image_base64"}), 400

        img = b64_to_img(body["image_base64"])
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        face = crop_face_best(img)

        if face is None:
            return jsonify({"error": "No face found"}), 400

        return jsonify({
            "status": "success",
            "processed_face": img_to_b64(face)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
