import base64
import io
import math
import logging
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from deepface import DeepFace

# ---------------------------------------
# CONFIG
# ---------------------------------------
HOST = "0.0.0.0"
PORT = 4000
DEBUG = True

TARGET_SIZE = 300
PADDING_RATIO = 0.22  # balanced crop pad

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("face_preprocess")

app = Flask(__name__)


# ---------------------------------------
# Utility: Base64 → Image
# ---------------------------------------
def decode_image(data: str) -> np.ndarray:
    """Convert base64 → BGR image."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]

    data += "=" * ((4 - len(data) % 4) % 4)
    img_bytes = base64.b64decode(data)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil_img)[:, :, ::-1]  # RGB→BGR
    return arr


# ---------------------------------------
# Utility: Image → base64 PNG
# ---------------------------------------
def encode_png(img: np.ndarray) -> str:
    """Convert BGR → base64 PNG."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------
# Eye Alignment
# ---------------------------------------
def align_eyes(face: np.ndarray, landmarks: Dict) -> np.ndarray:
    """Rotate face so eyes are horizontal."""
    try:
        left = np.array(landmarks["left_eye"]).mean(axis=0)
        right = np.array(landmarks["right_eye"]).mean(axis=0)

        dx, dy = right[0] - left[0], right[1] - left[1]
        angle = math.degrees(math.atan2(dy, dx))

        h, w = face.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aligned = cv2.warpAffine(face, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except:
        return face


# ---------------------------------------
# Crop → Square → Resize
# ---------------------------------------
def crop_face(img: np.ndarray, box: Dict, padding_ratio=PADDING_RATIO):
    """Perfect centered face crop with padding."""
    h, w = img.shape[:2]
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]

    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)

    crop = img[y1:y2, x1:x2]

    # Make square
    h2, w2 = crop.shape[:2]
    size = max(h2, w2)
    square = np.zeros((size, size, 3), dtype=np.uint8)

    oy = (size - h2) // 2
    ox = (size - w2) // 2
    square[oy:oy + h2, ox:ox + w2] = crop

    # Resize to model input size
    return cv2.resize(square, (TARGET_SIZE, TARGET_SIZE))


# ---------------------------------------
# Enhancement Pipeline (HD without ESRGAN)
# ---------------------------------------
def enhance_face(img: np.ndarray) -> np.ndarray:
    """HD enhancement safe for recognition."""
    # Step 1: Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # Step 2: Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img = cv2.filter2D(img, -1, kernel)

    # Step 3: Contrast & brightness correction
    img = cv2.convertScaleAbs(img, alpha=1.20, beta=8)

    # Step 4: Final resize (ensure perfect size)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    return img


# ---------------------------------------
# Face Detection
# ---------------------------------------
def detect_faces(img: np.ndarray):
    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend="retinaface",
        enforce_detection=False,
        align=False
    )

    outputs = []
    for f in faces:
        box = f["facial_area"]
        crop = crop_face(img, box)

        # Align by eyes
        landmarks = f.get("landmarks")
        if landmarks:
            crop = align_eyes(crop, landmarks)

        # Enhancement
        crop = enhance_face(crop)

        outputs.append({
            "crop": crop,
            "bbox": box
        })

    return outputs


# ---------------------------------------
# API Endpoint
# ---------------------------------------
@app.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        body = request.get_json()
        img_b64 = body.get("img_base64")
        if not img_b64:
            return jsonify({"error": "img_base64 required"}), 400

        img = decode_image(img_b64)
        faces = detect_faces(img)

        if not faces:
            return jsonify({"message": "No face detected", "faces": []}), 200

        result = []
        for f in faces:
            result.append({
                "processed_image": encode_png(f["crop"]),
                "bbox": f["bbox"]
            })

        return jsonify(result), 200

    except Exception as e:
        log.exception("Preprocess failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# Run server
# ---------------------------------------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
#final
