# app.py
import base64
import io
import math
import logging
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from deepface import DeepFace

# ---------------------------
# Config
# ---------------------------
HOST = "0.0.0.0"
PORT = 4000
DEBUG = True

# Model config (must be identical for embedding creation & recognition)
DEEPFACE_MODEL = "Facenet512"   # choose the model you use when storing embeddings
DETECTOR_BACKEND = "retinaface" # detector used for locating faces
EMBEDDING_SIZE = 512            # expected embedding length for Facenet512

# Crop/resize settings
TARGET_SIZE = 300
PADDING_RATIO = 0.25  # fraction of width/height to pad around detected box

# Image enhancement toggles (keep minimal; must be same for insert & query)
DO_DENOISE = False
DO_CLAHE = False
DO_SHARPEN = False

# Encoding
OUTPUT_FMT = "PNG"  # PNG is lossless: prefer for embeddings
OUTPUT_QUALITY = None  # not used for PNG

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("face_preprocess")

app = Flask(__name__)


# ---------------------------
# Utilities: Base64 <-> Image
# ---------------------------
def decode_image(img_data: str) -> np.ndarray:
    """
    Accepts base64 string (with/without data URI prefix) and returns BGR np.ndarray.
    """
    try:
        if img_data.startswith("data:"):
            img_data = img_data.split(",", 1)[1]

        # pad base64 if needed
        img_data += "=" * ((4 - len(img_data) % 4) % 4)
        img_bytes = base64.b64decode(img_data)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil)[:, :, ::-1]  # RGB -> BGR
        return arr
    except Exception:
        log.exception("Failed to decode base64 image")
        raise


def encode_image_b64(img_bgr: np.ndarray, fmt: str = OUTPUT_FMT, quality: Optional[int] = OUTPUT_QUALITY) -> str:
    """
    Convert BGR image to lossless PNG data URI (preferred for embedding pipelines).
    """
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).convert("RGB")
        buf = io.BytesIO()
        if fmt.upper() == "JPEG":
            if quality is None:
                quality = 95
            pil.save(buf, format="JPEG", quality=quality)
            prefix = "data:image/jpeg;base64,"
        else:
            pil.save(buf, format="PNG")
            prefix = "data:image/png;base64,"
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"{prefix}{b64}"
    except Exception:
        log.exception("Failed to encode image to base64")
        raise


# ---------------------------
# Quality scoring
# ---------------------------
def quality_score(img_bgr: np.ndarray) -> float:
    """
    Simple quality metric combining sharpness, brightness, and noise estimate.
    Higher is better.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    noise = float(np.std(gray))
    score = (0.55 * sharpness) + (0.35 * brightness) - (0.1 * noise)
    return float(round(score, 2))


# ---------------------------
# Cropping helpers
# ---------------------------
def perfect_square_crop(img: np.ndarray, box: Dict[str, int], padding_ratio: float = PADDING_RATIO,
                        target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Crop a tight region from the original image, pad to square, center the crop,
    and resize to target_size x target_size.
    `box` expected as {"x": int, "y": int, "w": int, "h": int}
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    # Expand box with padding
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        log.warning("Crop is empty, falling back to center square crop of original image")
        return safe_center_crop(img, target_size=target_size)

    h_crop, w_crop = crop.shape[:2]
    size = max(h_crop, w_crop)

    # create square canvas with black background
    square = np.zeros((size, size, 3), dtype=np.uint8)

    # center offsets
    off_y = (size - h_crop) // 2
    off_x = (size - w_crop) // 2
    square[off_y:off_y + h_crop, off_x:off_x + w_crop] = crop

    # final resize
    square = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return square


def safe_center_crop(img: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Safe fallback: center square crop and resize.
    """
    h, w = img.shape[:2]
    size = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    crop = img[y1:y1 + size, x1:x1 + size]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ---------------------------
# Alignment
# ---------------------------
def align_eyes(img: np.ndarray, landmarks: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """
    Rotate the image to make the line between eyes horizontal.
    landmarks expected to contain "left_eye" and "right_eye" as (x,y) tuples or lists.
    """
    try:
        left = np.array(landmarks["left_eye"], dtype=float)
        right = np.array(landmarks["right_eye"], dtype=float)

        # If landmarks are lists of points (some detectors give multiple), pick mean
        if left.ndim == 2:
            left = left.mean(axis=0)
        if right.ndim == 2:
            right = right.mean(axis=0)

        dx, dy = right[0] - left[0], right[1] - left[1]
        angle = math.degrees(math.atan2(dy, dx))

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except Exception:
        log.exception("Eye alignment failed, returning original crop")
        return img


# ---------------------------
# Enhancements (minimal)
# ---------------------------
def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


# ---------------------------
# Pipeline (minimal & consistent)
# ---------------------------
def pipeline(face_crop: np.ndarray,
             landmarks: Optional[Dict[str, Tuple[int, int]]] = None) -> Tuple[np.ndarray, float]:
    """
    Accepts a face crop (already square & resized to TARGET_SIZE), applies alignment if landmarks provided,
    light smoothing/contrast and returns processed face and quality score.
    This pipeline should be IDENTICAL for both embedding creation and recognition.
    """
    face = face_crop.copy()

    # 1) Align by eyes if landmarks present (important for consistent embeddings)
    if landmarks:
        face = align_eyes(face, landmarks)

    # 2) Optional small denoise (disabled by default; enable only if necessary)
    if DO_DENOISE:
        face = denoise(face)

    # 3) Small Gaussian blur to tame high-frequency noise (safe)
    face = cv2.GaussianBlur(face, (3, 3), 0)

    # 4) Optional CLAHE (disabled by default)
    if DO_CLAHE:
        face = apply_clahe(face)

    # 5) Slight global contrast/brightness (very mild)
    face = cv2.convertScaleAbs(face, alpha=1.02, beta=2)

    # 6) Optional sharpening
    if DO_SHARPEN:
        face = sharpen(face)

    # 7) Final resize (ensures exact shape)
    face = cv2.resize(face, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # 8) Compute quality score
    score = quality_score(face)

    return face, score


# ---------------------------
# Detection wrapper
# ---------------------------
def detect_faces_from_image(img_bgr: np.ndarray) -> List[Dict]:
    """
    Uses DeepFace.extract_faces to find faces and prepare crops + landmarks.
    Returns a list of dicts with keys: face (square crop), landmarks, facial_area (box)
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_bgr,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=False  # We perform our own alignment
        )
    except Exception:
        log.exception("DeepFace.extract_faces failed")
        faces = []

    results = []
    for f in faces:
        box = f.get("facial_area") or {}
        if not all(k in box for k in ("x", "y", "w", "h")):
            log.warning("Skipping face with invalid bbox: %s", box)
            continue

        try:
            crop = perfect_square_crop(img_bgr, box, padding_ratio=PADDING_RATIO, target_size=TARGET_SIZE)
        except Exception:
            log.exception("perfect_square_crop failed, using safe_center_crop")
            crop = safe_center_crop(img_bgr, target_size=TARGET_SIZE)

        landmarks = f.get("landmarks")  # may be None
        results.append({
            "face": crop,
            "landmarks": landmarks,
            "facial_area": box
        })

    return results


# ---------------------------
# Embedding helper
# ---------------------------
def get_embedding(img_bgr: np.ndarray) -> List[float]:
    """
    Returns a numeric embedding list for the given BGR image.
    Uses the same model & detector settings; ensure this is identical when storing & querying.
    """
    try:
        # DeepFace.represent expects RGB arrays or file paths; ensure correct input.
        # We pass enforce_detection=False so the function doesn't crash on imperfect crops.
        rep = DeepFace.represent(
            img_path=img_bgr,
            model_name=DEEPFACE_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=False
        )
        # DeepFace.represent may return a dict or list depending on version.
        # Normalize to list of floats.
        if isinstance(rep, dict):
            emb = rep.get("embedding") or rep.get("embedding_vector") or rep.get("embedding_list")
        elif isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict):
            emb = rep[0].get("embedding") or rep[0].get("embedding_vector")
        else:
            emb = rep

        # Convert to native python list of floats
        emb = np.asarray(emb, dtype=float).tolist()

        if len(emb) != EMBEDDING_SIZE:
            log.warning("Embedding length mismatch: expected %s got %s", EMBEDDING_SIZE, len(emb))
        return emb
    except Exception:
        log.exception("Failed to compute embedding")
        raise


# ---------------------------
# API Endpoints
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Input JSON:
      { "img_base64": "<base64 string or data URI>" }

    Output: list of face objects:
      [{ "original_image": "...", "processed_image": "...", "bbox": {...}, "quality_score": float }, ...]
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        img_b64 = payload.get("img_base64")
        if not img_b64:
            return jsonify({"error": "img_base64 missing"}), 400

        if not img_b64.startswith("data:image"):
            img_b64 = "data:image/jpeg;base64," + img_b64

        img = decode_image(img_b64)
        faces = detect_faces_from_image(img)
        if not faces:
            return jsonify({"message": "No face detected", "faces": []}), 200

        out = []
        for f in faces:
            processed_face, score = pipeline(f["face"], f.get("landmarks"))
            log.info("PROCESSED SHAPE: %s", processed_face.shape)
            log.info("PROCESSED MIN/MAX: %s / %s", int(processed_face.min()), int(processed_face.max()))

            out.append({
                "original_image": img_b64,
                "processed_image": encode_image_b64(processed_face),
                "bbox": f["facial_area"],
                "quality_score": score
            })

        return jsonify(out), 200

    except Exception as e:
        log.exception("Preprocess failed")
        return jsonify({"error": str(e)}), 500


@app.route("/represent", methods=["POST"])
def represent():
    """
    Create an embedding for an input image and return processed image + embedding.
    Input JSON:
      { "img_base64": "<base64 string or data URI>" }

    Output:
      { "faces": [ { "processed_image": "<data:...>", "bbox": {...}, "quality_score": float, "embedding": [..] }, ... ] }
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        img_b64 = payload.get("img_base64")
        if not img_b64:
            return jsonify({"error": "img_base64 missing"}), 400

        if not img_b64.startswith("data:image"):
            img_b64 = "data:image/jpeg;base64," + img_b64

        img = decode_image(img_b64)
        faces = detect_faces_from_image(img)
        if not faces:
            return jsonify({"message": "No face detected", "faces": []}), 200

        out = []
        for f in faces:
            processed_face, score = pipeline(f["face"], f.get("landmarks"))

            # Get embedding using same pipeline/model
            embedding = get_embedding(processed_face)

            out.append({
                "processed_image": encode_image_b64(processed_face),
                "bbox": f["facial_area"],
                "quality_score": score,
                "embedding": embedding
            })

        return jsonify({"faces": out}), 200

    except Exception as e:
        log.exception("Represent failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
