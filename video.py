#!/usr/bin/env python3
"""
ultra_fast_video_preprocess_full.py
Full production-ready video face preprocessing with:
 - YOLOv8 face detection (ultralytics)
 - Mediapipe face-mesh for iris sampling & alignment (optional but recommended)
 - face_recognition embeddings for identity verification (optional but recommended)
 - Heuristics for sunglasses, cap, beard detection
 - Aligned/upscaled embeddings to improve recognition under appearance change
 - /video_preprocess endpoint accepts {"video_base64": "..."} and returns face crops + metadata
 - /enroll endpoint (demo) to store reference embeddings in-memory
 - /match endpoint to compare a given image (base64) to enrolled identities
"""
import os
import io
import cv2
import time
import base64
import json
import tempfile
import traceback
import numpy as np
import warnings
from typing import Optional, List, Dict, Any
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch

# --------------------------
# Optional libraries
# --------------------------
MEDIAPIPE_AVAILABLE = False
FACE_RECOG_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except Exception:
    FACE_RECOG_AVAILABLE = False

# --------------------------
# Suppress noisy warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# --------------------------
# CONFIG (tune as needed)
# --------------------------
FRAME_INTERVAL_SEC = 1.0     # extract ~1 frame per sec (set <1 for more frames)
TARGET_SIZE = 320
MAX_FRAMES = 160
MIN_CONF = 0.55
RESIZE_WH = 640              # YOLO inference resolution (square)
BATCH_SIZE = 8
DO_ENHANCE = True
ENHANCE_TOP_N = 12           # how many top scored faces to enhance
ALIGN_AND_EMBED = True       # run alignment + face_recognition embedding if available
APPEARANCE_ANALYSIS = True   # run sunglasses/cap/beard/iris pipelines (costly)
EMBEDDING_UPSCALE_SIZE = 400 # upscale aligned crop before embedding
EMBEDDING_MATCH_THRESHOLD = 0.45  # euclidean distance threshold for matching (tune)
MAX_PAYLOAD_IMAGE = True     # send base64 crops in response (set False to reduce payload)

HOST = "0.0.0.0"
PORT = 4000

# Heuristic thresholds (tune on your dataset)
SUNGLASSES_BRIGHTNESS_THRESHOLD = 60
FOREHEAD_SKIN_RATIO_THRESHOLD = 0.15
BEARD_EDGE_DENSITY_THRESHOLD = 0.012
IRIS_SATURATION_THRESHOLD = 0.45

# --------------------------
# App & Models init
# --------------------------
app = Flask(__name__)
print("[INFO] Loading YOLOv8 face model...")
FACE_MODEL = YOLO("yolov8n-face.pt")  # ensure this file is available in working dir
if torch.cuda.is_available():
    FACE_MODEL.to("cuda")
print("[INFO] YOLO model ready. Mediapipe:", MEDIAPIPE_AVAILABLE, "FaceRecog:", FACE_RECOG_AVAILABLE)

# Mediapipe face-mesh worker for single-image processing (refined landmarks)
if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh_worker = mp_face_mesh.FaceMesh(static_image_mode=True,
                                                max_num_faces=1,
                                                refine_landmarks=True,
                                                min_detection_confidence=0.4,
                                                min_tracking_confidence=0.4)

# In-memory enrollment store for demo (id -> embedding)
ENROLLED_DB: Dict[str, List[float]] = {}

# --------------------------
# Utility helpers
# --------------------------
def b64_to_bytes(data_uri: str) -> bytes:
    if data_uri.startswith("data:"):
        data_uri = data_uri.split(",", 1)[1]
    data_uri += "=" * ((4 - len(data_uri) % 4) % 4)
    return base64.b64decode(data_uri)

def encode_image(img: np.ndarray) -> str:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buff = io.BytesIO()
    Image.fromarray(rgb).save(buff, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode()

def safe_resize(img: np.ndarray, wh: int) -> np.ndarray:
    return cv2.resize(img, (wh, wh), interpolation=cv2.INTER_AREA if img.shape[1] > wh else cv2.INTER_CUBIC)

def crop_face_with_padding(frame: np.ndarray, box: List[int], pad_ratio: float = 0.05) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    pad_x = max(8, int((x2 - x1) * pad_ratio))
    pad_y = max(8, int((y2 - y1) * pad_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))

def enhance_image(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.12, beta=12)

def fast_score(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_norm = min(1.0, sharp / 1800.0)
    mean_b = np.mean(gray) / 255.0
    bright_score = 1.0 - abs(mean_b - 0.50) * 2.2
    bright_score = max(0, min(1, bright_score))
    return 0.7 * sharp_norm + 0.3 * bright_score

# --------------------------
# Appearance heuristics
# --------------------------
def detect_sunglasses(crop: np.ndarray) -> Dict[str, Any]:
    h, w = crop.shape[:2]
    eye_band = crop[int(h*0.18):int(h*0.42), int(w*0.12):int(w*0.88)]
    if eye_band.size == 0:
        return {"sunglasses": False, "eye_band_brightness": None}
    gray = cv2.cvtColor(eye_band, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    sunglasses_by_brightness = mean_brightness < SUNGLASSES_BRIGHTNESS_THRESHOLD
    mp_possible = False
    if MEDIAPIPE_AVAILABLE:
        try:
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = mp_face_mesh_worker.process(img_rgb)
            mp_possible = bool(res.multi_face_landmarks)
        except Exception:
            mp_possible = False
    sunglasses = (not mp_possible) or sunglasses_by_brightness
    return {"sunglasses": bool(sunglasses), "eye_band_brightness": mean_brightness}

def detect_cap(crop: np.ndarray) -> Dict[str, Any]:
    h, w = crop.shape[:2]
    forehead = crop[0:int(h*0.25), int(w*0.2):int(w*0.8)]
    if forehead.size == 0:
        return {"cap": False, "forehead_skin_ratio": 0.0}
    hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    skin_mask = ((h_ch >= 0) & (h_ch <= 50) & (s_ch >= 50) & (v_ch >= 80)).astype(np.uint8)
    skin_ratio = float(skin_mask.sum()) / (forehead.shape[0] * forehead.shape[1])
    cap = skin_ratio < FOREHEAD_SKIN_RATIO_THRESHOLD
    return {"cap": bool(cap), "forehead_skin_ratio": skin_ratio}

def detect_beard(crop: np.ndarray) -> Dict[str, Any]:
    h, w = crop.shape[:2]
    lower = crop[int(h*0.55):int(h*0.95), int(w*0.12):int(w*0.88)]
    if lower.size == 0:
        return {"beard": False, "edge_density": 0.0}
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    edge_density = edges.sum() / (255.0 * lower.shape[0] * lower.shape[1])
    beard = edge_density > BEARD_EDGE_DENSITY_THRESHOLD
    return {"beard": bool(beard), "edge_density": float(edge_density)}

def estimate_iris_color(crop: np.ndarray) -> Dict[str, Any]:
    if not MEDIAPIPE_AVAILABLE:
        return {"iris_sampled": False, "iris_h": None, "iris_s": None, "iris_v": None, "iris_artificial": False}
    try:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh_worker.process(img_rgb)
        if not res.multi_face_landmarks:
            return {"iris_sampled": False, "iris_h": None, "iris_s": None, "iris_v": None, "iris_artificial": False}
        lm = res.multi_face_landmarks[0]
        ih, iw = crop.shape[:2]
        coords = []
        for idx in (468, 473):  # typical iris center indices
            try:
                p = lm.landmark[idx]
                x, y = int(p.x * iw), int(p.y * ih)
                coords.append((x, y))
            except Exception:
                pass
        samples = []
        for (x, y) in coords:
            x1 = max(0, x - 6); x2 = min(iw, x + 6)
            y1 = max(0, y - 6); y2 = min(ih, y + 6)
            patch = crop[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h_mean = float(hsv[:,:,0].mean())
            s_mean = float(hsv[:,:,1].mean())/255.0
            v_mean = float(hsv[:,:,2].mean())/255.0
            samples.append((h_mean, s_mean, v_mean))
        if not samples:
            return {"iris_sampled": False, "iris_h": None, "iris_s": None, "iris_v": None, "iris_artificial": False}
        h_avg = float(np.mean([s[0] for s in samples]))
        s_avg = float(np.mean([s[1] for s in samples]))
        v_avg = float(np.mean([s[2] for s in samples]))
        iris_artificial = s_avg > IRIS_SATURATION_THRESHOLD
        return {"iris_sampled": True, "iris_h": h_avg, "iris_s": s_avg, "iris_v": v_avg, "iris_artificial": bool(iris_artificial)}
    except Exception:
        return {"iris_sampled": False, "iris_h": None, "iris_s": None, "iris_v": None, "iris_artificial": False}

# --------------------------
# Alignment & Embedding
# --------------------------
def align_face(crop: np.ndarray) -> np.ndarray:
    """Rotate crop so eyes are horizontal. Return original crop if mediapipe not available."""
    if not MEDIAPIPE_AVAILABLE:
        return crop
    try:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh_worker.process(img_rgb)
        if not res.multi_face_landmarks:
            return crop
        lm = res.multi_face_landmarks[0]
        h, w = crop.shape[:2]
        # approximate eye landmarks (can be tuned)
        left = lm.landmark[33]   # left eye outer
        right = lm.landmark[263] # right eye outer
        lx, ly = int(left.x * w), int(left.y * h)
        rx, ry = int(right.x * w), int(right.y * h)
        dx = rx - lx
        dy = ry - ly
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(((lx + rx)//2, (ly + ry)//2), angle, 1.0)
        aligned = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned
    except Exception:
        return crop

def compute_embedding_strong(crop: np.ndarray) -> Optional[List[float]]:
    """Align, upscale and compute embedding if face_recognition is available."""
    if not FACE_RECOG_AVAILABLE:
        return None
    try:
        aligned = align_face(crop) if MEDIAPIPE_AVAILABLE else crop
        # upscale for more detail typical for dlib-based encoders
        upscale = cv2.resize(aligned, (EMBEDDING_UPSCALE_SIZE, EMBEDDING_UPSCALE_SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(upscale, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        if not locs:
            # fallback: treat full image as face
            locs = [(0, rgb.shape[1], rgb.shape[0], 0)]
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
        if encs:
            return [float(v) for v in encs[0]]
        return None
    except Exception:
        return None

def l2_distance(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a_arr - b_arr))

# --------------------------
# Main API endpoints
# --------------------------
@app.route("/video_preprocess", methods=["POST"])
def process_video():
    try:
        data = request.json
        if not data or "video_base64" not in data:
            return jsonify({"error": "Missing video_base64"}), 400

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(b64_to_bytes(data["video_base64"]))
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps * FRAME_INTERVAL_SEC))

        frames_resized = []
        frames_original = []
        frame_id = 0
        processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % interval != 0:
                continue
            if processed >= MAX_FRAMES:
                break
            processed += 1
            h, w = frame.shape[:2]
            resized = safe_resize(frame, RESIZE_WH)
            frames_resized.append(resized)
            frames_original.append((frame, w, h))

        cap.release()
        os.remove(tmp.name)

        if not frames_resized:
            return jsonify({"error": "No frames extracted"}), 400

        # YOLO batch inference
        yolo_results = []
        for i in range(0, len(frames_resized), BATCH_SIZE):
            batch = frames_resized[i:i+BATCH_SIZE]
            # FACE_MODEL.predict accepts list of images
            yolo_results.extend(FACE_MODEL.predict(batch, verbose=False))

        all_faces = []
        idx = 0
        for res in yolo_results:
            orig, OW, OH = frames_original[idx]
            scale_x = OW / RESIZE_WH
            scale_y = OH / RESIZE_WH
            idx += 1
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONF:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y); x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                crop = crop_face_with_padding(orig, (x1, y1, x2, y2))
                score = fast_score(crop)

                appearance = {}
                if APPEARANCE_ANALYSIS:
                    appearance.update(detect_sunglasses(crop))
                    appearance.update(detect_cap(crop))
                    appearance.update(detect_beard(crop))
                    appearance.update(estimate_iris_color(crop))
                else:
                    appearance = {"sunglasses": None, "cap": None, "beard": None, "iris_sampled": False}

                embedding = None
                if ALIGN_AND_EMBED and FACE_RECOG_AVAILABLE:
                    embedding = compute_embedding_strong(crop)

                all_faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "score": score,
                    "crop": crop,
                    "appearance": appearance,
                    "embedding": embedding
                })

        if not all_faces:
            return jsonify({"error": "No faces detected"}), 400

        # Sort by score and prepare output
        all_faces.sort(key=lambda x: x["score"], reverse=True)
        output_faces = []
        for i, face in enumerate(all_faces):
            img = face["crop"]
            if DO_ENHANCE and i < ENHANCE_TOP_N:
                img = enhance_image(img)
            processed_b64 = encode_image(img) if MAX_PAYLOAD_IMAGE else None

            app_summary = {
                "sunglasses": face["appearance"].get("sunglasses"),
                "eye_band_brightness": face["appearance"].get("eye_band_brightness"),
                "cap": face["appearance"].get("cap"),
                "forehead_skin_ratio": face["appearance"].get("forehead_skin_ratio"),
                "beard": face["appearance"].get("beard"),
                "edge_density": face["appearance"].get("edge_density"),
                "iris_sampled": face["appearance"].get("iris_sampled"),
                "iris_h": face["appearance"].get("iris_h"),
                "iris_s": face["appearance"].get("iris_s"),
                "iris_v": face["appearance"].get("iris_v"),
                "iris_artificial": face["appearance"].get("iris_artificial"),
                "embedding_present": face["embedding"] is not None
            }

            out = {
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "score": round(face["score"], 5),
                "appearance": app_summary,
                "processed_image": processed_b64,
                # include embedding optionally (be careful about privacy)
                "embedding": face["embedding"]
            }
            output_faces.append(out)

        return jsonify({
            "face_count": len(output_faces),
            "faces": output_faces,
            "meta": {
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "face_recognition_available": FACE_RECOG_AVAILABLE,
                "frame_count": len(frames_resized)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------------
# Enrollment & Matching (demo)
# --------------------------
@app.route("/enroll", methods=["POST"])
def enroll():
    """
    Demo endpoint to enroll a user embedding.
    POST JSON: {"id":"user1","image_base64":"data:..."} or {"id":"user1","video_base64":"..."}
    """
    try:
        data = request.json
        if not data or "id" not in data:
            return jsonify({"error": "Missing id"}), 400
        uid = str(data["id"])

        # prefer image, fallback to video
        if "image_base64" in data:
            img_bytes = b64_to_bytes(data["image_base64"])
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            crop = safe_resize(img, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        elif "video_base64" in data:
            # reuse preprocess pipeline but only first embedding
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(b64_to_bytes(data["video_base64"])); tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            ok, frame = cap.read()
            cap.release(); os.remove(tmp.name)
            if not ok:
                return jsonify({"error": "cannot read video frame"}), 400
            crop = safe_resize(frame, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        else:
            return jsonify({"error": "Provide image_base64 or video_base64"}), 400

        if emb is None:
            if not FACE_RECOG_AVAILABLE:
                return jsonify({"error": "face_recognition not available on server; cannot enroll"}), 500
            return jsonify({"error": "could not compute embedding"}), 500

        ENROLLED_DB[uid] = emb
        return jsonify({"status": "enrolled", "id": uid, "embedding_len": len(emb)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/match", methods=["POST"])
def match():
    """
    Match an input image against enrolled identities.
    POST JSON: {"image_base64":"..."} or {"video_base64":"..."}
    Returns top match with distance.
    """
    try:
        if not ENROLLED_DB:
            return jsonify({"error": "no enrolled identities"}), 400
        data = request.json
        if not data:
            return jsonify({"error": "no data"}), 400

        if "image_base64" in data:
            img = cv2.imdecode(np.frombuffer(b64_to_bytes(data["image_base64"]), np.uint8), cv2.IMREAD_COLOR)
            crop = safe_resize(img, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        elif "video_base64" in data:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(b64_to_bytes(data["video_base64"])); tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            ok, frame = cap.read()
            cap.release(); os.remove(tmp.name)
            if not ok:
                return jsonify({"error": "cannot read video frame"}), 400
            crop = safe_resize(frame, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        else:
            return jsonify({"error": "Provide image_base64 or video_base64"}), 400

        if emb is None:
            return jsonify({"error": "could not compute embedding; face_recognition missing or failed"}), 500

        best_id = None
        best_dist = float("inf")
        for uid, ref in ENROLLED_DB.items():
            d = l2_distance(emb, ref)
            if d < best_dist:
                best_dist = d; best_id = uid

        match_ok = best_dist <= EMBEDDING_MATCH_THRESHOLD
        return jsonify({"match_id": best_id, "distance": best_dist, "match": match_ok, "threshold": EMBEDDING_MATCH_THRESHOLD})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mediapipe": MEDIAPIPE_AVAILABLE, "face_recognition": FACE_RECOG_AVAILABLE})

# --------------------------
# Run server using waitress
# --------------------------
if __name__ == "__main__":
    from waitress import serve
    print(f"[INFO] Server running at http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
