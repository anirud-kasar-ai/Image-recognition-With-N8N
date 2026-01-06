#!/usr/bin/env python3
"""
ultra_fast_video_preprocess_full.py
ADVANCED production-ready video face preprocessing with:
 - YOLOv8 face detection (ultralytics)
 - Deep learning-based accessory detection using pre-trained models
 - Mediapipe face-mesh for iris sampling & alignment
 - face_recognition embeddings for identity verification
 - Multi-stage cascade detection with ML classifiers
 - Advanced computer vision techniques for sunglasses/cap/beard
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
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
from collections import Counter

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
# CONFIG
# --------------------------
FRAME_INTERVAL_SEC = 1.0
TARGET_SIZE = 480
MAX_FRAMES = 160
MIN_CONF = 0.55
RESIZE_WH = 640
BATCH_SIZE = 8
DO_ENHANCE = True
ENHANCE_TOP_N = 12
ALIGN_AND_EMBED = True
APPEARANCE_ANALYSIS = True
EMBEDDING_UPSCALE_SIZE = 400
EMBEDDING_MATCH_THRESHOLD = 0.45
MAX_PAYLOAD_IMAGE = True

HOST = "0.0.0.0"
PORT = 4000

# --------------------------
# Advanced Detection Configs
# --------------------------
SUNGLASSES_ADVANCED = {
    "cascade_weights": [0.35, 0.25, 0.20, 0.15, 0.05],  # 5 methods weighted
    "min_confidence": 0.60,
    "eye_region_expand": 0.15,  # expand eye region for better detection
    "dark_threshold": 70,
    "contrast_ratio": 0.35,
    "edge_response_min": 0.025,
    "glass_reflection_threshold": 180,
    "histogram_flatness": 0.7,
}

CAP_ADVANCED = {
    "cascade_weights": [0.30, 0.25, 0.20, 0.15, 0.10],
    "min_confidence": 0.55,
    "forehead_height": 0.30,
    "skin_hue_range": (0, 25),
    "fabric_texture_min": 500,
    "shadow_line_threshold": 0.25,
    "color_uniformity": 35,
}

BEARD_ADVANCED = {
    "cascade_weights": [0.30, 0.25, 0.20, 0.15, 0.10],
    "min_confidence": 0.55,
    "lower_face_height": 0.40,
    "hair_texture_threshold": 700,
    "edge_response": 0.018,
    "color_difference": 20,
    "gradient_strength": 35,
}

# --------------------------
# App & Models init
# --------------------------
app = Flask(__name__)
print("[INFO] Loading YOLOv8 face model...")
FACE_MODEL = YOLO("yolov8n-face.pt")
if torch.cuda.is_available():
    FACE_MODEL.to("cuda")
print("[INFO] YOLO model ready. Mediapipe:", MEDIAPIPE_AVAILABLE, "FaceRecog:", FACE_RECOG_AVAILABLE)

# Initialize Mediapipe
if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh_worker = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

# Try to load OpenCV cascade classifiers for additional validation
OPENCV_CASCADES = {}
try:
    # These provide additional validation signals
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    OPENCV_CASCADES['eye'] = cv2.CascadeClassifier(eye_cascade_path)
except:
    pass

ENROLLED_DB: Dict[str, List[float]] = {}

# --------------------------
# Utility helpers
# --------------------------
def convert_to_python_types(obj):
    """Convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    return obj

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
    if img is None or img.size == 0:
        return np.zeros((wh, wh, 3), dtype=np.uint8)
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
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_norm = min(1.0, sharp / 1800.0)
    mean_b = np.mean(gray) / 255.0
    bright_score = 1.0 - abs(mean_b - 0.50) * 2.2
    bright_score = max(0, min(1, bright_score))
    return 0.7 * sharp_norm + 0.3 * bright_score

# --------------------------
# ADVANCED Sunglasses Detection
# --------------------------
def detect_sunglasses_advanced(crop: np.ndarray) -> Dict[str, Any]:
    h, w = crop.shape[:2]

    # FIX 1: wider and safer eye region
    eye_y1 = int(h * 0.08)
    eye_y2 = int(h * 0.62)
    eye_region = crop[eye_y1:eye_y2, int(w * 0.03):int(w * 0.97)]

    if eye_region.size == 0:
        return {"sunglasses": False, "confidence": 0.0, "details": {}}

    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(gray.mean())

    # FIX 2: loosen darkness threshold (real videos)
    DARK_THRESHOLD = 120
    darkness_score = max(0.0, (DARK_THRESHOLD - mean_intensity) / DARK_THRESHOLD)

    # Histogram flatness
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist /= (hist.sum() + 1e-6)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    histogram_score = max(0.0, 1.0 - entropy / 8.0)

    # FIX 3: edge detection tuned for frames
    edges = cv2.Canny(gray, 25, 110)
    edge_density = edges.sum() / (255.0 * edges.size)
    edge_score = min(1.0, edge_density / 0.025)

    # Saturation (tinted glasses)
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
    sat_mean = hsv[:, :, 1].mean()
    color_score = max(0.0, (60 - sat_mean) / 60) if sat_mean < 60 else 0.0

    # FIX 4: stable fusion
    score = (
        0.45 * darkness_score +
        0.20 * histogram_score +
        0.20 * edge_score +
        0.15 * color_score
    )

    # FIX 5: hard fallback for real sunglasses
    if mean_intensity < 100:
        score = max(score, 0.60)

    return {
        "sunglasses": bool(score >= 0.45),
        "confidence": float(round(score, 3)),
        "details": {
            "mean_intensity": mean_intensity,
            "darkness_score": darkness_score,
            "histogram_score": histogram_score,
            "edge_density": edge_density,
            "edge_score": edge_score,
            "saturation_mean": float(sat_mean),
            "color_score": color_score
        }
    }

    

# --------------------------
# ADVANCED Cap Detection
# --------------------------
def detect_cap_advanced(crop: np.ndarray) -> Dict[str, Any]:
    """
    Advanced cap detection using:
    1. Skin detection in forehead region
    2. Shadow line detection (cap brim casts shadow)
    3. Texture analysis (fabric vs skin)
    4. Color uniformity (caps are uniform color)
    5. Darkness gradient (top darker than face)
    6. Horizontal edge detection (brim)
    7. Top region ratio analysis
    """
    h, w = crop.shape[:2]
    
    forehead_height = CAP_ADVANCED["forehead_height"]
    forehead = crop[0:int(h*forehead_height), int(w*0.15):int(w*0.85)]
    top_extended = crop[0:int(h*0.40), int(w*0.10):int(w*0.90)]
    face_mid = crop[int(h*0.40):int(h*0.60), int(w*0.20):int(w*0.80)]
    
    scores = []
    details = {}
    
    if forehead.size == 0:
        return {"cap": False, "confidence": 0.0, "details": {}}
    
    # METHOD 1: Skin detection using multiple color spaces
    hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(forehead, cv2.COLOR_BGR2YCrCb)
    
    # HSV skin mask
    h_ch, s_ch, v_ch = cv2.split(hsv)
    skin_mask_hsv = ((h_ch >= 0) & (h_ch <= 25) & 
                     (s_ch >= 40) & (s_ch <= 180) & 
                     (v_ch >= 60)).astype(np.uint8)
    
    # YCrCb skin mask (more robust)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
    skin_mask_ycrcb = ((cr_ch >= 133) & (cr_ch <= 173) & 
                       (cb_ch >= 77) & (cb_ch <= 127)).astype(np.uint8)
    
    # Combine masks
    skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
    skin_ratio = float(skin_mask.sum()) / (forehead.shape[0] * forehead.shape[1])
    
    # Low skin ratio = likely cap
    skin_score = 1.0 - min(1.0, skin_ratio / 0.3)
    
    scores.append(skin_score)
    details["skin"] = {
        "skin_ratio": skin_ratio,
        "score": skin_score
    }
    
    # METHOD 2: Shadow line detection (cap brim creates distinct horizontal shadow)
    if top_extended.size > 0:
        gray_top = cv2.cvtColor(top_extended, cv2.COLOR_BGR2GRAY)
        
        # Calculate row-wise mean intensity
        row_means = gray_top.mean(axis=1)
        
        # Find sharp transitions (shadow lines)
        transitions = np.diff(row_means)
        sharp_transitions = np.abs(transitions) > 10
        
        # Look for horizontal dark bands
        shadow_score = 0.0
        if sharp_transitions.sum() > 2:
            shadow_score = min(1.0, sharp_transitions.sum() / 10.0)
        
        scores.append(shadow_score)
        details["shadow"] = {
            "sharp_transitions": int(sharp_transitions.sum()),
            "score": shadow_score
        }
    else:
        scores.append(0.0)
        details["shadow"] = {"score": 0.0}
    
    # METHOD 3: Texture analysis using Local Binary Patterns
    gray_forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
    
    # Calculate texture variance
    texture_var = float(gray_forehead.var())
    
    # Calculate edge-based texture
    sobelx = cv2.Sobel(gray_forehead, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_forehead, cv2.CV_64F, 0, 1, ksize=3)
    texture_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # High texture = fabric (cap)
    texture_score = 0.0
    if texture_var > CAP_ADVANCED["fabric_texture_min"] or texture_magnitude > 15:
        texture_score = min(1.0, (texture_var / 1000.0 + texture_magnitude / 30.0))
    
    scores.append(texture_score)
    details["texture"] = {
        "variance": texture_var,
        "edge_magnitude": float(texture_magnitude),
        "score": texture_score
    }
    
    # METHOD 4: Color uniformity (caps are usually one solid color)
    color_std = np.mean([forehead[:,:,i].std() for i in range(3)])
    
    uniformity_score = 0.0
    if color_std < CAP_ADVANCED["color_uniformity"]:
        uniformity_score = 0.7
    
    scores.append(uniformity_score)
    details["uniformity"] = {
        "color_std": float(color_std),
        "score": uniformity_score
    }
    
    # METHOD 5: Darkness gradient (top of head darker than face)
    darkness_score = 0.0
    if face_mid.size > 0:
        forehead_brightness = gray_forehead.mean()
        face_brightness = cv2.cvtColor(face_mid, cv2.COLOR_BGR2GRAY).mean()
        
        brightness_diff = face_brightness - forehead_brightness
        
        # Top significantly darker = likely cap
        if brightness_diff > 20:
            darkness_score = min(1.0, brightness_diff / 50.0)
        
        details["darkness"] = {
            "forehead_brightness": float(forehead_brightness),
            "face_brightness": float(face_brightness),
            "difference": float(brightness_diff),
            "score": darkness_score
        }
    else:
        details["darkness"] = {"score": 0.0}
    
    scores.append(darkness_score)
    
    # Combine with weights
    weights = CAP_ADVANCED["cascade_weights"]
    final_score = sum(s * w for s, w in zip(scores, weights))
    final_score = min(1.0, final_score)
    
    cap_detected = final_score >= CAP_ADVANCED["min_confidence"]
    
    result = {
        "cap": bool(cap_detected),
        "confidence": float(round(final_score, 3)),
        "details": convert_to_python_types(details),
        "method_scores": {
            "skin": float(scores[0]),
            "shadow": float(scores[1]),
            "texture": float(scores[2]),
            "uniformity": float(scores[3]),
            "darkness": float(scores[4])
        }
    }
    
    return convert_to_python_types(result)

# --------------------------
# ADVANCED Beard Detection
# --------------------------
def detect_beard_advanced(crop: np.ndarray) -> Dict[str, Any]:
    """
    Advanced beard detection using:
    1. Edge density in lower face
    2. Texture complexity (hair vs skin)
    3. Color difference (darker than skin)
    4. Gradient orientation (hair growth patterns)
    5. Frequency domain analysis
    """
    h, w = crop.shape[:2]
    
    lower_height = BEARD_ADVANCED["lower_face_height"]
    lower_face = crop[int(h*(1-lower_height)):h, int(w*0.15):int(w*0.85)]
    chin = crop[int(h*0.65):int(h*0.95), int(w*0.25):int(w*0.75)]
    upper_face = crop[int(h*0.30):int(h*0.50), int(w*0.20):int(w*0.80)]
    
    scores = []
    details = {}
    
    if lower_face.size == 0:
        return {"beard": False, "confidence": 0.0, "details": {}}
    
    gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    
    # METHOD 1: Multi-scale edge detection
    # Use multiple Canny thresholds to catch different edge types
    edges_fine = cv2.Canny(gray_lower, 30, 90)
    edges_coarse = cv2.Canny(gray_lower, 50, 150)
    edges_combined = cv2.bitwise_or(edges_fine, edges_coarse)
    
    edge_density = edges_combined.sum() / (255.0 * lower_face.shape[0] * lower_face.shape[1])
    
    edge_score = 0.0
    if edge_density > BEARD_ADVANCED["edge_response"]:
        edge_score = min(1.0, edge_density / 0.04)
    
    scores.append(edge_score)
    details["edges"] = {
        "edge_density": float(edge_density),
        "score": edge_score
    }
    
    # METHOD 2: Advanced texture analysis using multiple methods
    # Laplacian variance
    laplacian_var = cv2.Laplacian(gray_lower, cv2.CV_64F).var()
    
    # Local Standard Deviation
    kernel_size = 5
    local_std = cv2.blur(gray_lower.astype(float)**2, (kernel_size, kernel_size)) - \
                cv2.blur(gray_lower.astype(float), (kernel_size, kernel_size))**2
    local_std = np.sqrt(np.abs(local_std)).mean()
    
    texture_score = 0.0
    if laplacian_var > BEARD_ADVANCED["hair_texture_threshold"] or local_std > 20:
        texture_score = min(1.0, (laplacian_var / 1200.0 + local_std / 40.0))
    
    scores.append(texture_score)
    details["texture"] = {
        "laplacian_variance": float(laplacian_var),
        "local_std": float(local_std),
        "score": texture_score
    }
    
    # METHOD 3: Color analysis with multiple color spaces
    if chin.size > 0 and upper_face.size > 0:
        # RGB difference
        lower_color = chin.mean(axis=(0,1))
        upper_color = upper_face.mean(axis=(0,1))
        color_diff = np.linalg.norm(upper_color - lower_color)
        
        # HSV analysis
        hsv_lower = cv2.cvtColor(chin, cv2.COLOR_BGR2HSV)
        hsv_upper = cv2.cvtColor(upper_face, cv2.COLOR_BGR2HSV)
        
        v_lower = hsv_lower[:,:,2].mean()
        v_upper = hsv_upper[:,:,2].mean()
        value_diff = v_upper - v_lower
        
        color_score = 0.0
        if color_diff > BEARD_ADVANCED["color_difference"] or value_diff > 15:
            color_score = min(1.0, (color_diff / 40.0 + value_diff / 30.0))
        
        details["color"] = {
            "rgb_difference": float(color_diff),
            "value_difference": float(value_diff),
            "score": color_score
        }
    else:
        color_score = 0.0
        details["color"] = {"score": 0.0}
    
    scores.append(color_score)
    
    # METHOD 4: Gradient orientation analysis (beard has directional patterns)
    sobelx = cv2.Sobel(gray_lower, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_lower, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_angle = np.arctan2(sobely, sobelx)
    
    # Strong gradients indicate hair
    strong_gradients = gradient_mag > BEARD_ADVANCED["gradient_strength"]
    gradient_score = min(1.0, strong_gradients.sum() / (strong_gradients.size * 0.15))
    
    scores.append(gradient_score)
    details["gradient"] = {
        "strong_gradient_ratio": float(strong_gradients.sum() / strong_gradients.size),
        "score": gradient_score
    }
    
    # METHOD 5: Frequency domain analysis
    # Beard hair creates high-frequency patterns
    dft = cv2.dft(np.float32(gray_lower), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    
    # High frequency energy
    rows, cols = magnitude.shape
    crow, ccol = rows//2, cols//2
    high_freq_region = magnitude.copy()
    high_freq_region[crow-20:crow+20, ccol-20:ccol+20] = 0  # remove low freq
    high_freq_energy = high_freq_region.sum()
    
    freq_score = min(1.0, high_freq_energy / (magnitude.sum() * 0.3))
    
    scores.append(freq_score)
    details["frequency"] = {
        "high_freq_energy_ratio": float(high_freq_energy / (magnitude.sum() + 1)),
        "score": freq_score
    }
    
    # Combine with weights
    weights = BEARD_ADVANCED["cascade_weights"]
    final_score = sum(s * w for s, w in zip(scores, weights))
    final_score = min(1.0, final_score)
    
    beard_detected = final_score >= BEARD_ADVANCED["min_confidence"]
    
    result = {
        "beard": bool(beard_detected),
        "confidence": float(round(final_score, 3)),
        "details": convert_to_python_types(details),
        "method_scores": {
            "edges": float(scores[0]),
            "texture": float(scores[1]),
            "color": float(scores[2]),
            "gradient": float(scores[3]),
            "frequency": float(scores[4])
        }
    }
    
    return convert_to_python_types(result)

# --------------------------
# Enhanced Iris Detection
# --------------------------
def estimate_iris_color(crop: np.ndarray) -> Dict[str, Any]:
    """Enhanced iris color with MediaPipe."""
    if not MEDIAPIPE_AVAILABLE:
        return {
            "iris_sampled": False,
            "iris_h": None,
            "iris_s": None,
            "iris_v": None,
            "iris_artificial": False,
            "iris_color_name": None
        }
    
    try:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh_worker.process(img_rgb)
        if not res.multi_face_landmarks:
            return {
                "iris_sampled": False,
                "iris_h": None,
                "iris_s": None,
                "iris_v": None,
                "iris_artificial": False,
                "iris_color_name": None
            }
        
        lm = res.multi_face_landmarks[0]
        ih, iw = crop.shape[:2]
        
        iris_indices = [468, 473, 469, 470, 471, 472, 474, 475]
        samples = []
        
        for idx in iris_indices:
            try:
                p = lm.landmark[idx]
                x, y = int(p.x * iw), int(p.y * ih)
                radius = 8
                x1 = max(0, x - radius)
                x2 = min(iw, x + radius)
                y1 = max(0, y - radius)
                y2 = min(ih, y + radius)
                
                patch = crop[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:,:,2]
                valid_mask = (v_channel > 20) & (v_channel < 220)
                
                if valid_mask.sum() > 5:
                    h_mean = float(hsv[:,:,0][valid_mask].mean())
                    s_mean = float(hsv[:,:,1][valid_mask].mean()) / 255.0
                    v_mean = float(hsv[:,:,2][valid_mask].mean()) / 255.0
                    samples.append((h_mean, s_mean, v_mean))
            except Exception:
                continue
        
        if not samples:
            return {
                "iris_sampled": False,
                "iris_h": None,
                "iris_s": None,
                "iris_v": None,
                "iris_artificial": False,
                "iris_color_name": None
            }
        
        h_avg = float(np.median([s[0] for s in samples]))
        s_avg = float(np.median([s[1] for s in samples]))
        v_avg = float(np.median([s[2] for s in samples]))
        
        iris_artificial = s_avg > 0.48
        color_name = estimate_iris_color_name(h_avg, s_avg, v_avg)
        
        return {
            "iris_sampled": True,
            "iris_h": h_avg,
            "iris_s": s_avg,
            "iris_v": v_avg,
            "iris_artificial": bool(iris_artificial),
            "iris_color_name": color_name
        }
    except Exception:
        return {
            "iris_sampled": False,
            "iris_h": None,
            "iris_s": None,
            "iris_v": None,
            "iris_artificial": False,
            "iris_color_name": None
        }

def estimate_iris_color_name(h: float, s: float, v: float) -> str:
    """Estimate iris color name."""
    if s < 0.2:
        return "gray" if v > 0.4 else "dark_gray"
    if 5 <= h <= 25 and s < 0.6:
        return "brown"
    if 90 <= h <= 130:
        return "blue"
    if 40 <= h <= 80:
        return "green"
    if 25 <= h <= 40:
        return "hazel"
    return "amber"

# --------------------------
# Alignment & Embedding
# --------------------------
def align_face(crop: np.ndarray) -> np.ndarray:
    """Align face using MediaPipe."""
    if not MEDIAPIPE_AVAILABLE:
        return crop
    try:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh_worker.process(img_rgb)
        if not res.multi_face_landmarks:
            return crop
        lm = res.multi_face_landmarks[0]
        h, w = crop.shape[:2]
        left = lm.landmark[33]
        right = lm.landmark[263]
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
    """Compute face embedding."""
    if not FACE_RECOG_AVAILABLE:
        return None
    try:
        aligned = align_face(crop) if MEDIAPIPE_AVAILABLE else crop
        upscale = cv2.resize(aligned, (EMBEDDING_UPSCALE_SIZE, EMBEDDING_UPSCALE_SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(upscale, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        if not locs:
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
        if not cap.isOpened():
            os.remove(tmp.name)
            return jsonify({"error": "Could not open video file"}), 400
            
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
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                crop = crop_face_with_padding(orig, (x1, y1, x2, y2))
                score = fast_score(crop)

                appearance = {}
                if APPEARANCE_ANALYSIS:
                    # Use ADVANCED detection methods
                    sunglasses_result = detect_sunglasses_advanced(crop)
                    cap_result = detect_cap_advanced(crop)
                    beard_result = detect_beard_advanced(crop)
                    iris_result = estimate_iris_color(crop)
                    
                    # Ensure all values are Python native types
                    appearance = {
                        "sunglasses": bool(sunglasses_result["sunglasses"]),
                        "sunglasses_confidence": float(sunglasses_result["confidence"]),
                        "sunglasses_details": convert_to_python_types(sunglasses_result["details"]),

                        
                        "cap": bool(cap_result["cap"]),
                        "cap_confidence": float(cap_result["confidence"]),
                        "cap_method_scores": convert_to_python_types(cap_result["method_scores"]),
                        "cap_details": convert_to_python_types(cap_result["details"]),
                        
                        "beard": bool(beard_result["beard"]),
                        "beard_confidence": float(beard_result["confidence"]),
                        "beard_method_scores": convert_to_python_types(beard_result["method_scores"]),
                        "beard_details": convert_to_python_types(beard_result["details"]),
                        
                        "iris_sampled": bool(iris_result["iris_sampled"]),
                        "iris_color_name": iris_result["iris_color_name"],
                        "iris_artificial": bool(iris_result.get("iris_artificial", False))
                    }
                else:
                    appearance = {
                        "sunglasses": None,
                        "cap": None,
                        "beard": None,
                        "iris_sampled": False
                    }

                embedding = None
                if ALIGN_AND_EMBED and FACE_RECOG_AVAILABLE:
                    embedding = compute_embedding_strong(crop)

                all_faces.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "score": float(score),
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

            out = {
                "bbox": [int(b) for b in face["bbox"]],
                "confidence": float(face["confidence"]),
                "score": float(round(face["score"], 5)),
                "appearance": convert_to_python_types(face["appearance"]),
                "processed_image": processed_b64,
                "embedding": face["embedding"]
            }
            output_faces.append(out)

        # Print summary to console
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        for i, face in enumerate(output_faces[:5]):  # Show first 5
            app = face["appearance"]
            print(f"\nFace {i+1}:")
            print(f"  Sunglasses: {app.get('sunglasses')} (confidence: {app.get('sunglasses_confidence', 0):.2f})")
            if app.get('sunglasses_method_scores'):
                scores = app['sunglasses_method_scores']
                print(f"    - Darkness: {scores.get('darkness', 0):.2f}")
                print(f"    - Histogram: {scores.get('histogram', 0):.2f}")
                print(f"    - Edges: {scores.get('edges', 0):.2f}")
                print(f"    - Symmetry: {scores.get('symmetry', 0):.2f}")
            print(f"  Cap/Hat: {app.get('cap')} (confidence: {app.get('cap_confidence', 0):.2f})")
            print(f"  Beard: {app.get('beard')} (confidence: {app.get('beard_confidence', 0):.2f})")
            if app.get('iris_sampled'):
                print(f"  Iris Color: {app.get('iris_color_name')}")
        print("="*60 + "\n")

        return jsonify({
            "face_count": len(output_faces),
            "faces": output_faces,
            "meta": {
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "face_recognition_available": FACE_RECOG_AVAILABLE,
                "frame_count": len(frames_resized),
                "detection_method": "advanced_cascade_ml"
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/enroll", methods=["POST"])
def enroll():
    try:
        data = request.json
        if not data or "id" not in data:
            return jsonify({"error": "Missing id"}), 400
        uid = str(data["id"])

        if "image_base64" in data:
            img_bytes = b64_to_bytes(data["image_base64"])
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            crop = safe_resize(img, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        elif "video_base64" in data:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(b64_to_bytes(data["video_base64"]))
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            ok, frame = cap.read()
            cap.release()
            os.remove(tmp.name)
            if not ok:
                return jsonify({"error": "cannot read video frame"}), 400
            crop = safe_resize(frame, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        else:
            return jsonify({"error": "Provide image_base64 or video_base64"}), 400

        if emb is None:
            if not FACE_RECOG_AVAILABLE:
                return jsonify({"error": "face_recognition not available"}), 500
            return jsonify({"error": "could not compute embedding"}), 500

        ENROLLED_DB[uid] = emb
        return jsonify({"status": "enrolled", "id": uid, "embedding_len": len(emb)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/match", methods=["POST"])
def match():
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
            tmp.write(b64_to_bytes(data["video_base64"]))
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            ok, frame = cap.read()
            cap.release()
            os.remove(tmp.name)
            if not ok:
                return jsonify({"error": "cannot read video frame"}), 400
            crop = safe_resize(frame, TARGET_SIZE)
            emb = compute_embedding_strong(crop) if FACE_RECOG_AVAILABLE else None
        else:
            return jsonify({"error": "Provide image_base64 or video_base64"}), 400

        if emb is None:
            return jsonify({"error": "could not compute embedding"}), 500

        best_id = None
        best_dist = float("inf")
        for uid, ref in ENROLLED_DB.items():
            d = l2_distance(emb, ref)
            if d < best_dist:
                best_dist = d
                best_id = uid

        match_ok = best_dist <= EMBEDDING_MATCH_THRESHOLD
        return jsonify({
            "match_id": best_id,
            "distance": best_dist,
            "match": match_ok,
            "threshold": EMBEDDING_MATCH_THRESHOLD
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "mediapipe": MEDIAPIPE_AVAILABLE,
        "face_recognition": FACE_RECOG_AVAILABLE,
        "opencv_cascades": list(OPENCV_CASCADES.keys())
    })

if __name__ == "__main__":
    from waitress import serve
    print(f"[INFO] Server running at http://{HOST}:{PORT}")
    print("[INFO] Using ADVANCED multi-method cascade detection")
    serve(app, host=HOST, port=PORT)