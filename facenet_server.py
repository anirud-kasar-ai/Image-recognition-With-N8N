from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import base64
import cv2
import io
from PIL import Image

app = Flask(__name__)

# -------------------------------
# LOAD MODELS
# -------------------------------

DEFAULT_MODEL = "Facenet512"

models = {
    "Facenet512": DeepFace.build_model("Facenet512"),
}

# -------------------------------
# HELPERS
# -------------------------------

def decode_image(img_data):
    # strip header if base64 has "data:image/xxx;base64,"
    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]

    img_bytes = base64.b64decode(img_data)
    pil_img = Image.open(io.BytesIO(img_bytes))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def get_faces(img):
    """Uses DeepFace internal detector."""
    results = DeepFace.extract_faces(
        img_path=img,
        detector_backend="retinaface",
        enforce_detection=False,
        align=True
    )
    return results

def compute_embedding(face_img, model_name):
    """Pass aligned face to Facenet512."""
    rep = DeepFace.represent(
        img_path=face_img,
        model_name=model_name,
        detector_backend="skip"
    )
    return rep[0]["embedding"]

# -------------------------------
# API: EMBEDDING
# -------------------------------

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    img_b64 = data["img_base64"]
    model_name = data.get("model_name", DEFAULT_MODEL)

    img = decode_image(img_b64)
    faces = get_faces(img)

    output = []

    for f in faces:
        emb = compute_embedding(f["face"], model_name)
        output.append({
            "embedding": emb,
            "bbox": f["facial_area"]
        })

    return jsonify(output)

# -------------------------------
# API: COMPARE
# -------------------------------

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    emb1 = np.array(data["embedding1"])
    emb2 = np.array(data["embedding2"])

    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    return jsonify({
        "similarity": similarity,
        "match": similarity > 0.55
    })

# -------------------------------
# START SERVER
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
