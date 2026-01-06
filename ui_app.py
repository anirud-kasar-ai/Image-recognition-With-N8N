import streamlit as st
import requests
import pandas as pd
import os
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/input_video"
FACE_DB_PATH = r"D:\Desktop\N8N\n8n image verification\Face AUG"

st.set_page_config(
    page_title="Video Face Recognition",
    page_icon="🎥",
    layout="centered"
)

# =====================================================
# SESSION STATE
# =====================================================
if "matches_df" not in st.session_state:
    st.session_state.matches_df = None

# =====================================================
# UI
# =====================================================
st.title("🎥 Video Face Recognition System")
st.write("Upload a video. Faces will be detected and matched against the employee database.")

uploaded_video = st.file_uploader(
    "📤 Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Uploaded Video (Preview)**")
        st.video(uploaded_video)

# =====================================================
# PROCESS VIDEO
# =====================================================
if st.button("🔍 Process Video"):
    if not uploaded_video:
        st.warning("⚠️ Please upload a video first.")
        st.stop()

    with st.spinner("⏳ Processing video... Please wait"):
        try:
            response = requests.post(
                N8N_WEBHOOK_URL,
                files={
                    "data": (
                        uploaded_video.name,
                        uploaded_video.getvalue(),
                        uploaded_video.type
                    )
                },
                timeout=300
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to connect to n8n: {e}")
            st.stop()

    if response.status_code != 200:
        st.error(f"❌ n8n Error: {response.text}")
        st.stop()

    # =====================================================
    # SAFE JSON PARSING
    # =====================================================
    if not response.text.strip():
        st.warning("🚫 No human face detected in this video.")
        st.session_state.matches_df = None
        st.stop()

    try:
        results = response.json()
    except ValueError:
        st.warning("🚫 Invalid response from n8n.")
        st.session_state.matches_df = None
        st.stop()

    df = pd.DataFrame(results)

    if df.empty or "status" not in df.columns:
        st.warning("🚫 No recognizable faces found.")
        st.session_state.matches_df = None
        st.stop()

    df["score"] = df["score"].astype(float)

    recognized_df = df[df["status"] == "recognized"]

    if recognized_df.empty:
        st.warning("🚫 Faces detected but no matches found.")
        st.session_state.matches_df = None
        st.stop()

    # =====================================================
    # NORMALIZE NAMES (IMPORTANT FIX)
    # =====================================================
    recognized_df["Base_Name"] = (
        recognized_df["file_name"]
        .str.lower()
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
        .str.replace(".png", "", regex=False)
        .str.split("_")
        .str[0]
    )

    # =====================================================
    # BEST MATCH PER PERSON
    # =====================================================
    final_df = (
        recognized_df
        .groupby("Base_Name", as_index=False)
        .agg(Best_Score=("score", "max"))
        .sort_values("Best_Score", ascending=False)
        .reset_index(drop=True)
    )

    # =====================================================
    # CONFIDENCE LABEL
    # =====================================================
    def confidence_label(score):
        if score >= 0.75:
            return "High"
        elif score >= 0.60:
            return "Medium"
        return "Low"

    final_df["Confidence"] = final_df["Best_Score"].apply(confidence_label)
    final_df.rename(columns={"Base_Name": "Name"}, inplace=True)
    final_df.drop(columns=["Best_Score"], inplace=True)

    st.session_state.matches_df = final_df

# =====================================================
# DISPLAY RESULTS (PERSISTENT)
# =====================================================
if st.session_state.matches_df is not None:
    st.subheader("📊 Recognized Employees")
    st.dataframe(st.session_state.matches_df, use_container_width=True)

    # =====================================================
    # SHOW MATCHED IMAGES
    # =====================================================
    if st.checkbox("🖼️ Show matched employee images"):
        st.subheader("Matched Employee Images")

        cols = st.columns(4)
        col_idx = 0

        for name in st.session_state.matches_df["Name"]:
            image_found = False

            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                img_path = os.path.join(FACE_DB_PATH, name + ext)

                if os.path.exists(img_path):
                    img = Image.open(img_path).resize((180, 180))
                    with cols[col_idx]:
                        st.image(img, caption=name, width=180)

                    col_idx = (col_idx + 1) % 4
                    image_found = True
                    break

            if not image_found:
                st.info(f"⚠️ Image not found for {name}")
