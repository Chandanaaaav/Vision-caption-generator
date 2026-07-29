import streamlit as st
import tempfile
import cv2
import time
import numpy as np
from collections import Counter
from PIL import Image
from utils import extract_feature
from predict import predict_caption

@st.cache_resource
def load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

def generate_caption_from_frame(frame, processor, model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_video_captions(video_path, processor, model, max_samples=3):
    cap = cv2.VideoCapture(video_path)
    captions = []

    if not cap.isOpened():
        return captions

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0

    if duration <= 0:
        sample_times = [0.0]
    else:
        interval_seconds = max(1.0, duration / max_samples)
        sample_times = [min(duration, interval_seconds * i) for i in range(1, max_samples + 1)]

    for ts in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        caption = generate_caption_from_frame(frame, processor, model)
        captions.append((round(ts, 1), caption))

    cap.release()
    return captions

st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

header{
    visibility:hidden;
}

body{
    background: radial-gradient(circle at 10% 0%, rgba(99,102,241,0.22), transparent 20%),
                radial-gradient(circle at 86% 12%, rgba(236,72,153,0.18), transparent 20%),
                linear-gradient(180deg,#020617,#0b1220 32%,#111827 70%,#0f172a);
    background-attachment: fixed;
}

.stApp{
    background: transparent;
    color: #e2e8f0;
}

.block-container{
    max-width:1200px;
    padding-top:1rem;
    padding-bottom:1rem;
    background: rgba(8, 15, 32, 0.76);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 32px;
    box-shadow: 0 40px 110px rgba(15, 23, 42, 0.42);
    backdrop-filter: blur(20px);
}

.hero{
    display:flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: flex-start;
    gap: 2rem;
    padding: 54px 42px 58px;
    min-height: 340px;
    border-radius: 32px;
    background: linear-gradient(135deg, rgba(139,92,246,0.35), rgba(59,130,246,0.22) 35%, rgba(236,72,153,0.28));
    border: 1px solid rgba(167,139,250,0.22);
    box-shadow: 0 32px 95px rgba(79,70,229,0.32);
    margin-bottom: 24px;
}

.hero .left{
    width: 100%;
    max-width: 100%;
}

.hero .left h1{
    font-size: 3.6rem;
    font-weight: 800;
    margin-bottom: 18px;
    color: #f8fafc;
    line-height: 1.02;
}

.hero .left p{
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-bottom: 26px;
    max-width: 640px;
}



.features{
    display:flex;
    flex-wrap: wrap;
    gap: 18px;
    margin-top: 16px;
}

.feature{
    flex:1;
    min-width: 220px;
    background: rgba(99,102,241,0.14);
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(167,139,250,0.28);
    box-shadow: 0 18px 38px rgba(124,58,237,0.18);
    transition: transform 0.3s ease, background 0.3s ease, border-color 0.3s ease;
}

.feature:hover{
    transform: translateY(-4px);
    background: rgba(167,139,250,0.16);
    border-color: rgba(236,72,153,0.28);
}

.feature h4{
    color:#f8fafc;
    margin-top: 18px;
    margin-bottom: 10px;
    font-size: 1rem;
}

.feature p{
    color:#cbd5e1;
    margin:0;
    font-size:0.95rem;
}

.feature-icon{
    width: 50px;
    height: 50px;
    border-radius: 18px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(135deg,#ec4899,#7c3aed);
    color:#ffffff;
    font-size:1.2rem;
    box-shadow: 0 10px 24px rgba(236,72,153,0.18);
}

hr{
    border:none;
    height:1px;
    background: rgba(255,255,255,0.10);
    margin:24px 0;
}

.stRadio [role='radiogroup']{
    display:flex;
    gap:12px;
    flex-wrap:wrap;
    justify-content:center;
}

.stRadio label{
    background: rgba(99,102,241,0.14);
    color:#f8fafc;
    border: 1px solid rgba(236,72,153,0.22);
    border-radius: 999px;
    padding: 0.95rem 1.3rem;
    min-width: 170px;
    text-align:center;
    transition: all 0.2s ease;
}

.stRadio label:hover{
    background: rgba(255,255,255,0.12);
}

.stButton>button{
    background: linear-gradient(135deg,#7c3aed,#ec4899) !important;
    color:#f8fafc !important;
    border:none !important;
    border-radius:999px !important;
    box-shadow: 0 18px 45px rgba(124,58,237,0.24) !important;
    padding:0.95rem 1.45rem !important;
    font-size:1rem !important;
    font-weight:700 !important;
}

.stButton>button:hover{
    transform: translateY(-1px);
}

.stFileUploader, .stFileUploaders{
    border-radius: 24px !important;
    border: 1px dashed rgba(255,255,255,0.16) !important;
    background: rgba(255,255,255,0.04) !important;
    padding: 1rem !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="left">
        <h1>🤖 AI Vision Caption Generator</h1>
        <p>Generate intelligent captions from images, videos, and live camera feeds using Deep Learning, CNN+LSTM, and BLIP models.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Choose an input method and start captioning in style.")

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Upload Image"

hero_buttons = st.columns([1, 1, 1])

with hero_buttons[0]:
    if st.button("Image Caption", key="hero_image"):
        st.session_state.selected_mode = "Upload Image"
with hero_buttons[1]:
    if st.button("Video Caption", key="hero_video"):
        st.session_state.selected_mode = "Upload Video"
with hero_buttons[2]:
    if st.button("Live Camera", key="hero_live"):
        st.session_state.selected_mode = "Live Camera"

input_method = st.session_state.selected_mode

if input_method == "Upload Image":
    st.header("🖼️ Upload Image")
    st.write("Choose an image and generate a caption from it.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("✨ Generate Caption", key="image_caption_button"):
            with st.spinner("Generating caption..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.read())
                    image_path = tmp.name

                feature = extract_feature(image_path)
                caption = predict_caption(feature)

            st.markdown("### 📝 Caption")
            st.success(caption)

elif input_method == "Upload Video":
    st.header("🎬 Upload Video")
    st.write("Upload a video and get captions generated from sample video frames.")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "webm"],
        key="video_upload",
    )

    if uploaded_video is not None:
        uploaded_video_bytes = uploaded_video.read()
        st.video(uploaded_video_bytes)

        if st.button("✨ Generate Video Caption", key="video_caption_button"):
            with st.spinner("Generating captions from the video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video_bytes)
                    video_path = tmp.name

                processor, model = load_blip()
                captions = generate_video_captions(video_path, processor, model)

            if captions:
                st.markdown("### 📝 Captions from video samples")
                for ts, caption in captions:
                    st.write(f"**{ts}s:** {caption}")

                top_caption = Counter([caption for _, caption in captions]).most_common(1)[0][0]
                st.markdown("### 🎯 Suggested Caption")
                st.success(top_caption)
            else:
                st.error("Unable to generate captions from the uploaded video.")

else:
    st.header("📹 Live Camera Caption")
    st.write("Start the camera and see captions update while the video feed plays.")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    cols = st.columns([1, 1])
    if cols[0].button("▶ Start Live Camera", key="start_camera"):
        st.session_state.camera_active = True
    if cols[1].button("⏹ Stop Live Camera", key="stop_camera"):
        st.session_state.camera_active = False

    frame_placeholder = st.empty()
    caption_placeholder = st.empty()

    if st.session_state.camera_active:
        processor, model = load_blip()
        cap = cv2.VideoCapture(0)
        last_time = 0
        caption_placeholder.info("Starting live caption...")

        while cap.isOpened() and st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            if time.time() - last_time > 1:
                caption = generate_caption_from_frame(frame, processor, model)
                caption_placeholder.success(caption)
                last_time = time.time()

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            time.sleep(0.05)

        cap.release()
        st.session_state.camera_active = False
        caption_placeholder.warning("Live camera session ended. Click Start Live Camera to run again.")
    else:
        st.info("Click Start Live Camera to open the feed and generate captions.")
