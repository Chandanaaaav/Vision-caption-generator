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
    page_title="Caption Generator",
    page_icon="\u00a9\ufe0f",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

header{
    visibility:hidden;
}

body{
    background: #1a1d23;
    background-attachment: fixed;
}

.stApp{
    background: transparent;
    color: #e2e2e2;
}

.block-container{
    max-width:1200px;
    padding-top:1.5rem;
    padding-bottom:1.5rem;
    background: #23272e;
    border: 1px solid #2f353e;
    border-radius: 24px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.top-nav{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 28px 12px 28px;
    margin-bottom: 20px;
    background: #2c3e50;
    border-radius: 16px;
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(44,62,80,0.35);
}

.top-nav .nav-left{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.3px;
}

.top-nav .nav-left .logo-icon{
    width: 32px;
    height: 32px;
    background: #c9975e;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 700;
    color: #fff;
}

.top-nav .nav-right{
    display: flex;
    gap: 8px;
}

.top-nav .badge{
    background: rgba(255,255,255,0.08);
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    color: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.06);
}

.hero{
    display:flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    padding: 48px 42px 48px;
    min-height: 220px;
    border-radius: 24px;
    background: linear-gradient(135deg, #2c3e50 0%, #1a2533 100%);
    border: 1px solid #3a4a5c;
    box-shadow: 0 16px 48px rgba(0,0,0,0.3);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero::before{
    content: '';
    position: absolute;
    top: -60%;
    right: -15%;
    width: 350px;
    height: 350px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,151,94,0.06), transparent);
    pointer-events: none;
}

.hero .left{
    width: 100%;
    max-width: 100%;
    position: relative;
    z-index: 1;
}

.hero .left h1{
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 12px;
    color: #ffffff;
    line-height: 1.15;
    letter-spacing: -0.5px;
}

.hero .left .accent{
    color: #c9975e;
}

.hero .left p{
    font-size: 0.95rem;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0;
    max-width: 600px;
    line-height: 1.6;
}

.mode-btn{
    background: #2f353e;
    border: 1.5px solid #3f4651;
    border-radius: 12px;
    padding: 18px 24px;
    min-width: 180px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.mode-btn:hover{
    border-color: #c9975e;
    box-shadow: 0 8px 24px rgba(201,151,94,0.15);
    transform: translateY(-2px);
}

.mode-btn.active{
    border-color: #c9975e;
    background: #2c3e50;
    color: #ffffff;
    box-shadow: 0 8px 24px rgba(44,62,80,0.3);
}

.mode-btn .icon{
    font-size: 1.8rem;
    display: block;
    margin-bottom: 8px;
}

.mode-btn .label{
    font-weight: 600;
    font-size: 0.9rem;
    color: #d5d5d5;
}

.mode-btn.active .label{
    color: #ffffff;
}

.features{
    display:flex;
    flex-wrap: wrap;
    gap: 16px;
    margin: 20px 0 10px 0;
}

.feature{
    flex:1;
    min-width: 200px;
    background: #2f353e;
    padding: 24px 22px;
    border-radius: 16px;
    border: 1px solid #3f4651;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    transition: all 0.25s ease;
}

.feature:hover{
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(201,151,94,0.1);
    border-color: #c9975e;
}

.feature h4{
    color: #e8e8e8;
    margin-top: 14px;
    margin-bottom: 8px;
    font-size: 1rem;
    font-weight: 600;
}

.feature p{
    color: #9a9a9a;
    margin:0;
    font-size:0.85rem;
    line-height: 1.55;
}

.feature-icon{
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: #3a4150;
    color: #c9975e;
    font-size:1.2rem;
    border: 1px solid #4a5260;
}

hr{
    border:none;
    height: 1px;
    background: #3a4150;
    margin: 24px 0;
}

.mode-content{
    background: #2f353e;
    border: 1px solid #3f4651;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    margin: 10px 0 20px 0;
}

.mode-content h2{
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e8e8;
    margin-bottom: 6px;
    letter-spacing: -0.3px;
}

.mode-content p{
    color: #9a9a9a;
    font-size: 0.9rem;
}

.stButton>button{
    background: #c9975e !important;
    color:#1a1d23 !important;
    border:none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(201,151,94,0.25) !important;
    padding:0.7rem 1.3rem !important;
    font-size:0.9rem !important;
    font-weight:600 !important;
    transition: all 0.2s ease !important;
}

.stButton>button:hover{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201,151,94,0.4) !important;
    background: #d4a76e !important;
}

.stFileUploader, .stFileUploaders{
    border-radius: 12px !important;
    border: 1.5px dashed #4a5260 !important;
    background: #23272e !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}

.stFileUploader:hover{
    border-color: #c9975e !important;
    background: rgba(201,151,94,0.06) !important;
}

.stAlert{
    border-radius: 12px !important;
    border: none !important;
}

.stSuccess{
    background: #1a2e2a !important;
    color: #a7d4bd !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.3rem !important;
    border-left: 4px solid #52b788 !important;
}

.stInfo{
    background: #1a2735 !important;
    color: #a3c4e8 !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.3rem !important;
    border-left: 4px solid #4299e1 !important;
}

.stError{
    background: #2e1a1a !important;
    color: #f0a0a0 !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.3rem !important;
    border-left: 4px solid #fc8181 !important;
}

.stWarning{
    background: #2e2818 !important;
    color: #e8c97a !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.3rem !important;
    border-left: 4px solid #f6ad55 !important;
}

.footer{
    text-align: center;
    padding: 20px 20px 8px;
    color: #6a6a6a;
    font-size: 0.8rem;
    border-top: 1px solid #3a4150;
    margin-top: 20px;
}

.footer strong{
    color: #c9975e;
}

.caption-item{
    background: #23272e;
    border: 1px solid #3a4150;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.caption-item:hover{
    border-color: #c9975e;
    box-shadow: 0 4px 12px rgba(201,151,94,0.1);
}

.caption-item .time{
    font-weight: 600;
    color: #c9975e;
}

.caption-item .text{
    color: #b0b0b0;
}

.stSpinner > div{
    border-top-color: #c9975e !important;
}

.stTextInput label, .stSelectbox label, .stRadio label{
    color: #c0c0c0 !important;
}

.stMarkdown, .stWrite, .stText{
    color: #d0d0d0;
}

.stVideo{
    border-radius: 12px;
    overflow: hidden;
}

.stImage figcaption{
    color: #9a9a9a !important;
}

</style>
""", unsafe_allow_html=True)

# ===== TOP NAVIGATION BAR =====
st.markdown("""
<div class="top-nav">
    <div class="nav-left">
        <span class="logo-icon">CC</span>
        <span>Caption Generator</span>
    </div>
    <div class="nav-right">
        <span class="badge">BLIP Model</span>
        <span class="badge">CNN+LSTM</span>
    </div>
""", unsafe_allow_html=True)

# ===== HERO SECTION =====
st.markdown("""
<div class="hero">
    <div class="left">
        <h1>Vision <span class="accent">Caption</span> Generator</h1>
        <p>Generate descriptive captions from images, videos, and live camera feeds using deep learning models.</p>
    </div>
""", unsafe_allow_html=True)

# ===== FEATURE CARDS SECTION =====
st.markdown("""
<div class="features">
    <div class="feature">
        <div class="feature-icon">\U0001f5bc</div>
        <h4>Image Captioning</h4>
        <p>Upload any image and receive an accurate, context-aware caption describing its contents.</p>
    </div>
    <div class="feature">
        <div class="feature-icon">\U0001f3ac</div>
        <h4>Video Captioning</h4>
        <p>Upload a video and sample frames are analyzed to produce the best overall caption.</p>
    </div>
    <div class="feature">
        <div class="feature-icon">\U0001f4f9</div>
        <h4>Live Camera</h4>
        <p>Use your webcam in real-time to generate live captions directly from the video feed.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===== MODE SELECTOR =====
st.markdown('<p style="font-size:1.15rem; font-weight:600; color:#d0d0d0; margin-bottom:4px;">Select Input Mode</p>', unsafe_allow_html=True)

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Upload Image"

col1, col2, col3 = st.columns(3)

with col1:
    is_active = st.session_state.selected_mode == "Upload Image"
    btn_style = "active" if is_active else ""
    st.markdown(f"""
    <div class="mode-btn {btn_style}">
        <span class="icon">\U0001f5bc\ufe0f</span>
        <span class="label">Image Caption</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Image", key="hero_image", use_container_width=True):
        st.session_state.selected_mode = "Upload Image"

with col2:
    is_active = st.session_state.selected_mode == "Upload Video"
    btn_style = "active" if is_active else ""
    st.markdown(f"""
    <div class="mode-btn {btn_style}">
        <span class="icon">\U0001f3ac</span>
        <span class="label">Video Caption</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Video", key="hero_video", use_container_width=True):
        st.session_state.selected_mode = "Upload Video"

with col3:
    is_active = st.session_state.selected_mode == "Live Camera"
    btn_style = "active" if is_active else ""
    st.markdown(f"""
    <div class="mode-btn {btn_style}">
        <span class="icon">\U0001f4f9</span>
        <span class="label">Live Camera</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Camera", key="hero_live", use_container_width=True):
        st.session_state.selected_mode = "Live Camera"

input_method = st.session_state.selected_mode

# ===== MODE CONTENT SECTIONS =====
if input_method == "Upload Image":
    st.markdown('<div class="mode-content">', unsafe_allow_html=True)
    st.markdown('<h2>\U0001f5bc\ufe0f Upload Image</h2>', unsafe_allow_html=True)
    st.markdown('<p>Choose an image and generate a caption from it.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption", key="image_caption_button", use_container_width=True):
            with st.spinner("Generating caption..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.read())
                    image_path = tmp.name

                feature = extract_feature(image_path)
                caption = predict_caption(feature)

            st.markdown("### Caption")
            st.success(caption)

    st.markdown('</div>', unsafe_allow_html=True)

elif input_method == "Upload Video":
    st.markdown('<div class="mode-content">', unsafe_allow_html=True)
    st.markdown('<h2>\U0001f3ac Upload Video</h2>', unsafe_allow_html=True)
    st.markdown('<p>Upload a video and get captions generated from sample video frames.</p>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "webm"],
        key="video_upload",
    )

    if uploaded_video is not None:
        uploaded_video_bytes = uploaded_video.read()
        st.video(uploaded_video_bytes)

        if st.button("Generate Video Caption", key="video_caption_button", use_container_width=True):
            with st.spinner("Generating captions from the video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video_bytes)
                    video_path = tmp.name

                processor, model = load_blip()
                captions = generate_video_captions(video_path, processor, model)

            if captions:
                st.markdown("### Captions from video samples")
                for ts, caption in captions:
                    st.markdown(f"""
                    <div class="caption-item">
                        <span class="time">\u23f1 {ts}s</span> — <span class="text">{caption}</span>
                    </div>
                    """, unsafe_allow_html=True)

                top_caption = Counter([caption for _, caption in captions]).most_common(1)[0][0]
                st.markdown("### Suggested Caption")
                st.success(top_caption)
            else:
                st.error("Unable to generate captions from the uploaded video.")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="mode-content">', unsafe_allow_html=True)
    st.markdown('<h2>\U0001f4f9 Live Camera Caption</h2>', unsafe_allow_html=True)
    st.markdown('<p>Start the camera and see captions update while the video feed plays.</p>', unsafe_allow_html=True)

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    cols = st.columns([1, 1])
    if cols[0].button("Start Live Camera", key="start_camera", use_container_width=True):
        st.session_state.camera_active = True
    if cols[1].button("Stop Live Camera", key="stop_camera", use_container_width=True):
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

    st.markdown('</div>', unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
<div class="footer">
    Built with Streamlit &middot; BLIP &amp; CNN+LSTM models
</div>
""", unsafe_allow_html=True)
