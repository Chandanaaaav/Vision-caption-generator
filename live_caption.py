import streamlit as st
import cv2
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
@st.cache_resource
def load_blip():
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


def draw_caption(frame, caption):
    cv2.putText(
        frame,
        caption,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    st.set_page_config(
        page_title="Live Caption",
        page_icon="🎥",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        body {
            background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 25%),
                        radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.18), transparent 22%),
                        linear-gradient(180deg, #020617 0%, #131a2a 45%, #0f172a 100%);
            background-attachment: fixed;
        }
        .stApp {
            background: transparent;
            color: #e2e8f0;
        }
        .block-container {
            max-width: 900px;
            padding: 1.4rem;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 24px;
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.35);
            backdrop-filter: blur(18px);
        }
        .stButton>button {
            background: linear-gradient(135deg, #1e293b, #334155) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 999px !important;
            padding: 0.85rem 1.4rem !important;
            font-weight: 600 !important;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.35);
        }
        .caption-box {
            background: rgba(15, 23, 42, 0.95);
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.12);
            margin-top: 1rem;
            color: #e2e8f0;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #f8fafc;
        }
        .stImage img {
            border-radius: 20px;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# 🎥 Live Caption Generator")
    st.markdown(
        "Use your webcam or upload an image to generate captions with BLIP in a clean UI."
    )
    st.markdown("---")

    mode = st.radio(
        "Select input mode:",
        ["Live Camera", "Upload Image"],
        horizontal=True,
    )

    processor, model = load_blip()

    if mode == "Live Camera":
        st.header("Live Camera Captioning")
        st.write("Press the button to open your webcam and see captions generated live.")

        if "live_running" not in st.session_state:
            st.session_state.live_running = False

        start, stop = st.columns([1, 1])

        with start:
            if st.button("Start Camera", key="start_camera"):
                st.session_state.live_running = True

        with stop:
            if st.button("Stop Camera", key="stop_camera"):
                st.session_state.live_running = False

        frame_placeholder = st.empty()
        caption_placeholder = st.empty()

        if st.session_state.live_running:
            cap = cv2.VideoCapture()
            if not cap.isOpened():
                st.error("Unable to access the webcam. Please check your camera settings.")
                st.session_state.live_running = False
            else:
                st.info("Generating captions. Close the stream by pressing Stop Camera.")
                frame_count = 0
                caption = "Waiting for caption..."

                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % 25 == 0:
                        caption = generate_caption_from_frame(frame, processor, model)

                    frame_with_text = draw_caption(frame.copy(), caption)
                    frame_rgb = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    caption_placeholder.markdown(
                        f"<div class='caption-box'><strong>Caption:</strong> {caption}</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.03)

                cap.release()
                st.session_state.live_running = False

    else:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader(
            "Upload a photo to generate a caption",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Generate Caption", key="upload_caption"):
                with st.spinner("Generating caption..."):
                    caption = generate_caption_from_frame(
                        cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                        processor,
                        model,
                    )
                st.markdown(
                    f"<div class='caption-box'><strong>Caption:</strong> {caption}</div>",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    import numpy as np

    main()
