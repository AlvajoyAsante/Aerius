"""
Aerius - Crack & Puddle Detection Tool

Scaffold for a Streamlit app that analyzes phone/drone videos of buildings/structures using a hybrid approach:
- Local CV for puddles (offline)
- Roboflow API for crack detection
- Temporal merge, scoring (0-100), overlays, and PDF export

Current features:
1. Image Crack Test: Upload crack photos, run Roboflow inference, visualize predictions
2. Video Scaffold: Upload videos, display metadata and first frame

TODO: Integrate core/ingest.sample_frames for video frame extraction
TODO: Implement core/puddles.detect_puddles for offline puddle detection
TODO: Batch Roboflow crack detection in core/cracks_api
TODO: Implement temporal tracking and overlay rendering
TODO: Wire PDF report generation in core/report.py
"""

import os
import io
import tempfile
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

from core.cracks_api import ROBOFLOW_MODEL_ID, ROBOFLOW_CONFIDENCE, TARGET_RESIZE_WIDTH


# ============================================================================
# PAGE CONFIG & SECRETS
# ============================================================================

st.set_page_config(page_title="Aerius", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
      /* Slightly brighten headings to match the light-blue accent */
      h1, h2, h3 { color: #E6F1FF !important; }
      /* Buttons already inherit primaryColor; no change needed */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## Aerius")
st.caption(
    "Detect cracks and puddles in buildings/structures: local CV for puddles + Roboflow API for cracks → "
    "temporal merge, scoring (0–100), overlays, and 1-page PDF reports."
)

# Read API key from secrets or environment
api_key = None
try:
    api_key = st.secrets.get("ROBOFLOW_API_KEY")
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("ROBOFLOW_API_KEY")


# ============================================================================
# SIDEBAR: API KEY STATUS & CONFIG SLIDERS
# ============================================================================

with st.sidebar:
    st.header("Configuration")
    
    # API key status
    if api_key and api_key.strip():
        st.success("ROBOFLOW_API_KEY is set")
    else:
        st.warning("ROBOFLOW_API_KEY not set. Set in `.streamlit/secrets.toml` or environment.")
    
    st.divider()
    
    # Future config sliders (not used yet)
    st.subheader("Video Processing (Coming Soon)")
    sampling_fps = st.slider(
        "Sampling FPS",
        min_value=0.5,
        max_value=10.0,
        value=1.5,
        step=0.5,
        help="Frame sampling rate for video analysis (not yet implemented)"
    )
    max_frames = st.slider(
        "Max Frames",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Maximum frames to process per video (not yet implemented)"
    )


# ============================================================================
# HELPER: DRAW PREDICTIONS ON IMAGE
# ============================================================================

def draw_polygons_on_image(image: Image.Image, predictions: list) -> Image.Image:
    """
    Draw prediction polygons (if present) on a PIL Image.
    
    Assumes predictions have a 'points' key with list of {x, y} dicts.
    """
    if not predictions:
        return image
    
    draw = ImageDraw.Draw(image, "RGBA")
    for pred in predictions:
        if "points" in pred:
            points = pred["points"]
            if isinstance(points, list) and len(points) > 0:
                # Convert points to flat coordinate list for PIL polygon
                coords = [(p.get("x", 0), p.get("y", 0)) for p in points]
                if len(coords) >= 3:
                    # Draw filled polygon with semi-transparent color
                    draw.polygon(coords, fill=(0, 255, 0, 80), outline=(0, 255, 0, 255), width=2)
    return image


# ============================================================================
# HELPER: RESIZE IMAGE FOR INFERENCE
# ============================================================================

def resize_image_for_inference(image: Image.Image, target_width: int) -> Image.Image:
    """Resize image to target width while maintaining aspect ratio."""
    ratio = target_width / image.width
    new_height = int(image.height * ratio)
    return image.resize((target_width, new_height), Image.Resampling.LANCZOS)


# ============================================================================
# MAIN TABS
# ============================================================================

tab_image, tab_video = st.tabs(["Image Crack Test (Roboflow)", "Video Scaffold"])


# ============================================================================
# TAB 1: IMAGE CRACK TEST
# ============================================================================

with tab_image:
    st.header("Image Crack Test")
    
    # Check for required config
    if not api_key or not api_key.strip():
        st.error("ROBOFLOW_API_KEY not set. Please configure in `.streamlit/secrets.toml` or environment.")
    elif "xxxxx" in ROBOFLOW_MODEL_ID:
        st.error(
            "ROBOFLOW_MODEL_ID contains placeholder 'xxxxx'. "
            "Update `core/cracks_api.py` with your actual model ID from Roboflow."
        )
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a crack image",
                type=["jpg", "jpeg", "png"],
                help="JPG, JPEG, or PNG format"
            )
            
            # Sample image button
            if st.button("Use Sample Image"):
                # Look for first .jpg in samples/
                samples_dir = "samples"
                if os.path.isdir(samples_dir):
                    images = [f for f in os.listdir(samples_dir) if f.lower().endswith(".jpg")]
                    if images:
                        sample_path = os.path.join(samples_dir, images[0])
                        uploaded_file = open(sample_path, "rb")
                        st.info(f"Loaded sample: {images[0]}")
        
        with col2:
            st.subheader("Run Inference")
            run_button = st.button("Run Crack Inference", type="primary")
        
        # Process image if uploaded and button clicked
        if uploaded_file and run_button:
            # Read and display image
            image = Image.open(uploaded_file)
            st.write(f"Original size: {image.size}")
            
            # Resize for inference
            image_resized = resize_image_for_inference(image, TARGET_RESIZE_WIDTH)
            st.write(f"Resized for inference: {image_resized.size}")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image_resized.save(tmp.name, "JPEG")
                temp_path = tmp.name
            
            try:
                # Call Roboflow API
                with st.spinner("Calling Roboflow API..."):
                    client = InferenceHTTPClient(
                        api_url="https://serverless.roboflow.com",
                        api_key=api_key
                    )
                    result = client.infer(
                        temp_path,
                        model_id=ROBOFLOW_MODEL_ID
                    )
                
                # Display results
                st.success("Inference complete!")
                
                predictions = result.get("predictions", [])
                st.write(f"**Predictions found:** {len(predictions)}")
                
                if len(predictions) > 0:
                    confidences = [p.get("confidence", 0) for p in predictions]
                    avg_conf = np.mean(confidences) if confidences else 0
                    st.write(f"**Average confidence:** {avg_conf:.2f}")
                
                # Show JSON excerpt (first 2 keys + prediction count)
                excerpt = {
                    "model_id": result.get("model_id", ""),
                    "prediction_count": len(predictions),
                    "time": result.get("time", "")
                }
                st.json(excerpt)
                
                # Draw and display overlay
                if len(predictions) > 0:
                    overlay = draw_polygons_on_image(image_resized.copy(), predictions)
                    st.image(overlay, caption="Crack predictions overlayed", use_column_width=True)
                else:
                    st.info("No crack predictions found in this image.")
                    st.image(image_resized, caption="Original image", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)


# ============================================================================
# TAB 2: VIDEO SCAFFOLD
# ============================================================================

with tab_video:
    st.header("Video Analysis Scaffold")
    st.info("🔧 Video frame extraction and puddle/crack detection pipeline coming soon.")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "m4v"],
        help="MP4, MOV, or M4V format"
    )
    
    if uploaded_video:
        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("FPS", f"{fps:.2f}")
            with col2:
                st.metric("Total Frames", frame_count)
            with col3:
                st.metric("Duration (sec)", f"{duration_sec:.2f}")
            
            # Read and display first frame
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="First frame (preview)", use_column_width=True)
            
            cap.release()
            
            # Placeholder button
            st.button(
                "🔍 Analyze — Puddles Only (coming soon)",
                disabled=True,
                help="Not yet implemented. Will extract frames, detect puddles locally, and batch Roboflow cracks."
            )
            
        except Exception as e:
            st.error(f"❌ Error reading video: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(video_path):
                os.remove(video_path)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(
    "**Aerius Scaffold** — Next steps: "
    "wire `core/ingest` for frame extraction, "
    "implement `core/puddles` for local detection, "
    "batch Roboflow requests in `core/cracks_api`, "
    "temporal tracking, overlays, and PDF export."
)
