"""
Aerius - Crack & Puddle Detection Tool

Scaffold for a Streamlit app that analyzes phone/drone videos of buildings/structures:
- Roboflow API for crack detection
- Roboflow API for puddle detection (if configured), else local CV fallback
- Temporal merge, scoring (0-100), overlays, and PDF export

Current features:
1. Image Test: Upload photos, run Roboflow inference for cracks + puddles, visualize predictions
2. Video Scaffold: Upload videos, display metadata and first frame

TODO: Integrate core/ingest.sample_frames for video frame extraction
TODO: Batch Roboflow crack & puddle detection
TODO: Implement temporal tracking and overlay rendering
TODO: Wire PDF report generation in core/report.py
"""

import os
import io
import tempfile
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

from core.cracks_api import ROBOFLOW_MODEL_ID, ROBOFLOW_CONFIDENCE, TARGET_RESIZE_WIDTH
from core.puddle_api import infer_puddles_mask_from_rgb
from core.puddles import detect_puddles, DEFAULT_CFG as PUD_CFG
from core.overlay import draw_puddles_overlay
from core.scoring import calculate_severity
from core.report import generate_pdf_report


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

tab_image, tab_video = st.tabs(["Image Test (Cracks + Puddles)", "Video Scaffold"])


# ============================================================================
# TAB 1: IMAGE TEST (CRACKS + PUDDLES)
# ============================================================================

with tab_image:
    st.header("Image Test")
    st.caption("Detects cracks (Roboflow) and puddles (Roboflow if set, else local CV).")
    
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
            
            # Store uploaded file in session state
            if uploaded_file:
                st.session_state['current_image_file'] = uploaded_file
            
            # Sample image button
            if st.button("Use Sample Image"):
                # Load sample image from assets
                sample_path = "assets/test_crack.jpg"
                if os.path.isfile(sample_path):
                    st.session_state['current_image_file'] = open(sample_path, "rb")
                    st.rerun()
        
        with col2:
            st.subheader("Run Inference")
            run_button = st.button("Run Crack Inference", type="primary")
        
        # Get image from session state
        current_image = st.session_state.get('current_image_file')
        
        # Show message if image is loaded
        if current_image:
            st.success("Image loaded. Click 'Run Crack Inference' to analyze.")
        
        # Process image if uploaded and button clicked
        if current_image and run_button:
            # Read and display image
            image = Image.open(current_image)
            st.write(f"Original size: {image.size}")
            
            # Resize for inference
            image_resized = resize_image_for_inference(image, TARGET_RESIZE_WIDTH)
            st.write(f"Resized for inference: {image_resized.size}")
            
            # Convert to RGB ndarray for puddle detection
            image_rgb = np.array(image_resized)
            
            # ================================================================
            # RUN PUDDLE DETECTION (Roboflow API preferred, else local CV)
            # ================================================================
            st.info("Running puddle detection...")
            
            # Check if puddle API is available
            try:
                from core import puddle_api
                use_puddle_api = bool(api_key) and "xxxxx" not in puddle_api.PUDDLE_ROBOFLOW_MODEL_ID
            except:
                use_puddle_api = False
            
            if use_puddle_api:
                puddle_result = infer_puddles_mask_from_rgb(image_rgb, api_key)
                puddle_mask = puddle_result["mask"]
                puddle_cov_pct = float(puddle_result["coverage_pct"])
                
                if puddle_result["error"]:
                    st.warning(f"Puddle API fallback to local CV: {puddle_result['error']}")
                    # Fallback to local CV
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    local_result = detect_puddles(image_bgr, PUD_CFG)
                    puddle_mask = local_result["mask"]
                    puddle_cov_pct = float(local_result.get("coverage_pct", 0.0))
            else:
                # Fallback to local CV
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                local_result = detect_puddles(image_bgr, PUD_CFG)
                puddle_mask = local_result["mask"]
                puddle_cov_pct = float(local_result.get("coverage_pct", 0.0))
            
            # ================================================================
            # RUN CRACK DETECTION (Roboflow API)
            # ================================================================
            st.info("Running crack detection via Roboflow...")
            
            # Save resized image to temp file for crack inference
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image_resized.save(tmp.name, "JPEG")
                temp_path = tmp.name
            
            crack_predictions = []
            crack_coverage_pct = 0.0
            crack_severity_score = 0
            crack_recommendation = "No cracks detected."
            
            try:
                # Call Roboflow API for cracks
                with st.spinner("Calling Roboflow API..."):
                    client = InferenceHTTPClient(
                        api_url="https://serverless.roboflow.com",
                        api_key=api_key
                    )
                    result = client.infer(
                        temp_path,
                        model_id=ROBOFLOW_MODEL_ID
                    )
                
                crack_predictions = result.get("predictions", [])
                
                # Calculate crack severity score and recommendation
                crack_severity_score, crack_coverage_pct, crack_recommendation = calculate_severity(
                    crack_predictions,
                    image_width=image_resized.width,
                    image_height=image_resized.height
                )
                
            except Exception as e:
                st.error(f"Crack inference failed: {str(e)}")
                crack_predictions = []
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            st.success("Analysis complete!")
            
            # Compute combined metrics
            # Puddle severity based on coverage
            puddle_severity = int(min(puddle_cov_pct * 5, 100))  # 20% coverage = 100 severity
            
            # Combined severity: 60% puddle + 40% crack
            combined_severity = int(round(0.6 * puddle_severity + 0.4 * float(crack_severity_score)))
            
            # Total defect count: cracks + presence of puddles
            total_defects = len(crack_predictions) + (1 if puddle_cov_pct > 0 else 0)
            
            # Average coverage across both detections
            avg_coverage_pct = (puddle_cov_pct + float(crack_coverage_pct or 0.0)) / 2
            
            # Display individual metrics for cracks
            st.subheader("Crack Analysis")
            crack_cols = st.columns(4)
            with crack_cols[0]:
                st.metric("Cracks Detected", len(crack_predictions))
            with crack_cols[1]:
                st.metric("Crack Coverage", f"{crack_coverage_pct:.1f}%")
            with crack_cols[2]:
                st.metric("Crack Severity", f"{crack_severity_score}/100")
            with crack_cols[3]:
                if crack_predictions:
                    avg_conf = np.mean([p.get("confidence", 0) for p in crack_predictions])
                    st.metric("Avg Confidence", f"{avg_conf:.2f}")
                else:
                    st.metric("Avg Confidence", "N/A")
            
            # Display individual metrics for puddles
            st.subheader("Puddle Analysis")
            puddle_cols = st.columns(3)
            with puddle_cols[0]:
                st.metric("Puddle Coverage", f"{puddle_cov_pct:.1f}%")
            with puddle_cols[1]:
                st.metric("Puddle Severity", f"{puddle_severity}/100")
            with puddle_cols[2]:
                st.metric("Puddle Detected", "Yes" if puddle_cov_pct > 0 else "No")
            
            # Display combined metrics
            st.subheader("Combined Assessment")
            st.metric("Total Defects", f"{total_defects} (Cracks: {len(crack_predictions)}, Puddles: {'Yes' if puddle_cov_pct > 0 else 'No'})")
            
            # Display recommendation in a highlighted box
            if combined_severity < 20:
                st.info(f"✓ Structure appears sound. Monitor regularly.")
            elif combined_severity < 60:
                st.warning(f"⚠ {crack_recommendation if crack_predictions else 'Puddle detected. Monitor.'}")
            else:
                st.error(f"🔴 {crack_recommendation if crack_predictions else 'Significant puddle. Plan remediation.'}")
            
            # ================================================================
            # BUILD COMBINED OVERLAY
            # ================================================================
            # Draw crack polygons first
            if len(crack_predictions) > 0:
                overlay_image = draw_polygons_on_image(image_resized.copy(), crack_predictions)
                overlay_np = np.array(overlay_image)
                # Convert to BGR for puddle overlay function
                combined_overlay = draw_puddles_overlay(
                    cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR),
                    puddle_mask,
                    alpha=0.4
                )
            else:
                # No cracks, just puddles
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                combined_overlay = draw_puddles_overlay(image_bgr, puddle_mask, alpha=0.4)
            
            st.image(combined_overlay, caption="Cracks + Puddles overlay", use_column_width=True)
            
            # ================================================================
            # GENERATE PDF REPORT
            # ================================================================
            st.divider()
            st.subheader("Export Report")
            
            # Prepare summary recommendation
            summary_reco = f"Puddles: {puddle_cov_pct:.1f}% • Cracks: {float(crack_coverage_pct or 0.0):.1f}%"
            
            # Convert combined overlay (BGR ndarray) back to PIL for PDF
            overlay_rgb = cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            
            pdf_buffer = generate_pdf_report(
                overlay_pil,
                combined_severity,
                avg_coverage_pct,
                total_defects,
                summary_reco
            )
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"aerius_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary"
            )


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
