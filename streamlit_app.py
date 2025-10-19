"""
Aerius - Crack & Water Damage Detection Tool

Scaffold for a Streamlit app that analyzes phone/drone videos of buildings/structures:
- Roboflow API for crack detection
- Roboflow API for water damage detection (if configured), else local CV fallback
- Temporal merge, scoring (0-100), overlays, and PDF export

Current features:
1. Image Test: Upload photos, run Roboflow inference for cracks + water damage, visualize predictions
2. Video Scaffold: Upload videos, display metadata and first frame
3. Live Stream: Real-time RTMP stream analysis with continuous frame capture and PDF export

TODO: Integrate core/ingest.sample_frames for video frame extraction
TODO: Batch Roboflow crack & water damage detection
TODO: Implement temporal tracking and overlay rendering
TODO: Wire PDF report generation in core/report.py
"""

import os
import io
import time
import tempfile
import base64
from pathlib import Path
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

from core.cracks_api import ROBOFLOW_MODEL_ID, ROBOFLOW_CONFIDENCE, TARGET_RESIZE_WIDTH, ROBOFLOW_MODEL_ID as CRACK_MODEL_ID, ROBOFLOW_CONFIDENCE as CRACK_CONF
from core.puddle_api import infer_puddles_mask_from_rgb
from core.puddles import detect_puddles, DEFAULT_CFG as PUD_CFG
from core.overlay import draw_puddles_overlay
from core.scoring import calculate_severity
from core.report import generate_pdf_report

# Import puddle model config if available
try:
    from core.puddle_api import PUDDLE_ROBOFLOW_MODEL_ID as PUD_MODEL_ID
except Exception:
    PUD_MODEL_ID = "xxxxx/0"


# ============================================================================
# BRAND HEADER HELPER
# ============================================================================

def _get_brand_header_html(title_text: str = "Aerius", icon_path: str = "assets/drone_icon.png", icon_em: float = 1.3):
    """
    Return HTML for the brand line with an inline PNG icon next to the title.
    icon_em controls the icon height relative to the heading font-size (keeps size equal to the old emoji).
    """
    icon_file = Path(__file__).parent / icon_path
    if icon_file.exists():
        b64 = base64.b64encode(icon_file.read_bytes()).decode("utf-8")
        # Return clean HTML only (styles defined in global CSS)
        html = f"""<div class="aerius-brand" style="--icon-em: {icon_em}em;">
          <img src="data:image/png;base64,{b64}" alt="Aerius drone icon" />
          <div class="brand-text">Aerius</div>
        </div>"""
        return html
    else:
        # Fallback to text if the file is missing
        return '<div class="brand-text">Aerius</div>'


# ============================================================================
# PAGE CONFIG & SECRETS
# ============================================================================

st.set_page_config(page_title="Aerius", page_icon="🔍", layout="wide")

# Custom CSS for professional appearance
st.markdown(
    """
    <style>
        /* Brand header styling */
        .aerius-brand {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            justify-content: center;
            margin-bottom: 10px;
        }
        .aerius-brand img {
            height: var(--icon-em, 1.3em);
            width: auto;
            vertical-align: middle;
            display: inline-block;
        }
        .aerius-brand .brand-text {
            line-height: 1;
            margin: 0;
            font-size: 2.5em;
            font-weight: 700;
            color: #60A5FA;
        }
        
        /* Hero section styling */
        .hero-header {
            background: linear-gradient(135deg, #0B1220 0%, #1a2f4f 100%);
            padding: 40px 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
        }
        .hero-title {
            font-size: 2.5em;
            font-weight: 700;
            color: #60A5FA;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 1.1em;
            color: #D1D5DB;
            margin-bottom: 5px;
        }
        .hero-caption {
            font-size: 0.95em;
            color: #9CA3AF;
        }
        
        /* Metric card styling */
        .metric-card {
            background-color: #f8fafc;
            border-left: 4px solid #60A5FA;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        /* Section styling */
        .section-header {
            color: #0B1220;
            font-size: 1.4em;
            font-weight: 600;
            margin-top: 25px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #60A5FA;
        }
        
        /* Divider */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #60A5FA, transparent);
            margin: 30px 0;
        }
        
        h1, h2 { 
            color: #0B1220 !important; 
        }
        
        /* Sidebar heading styling - make them white for contrast */
        [data-testid="sidebar"] h3,
        [data-testid="sidebar"] h3 * {
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    brand_html = _get_brand_header_html("Aerius", "assets/drone_icon.png", icon_em=1.3)
    hero_html = """<div class="hero-header">
""" + brand_html + """
            <div class="hero-subtitle">Infrastructure Inspection Platform</div>
            <div class="hero-caption">Intelligent detection for building assessment & maintenance</div>
        </div>"""
    st.markdown(hero_html, unsafe_allow_html=True)

# Read API key from secrets or environment
api_key = None
try:
    api_key = st.secrets.get("ROBOFLOW_API_KEY")
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("ROBOFLOW_API_KEY")


# ============================================================================
# SIDEBAR: SYSTEM STATUS & CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown('<div style="font-size: 1.3em; font-weight: 600; color: #FFFFFF !important; margin-bottom: 10px;">System Configuration</div>', unsafe_allow_html=True)
    
    # API key status with better styling
    st.markdown("**⚙️ API & Services**")
    if api_key and api_key.strip():
        st.success("Roboflow API Connected", icon="🟢")
    else:
        st.error("Roboflow API Not Configured", icon="❌")
        st.info("Set `ROBOFLOW_API_KEY` in `.streamlit/secrets.toml` to enable cloud detection services.")
    
    st.divider()
    
    # Advanced settings in expander
    with st.expander("Advanced Settings", expanded=False):
        st.markdown("**Video Processing (Future)**")
        sampling_fps = st.slider(
            "Sampling FPS",
            min_value=0.5,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help="Frame sampling rate for video analysis"
        )
        max_frames = st.slider(
            "Max Frames",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Maximum frames to process"
        )
    
    st.divider()
    
    # About section
    st.markdown('<div style="font-size: 1.3em; font-weight: 600; color: #FFFFFF !important; margin-bottom: 10px;">About</div>', unsafe_allow_html=True)
    st.caption("""
    **Aerius** is an intelligent infrastructure inspection platform that uses 
    computer vision and deep learning to detect cracks and water damage in buildings 
    and structures.
    
    - AI-Powered Detection
    - Detailed Analytics
    - Professional Reports
    - Real-time Streaming
    
    v1.0.0 • Built with Streamlit + Roboflow
    """)


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


def draw_bounding_boxes_on_image(image: Image.Image, predictions: list) -> Image.Image:
    """
    Draw prediction segmentation masks or bounding boxes on a PIL Image.
    
    Prioritizes polygon points (instance segmentation) for accurate crack visualization.
    Falls back to bounding boxes if polygon points aren't available.
    """
    if not predictions:
        return image
    
    draw = ImageDraw.Draw(image, "RGBA")
    for pred in predictions:
        # Prioritize polygon format (instance segmentation - most accurate for cracks)
        if "points" in pred:
            points = pred["points"]
            if isinstance(points, list) and len(points) > 0:
                coords = [(p.get("x", 0), p.get("y", 0)) for p in points]
                if len(coords) >= 3:
                    # Draw filled polygon with semi-transparent color
                    draw.polygon(coords, fill=(0, 255, 0, 80), outline=(0, 255, 0, 255), width=2)
        # Fallback to bounding box format if polygon points aren't available
        elif "x" in pred and "y" in pred and "width" in pred and "height" in pred:
            x = pred["x"]
            y = pred["y"]
            width = pred["width"]
            height = pred["height"]
            
            # Draw bounding box rectangle
            x0, y0 = x, y
            x1, y1 = x + width, y + height
            draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 0, 80), outline=(0, 255, 0, 255), width=2)
    
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

st.markdown("---")

tab_image, tab_video, tab_live = st.tabs([
    "Image Analysis",
    "Video Analysis",
    "Live Stream"
])


# ============================================================================
# TAB 1: IMAGE ANALYSIS
# ============================================================================

with tab_image:
    st.markdown("#### Single Image Analysis")
    st.markdown("Upload a photo to detect cracks and water damage with AI-powered analysis.")
    
    # Check for required config
    if not api_key or not api_key.strip():
        st.error("Roboflow API not configured. Please set ROBOFLOW_API_KEY to enable detection.")
    elif "xxxxx" in ROBOFLOW_MODEL_ID:
        st.error(
            "Crack detection model not configured. "
            "Update `core/cracks_api.py` with your Roboflow model ID."
        )
    else:
        st.divider()
        
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a crack image",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, or PNG format"
        )
        
        # Store uploaded file in session state
        if uploaded_file:
            st.session_state['current_image_file'] = uploaded_file
        
        # Buttons in a row beneath the uploader
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Use Sample Image", use_container_width=True):
                # Load sample image from assets
                sample_path = "assets/test_crack.jpg"
                if os.path.isfile(sample_path):
                    st.session_state['current_image_file'] = open(sample_path, "rb")
                    st.rerun()
        
        with col_btn2:
            run_button = st.button("Run Diagnosis", type="primary", use_container_width=True)
        
        # Get image from session state
        current_image = st.session_state.get('current_image_file')
        
        # Show message if image is loaded
        if current_image:
            st.success("Image loaded. Click 'Run Diagnosis' to analyze.")
        
        # Process image if uploaded and button clicked
        if current_image and run_button:
            # Read and display image
            image = Image.open(current_image)
            st.write(f"Original size: {image.size}")
            
            # Resize for inference
            image_resized = resize_image_for_inference(image, TARGET_RESIZE_WIDTH)
            st.write(f"Resized for inference: {image_resized.size}")
            
            # Convert to RGB ndarray for water damage detection
            image_rgb = np.array(image_resized)
            
            # ================================================================
            # RUN WATER DAMAGE DETECTION (Roboflow API preferred, else local CV)
            # ================================================================
            st.info("Running water damage detection...")
            
            # Check if water damage API is available
            try:
                from core import puddle_api
                use_water_api = bool(api_key) and "xxxxx" not in puddle_api.PUDDLE_ROBOFLOW_MODEL_ID
            except:
                use_water_api = False
            
            if use_water_api:
                water_result = infer_puddles_mask_from_rgb(image_rgb, api_key)
                water_mask = water_result["mask"]
                water_cov_pct = float(water_result["coverage_pct"])
                
                if water_result["error"]:
                    st.warning(f"Water damage API fallback to local CV: {water_result['error']}")
                    # Fallback to local CV
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    local_result = detect_puddles(image_bgr, PUD_CFG)
                    water_mask = local_result["mask"]
                    water_cov_pct = float(local_result.get("coverage_pct", 0.0))
            else:
                # Fallback to local CV
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                local_result = detect_puddles(image_bgr, PUD_CFG)
                water_mask = local_result["mask"]
                water_cov_pct = float(local_result.get("coverage_pct", 0.0))
            
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
            # Water damage severity based on coverage
            water_severity = int(min(water_cov_pct * 5, 100))  # 20% coverage = 100 severity
            
            # Combined severity: 60% water damage + 40% crack
            combined_severity = int(round(0.6 * water_severity + 0.4 * float(crack_severity_score)))
            
            # Total defect count: cracks + presence of water damage
            total_defects = len(crack_predictions) + (1 if water_cov_pct > 0 else 0)
            
            # Average coverage across both detections
            avg_coverage_pct = (water_cov_pct + float(crack_coverage_pct or 0.0)) / 2
            
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
            
            # Display individual metrics for water damage
            st.subheader("Water Damage Analysis")
            water_cols = st.columns(3)
            with water_cols[0]:
                st.metric("Water Damage Coverage", f"{water_cov_pct:.1f}%")
            with water_cols[1]:
                st.metric("Water Damage Severity", f"{water_severity}/100")
            with water_cols[2]:
                st.metric("Water Damage Detected", "Yes" if water_cov_pct > 0 else "No")
            
            # Display combined metrics
            st.subheader("Combined Assessment")
            st.metric("Total Defects", f"{total_defects} (Cracks: {len(crack_predictions)}, Water Damage: {'Yes' if water_cov_pct > 0 else 'No'})")
            
            # Display recommendation in a highlighted box
            if combined_severity < 20:
                st.info(f"✓ Structure appears sound. Monitor regularly.")
            elif combined_severity < 60:
                st.warning(f"⚠ {crack_recommendation if crack_predictions else 'Water damage detected. Monitor.'}")
            else:
                st.error(f"{crack_recommendation if crack_predictions else 'Significant water damage. Plan remediation.'}")
            
            # ================================================================
            # BUILD COMBINED OVERLAY
            # ================================================================
            # Draw crack polygons/bounding boxes first
            if len(crack_predictions) > 0:
                overlay_image = draw_bounding_boxes_on_image(image_resized.copy(), crack_predictions)
                overlay_np = np.array(overlay_image)
                # Convert to BGR for water damage overlay function
                combined_overlay = draw_puddles_overlay(
                    cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR),
                    water_mask,
                    alpha=0.4
                )
            else:
                # No cracks, just water damage
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                combined_overlay = draw_puddles_overlay(image_bgr, water_mask, alpha=0.4)
            
            st.image(combined_overlay, caption="Cracks + Water Damage overlay", use_column_width=True)
            
            # ================================================================
            # GENERATE PDF REPORT
            # ================================================================
            st.divider()
            st.subheader("Export Report")
            
            # Generate detailed recommendations based on findings
            recommendations = []
            if len(crack_predictions) > 0:
                recommendations.append(f"Cracks detected covering {crack_coverage_pct:.1f}% of inspected area with severity {int(crack_severity_score)}/100.")
                if crack_severity_score > 70:
                    recommendations.append("URGENT: Structural integrity may be compromised. Consult a structural engineer immediately.")
                elif crack_severity_score > 40:
                    recommendations.append("Monitor cracks regularly. Consider repairs to prevent further deterioration.")
                else:
                    recommendations.append("Minor cracks detected. Monitor and seal to prevent water infiltration.")
            
            if water_cov_pct > 0:
                recommendations.append(f"Water damage detected covering {water_cov_pct:.1f}% of inspected area with severity {int(water_severity)}/100.")
                if water_severity > 70:
                    recommendations.append("CRITICAL: Water intrusion present. Immediate remediation required to prevent structural damage.")
                elif water_severity > 40:
                    recommendations.append("Water damage present. Identify and seal moisture sources. Schedule repairs promptly.")
                else:
                    recommendations.append("Minor water staining detected. Monitor for progression and consider preventive sealing.")
            
            if len(crack_predictions) == 0 and water_cov_pct == 0:
                recommendations.append("No structural defects detected during this inspection. Continue regular monitoring.")
            
            # Combine recommendations into a paragraph
            summary_recommendations = " ".join(recommendations)
            
            # Convert combined overlay (BGR ndarray) back to PIL for PDF
            overlay_rgb = cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            
            pdf_buffer = generate_pdf_report(
                overlay_pil,
                combined_severity,
                avg_coverage_pct,
                total_defects,
                summary_recommendations,
                {
                    "cracks": len(crack_predictions),
                    "crack_coverage": crack_coverage_pct,
                    "crack_severity": int(crack_severity_score),
                    "water_coverage": water_cov_pct,
                    "water_severity": int(water_severity)
                }
            )
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"aerius_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary"
            )


# ============================================================================
# TAB 2: VIDEO ANALYSIS
# ============================================================================

with tab_video:
    st.markdown("#### Video Analysis with Frame Selection")
    st.markdown("""
    Upload a video and use the interactive timeline to preview and select individual frames for analysis. 
    Choose the frames you want inspected, then run detection on just those frames.
    """)
    st.divider()
    
    uploaded_video = st.file_uploader(
        "📁 Choose a video file",
        type=["mp4", "mov", "m4v"],
        help="Supported formats: MP4, MOV, M4V (up to 100MB recommended for smooth preview)"
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
            
            # Video info metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("FPS", f"{fps:.2f}")
            with col2:
                st.metric("Total Frames", frame_count)
            with col3:
                st.metric("Duration", f"{duration_sec:.1f}s")
            
            st.divider()
            
            # Initialize session state for video frame tracking
            if "selected_video_frames" not in st.session_state:
                st.session_state["selected_video_frames"] = []
            if "current_video_frame_idx" not in st.session_state:
                st.session_state["current_video_frame_idx"] = 0
            
            # Timeline slider - select current frame
            st.markdown("**Interactive Timeline**")
            current_frame_idx = st.slider(
                "Drag to preview frames",
                min_value=0,
                max_value=frame_count - 1,
                value=st.session_state.get("current_video_frame_idx", 0),
                step=1,
                label_visibility="collapsed"
            )
            st.session_state["current_video_frame_idx"] = current_frame_idx
            
            # Get and display current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                col_preview, col_actions = st.columns([3, 1])
                
                with col_preview:
                    current_time = current_frame_idx / fps if fps > 0 else 0
                    minutes = int(current_time // 60)
                    seconds = int(current_time % 60)
                    st.image(frame_rgb, caption=f"Frame {current_frame_idx} | Time: {minutes}:{seconds:02d}", use_column_width=True)
                
                with col_actions:
                    st.markdown("**Add to Selection**")
                    
                    # Check if current frame is already selected
                    is_selected = current_frame_idx in st.session_state["selected_video_frames"]
                    
                    if st.button(
                        "✅ Select Frame" if is_selected else "⬜ Select Frame",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        if is_selected:
                            st.session_state["selected_video_frames"].remove(current_frame_idx)
                            st.info(f"Frame {current_frame_idx} removed from selection")
                        else:
                            st.session_state["selected_video_frames"].append(current_frame_idx)
                            st.session_state["selected_video_frames"].sort()
                            st.success(f"Frame {current_frame_idx} added to selection")
                        st.rerun()
            
            st.divider()
            
            # Display selected frames
            if st.session_state["selected_video_frames"]:
                st.markdown(f"**Selected Frames ({len(st.session_state['selected_video_frames'])})**")
                
                selected_frames_str = ", ".join([str(f) for f in st.session_state["selected_video_frames"]])
                st.info(f"📍 Frames to analyze: {selected_frames_str}")
                
                # Option to clear selection
                if st.button("🗑️ Clear Selection", use_container_width=True):
                    st.session_state["selected_video_frames"] = []
                    st.rerun()
                
                st.divider()
                
                # Detection options
                st.markdown("**Detection Settings**")
                col1, col2 = st.columns(2)
                
                with col1:
                    enable_cracks_video = st.checkbox(
                        "🔍 Enable Crack Detection",
                        value=False,
                        disabled=not (api_key and "xxxxx" not in CRACK_MODEL_ID),
                        help="Uses AI model to detect structural cracks. Requires API key and valid model ID.",
                        key="video_enable_cracks"
                    )
                
                with col2:
                    enable_puddles_video = st.checkbox(
                        "💧 Enable Water Damage Detection",
                        value=True,
                        help="Detects water stains, puddles, and moisture damage using our puddle detection API.",
                        key="video_enable_puddles"
                    )
                
                # Validation
                if not enable_cracks_video and not enable_puddles_video:
                    st.error("⚠️ **Please enable at least one detection method**")
                else:
                    # Analyze button
                    if st.button("🔎 Analyze Selected Frames", type="primary", use_container_width=True):
                        st.markdown("**Analysis Results**")
                        
                        # Store results
                        video_analysis_results = []
                        progress_bar = st.progress(0)
                        status_container = st.empty()
                        
                        try:
                            for idx_num, frame_num in enumerate(st.session_state["selected_video_frames"]):
                                status_container.info(f"Analyzing frame {frame_num}... ({idx_num + 1}/{len(st.session_state['selected_video_frames'])})")
                                progress_bar.progress((idx_num + 1) / len(st.session_state["selected_video_frames"]))
                                
                                # Extract frame
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                                ret, analysis_frame = cap.read()
                                
                                if not ret:
                                    continue
                                
                                frame_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                                
                                # ============================================
                                # CRACK DETECTION
                                # ============================================
                                crack_predictions = []
                                crack_severity_score = 0.0
                                crack_coverage_pct = 0.0
                                
                                if enable_cracks_video and api_key and "xxxxx" not in CRACK_MODEL_ID:
                                    try:
                                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                                            cv2.imwrite(tmp_img.name, analysis_frame)
                                            temp_img_path = tmp_img.name
                                        
                                        client = InferenceHTTPClient(
                                            api_url="https://serverless.roboflow.com",
                                            api_key=api_key
                                        )
                                        result = client.infer(
                                            temp_img_path,
                                            model_id=CRACK_MODEL_ID,
                                            confidence=CRACK_CONF
                                        )
                                        
                                        crack_predictions = result.get("predictions", [])
                                        crack_severity_score, crack_coverage_pct, _ = calculate_severity(
                                            crack_predictions,
                                            image_width=analysis_frame.shape[1],
                                            image_height=analysis_frame.shape[0]
                                        )
                                        
                                        if os.path.exists(temp_img_path):
                                            os.remove(temp_img_path)
                                    
                                    except Exception as crack_err:
                                        st.warning(f"Frame {frame_num} - Crack detection error: {str(crack_err)}")
                                
                                # ============================================
                                # WATER DAMAGE DETECTION
                                # ============================================
                                puddle_mask = None
                                puddle_cov_pct = 0.0
                                
                                if enable_puddles_video:
                                    try:
                                        if api_key and "xxxxx" not in PUD_MODEL_ID:
                                            puddle_result = infer_puddles_mask_from_rgb(frame_rgb, api_key)
                                            puddle_mask = puddle_result["mask"]
                                            puddle_cov_pct = float(puddle_result["coverage_pct"])
                                            if puddle_result["error"]:
                                                local_result = detect_puddles(analysis_frame, PUD_CFG)
                                                puddle_mask = local_result["mask"]
                                                puddle_cov_pct = float(local_result.get("coverage_pct", 0.0))
                                        else:
                                            local_result = detect_puddles(analysis_frame, PUD_CFG)
                                            puddle_mask = local_result["mask"]
                                            puddle_cov_pct = float(local_result.get("coverage_pct", 0.0))
                                    except Exception as water_err:
                                        st.warning(f"Frame {frame_num} - Water damage detection error: {str(water_err)}")
                                
                                # ============================================
                                # CALCULATE COMBINED METRICS
                                # ============================================
                                puddle_severity = 0.0
                                if puddle_cov_pct > 0:
                                    puddle_severity = min(100, (puddle_cov_pct / 100) * 100)
                                
                                combined_severity = (puddle_severity * 0.6) + (crack_severity_score * 0.4)
                                
                                # Create overlay
                                overlay_frame = analysis_frame.copy()
                                
                                # Draw cracks
                                if crack_predictions:
                                    for pred in crack_predictions:
                                        x, y, w, h = int(pred['x'] - pred['width']/2), int(pred['y'] - pred['height']/2), int(pred['width']), int(pred['height'])
                                        cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                
                                # Draw water damage mask
                                if puddle_mask is not None:
                                    overlay_frame[puddle_mask > 0] = (100, 150, 255)
                                
                                overlay_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
                                
                                # Store result
                                current_time = frame_num / fps if fps > 0 else 0
                                minutes = int(current_time // 60)
                                seconds = int(current_time % 60)
                                
                                video_analysis_results.append({
                                    "frame_number": frame_num,
                                    "timestamp": f"{minutes}:{seconds:02d}",
                                    "overlay_image": overlay_rgb,
                                    "cracks_detected": len(crack_predictions),
                                    "crack_severity": crack_severity_score,
                                    "crack_coverage": crack_coverage_pct,
                                    "water_coverage": puddle_cov_pct,
                                    "water_severity": puddle_severity,
                                    "combined_severity": combined_severity
                                })
                        
                        except Exception as e:
                            st.error(f"❌ Analysis error: {str(e)}")
                        
                        finally:
                            progress_bar.empty()
                            status_container.empty()
                        
                        # Display results
                        if video_analysis_results:
                            st.success(f"✓ Analysis complete for {len(video_analysis_results)} frame(s)")
                            st.divider()
                            
                            for result in video_analysis_results:
                                with st.expander(f"Frame {result['frame_number']} | {result['timestamp']} | Severity {result['combined_severity']:.0f}/100", expanded=False):
                                    col_img, col_metrics = st.columns([2, 1])
                                    
                                    with col_img:
                                        st.image(result['overlay_image'], use_column_width=True, caption="Detection overlay")
                                    
                                    with col_metrics:
                                        if enable_cracks_video:
                                            st.metric("Cracks", result['cracks_detected'])
                                            st.metric("Crack Severity", f"{result['crack_severity']:.0f}/100")
                                            st.metric("Crack Coverage", f"{result['crack_coverage']:.1f}%")
                                        
                                        if enable_puddles_video:
                                            st.metric("Water Coverage", f"{result['water_coverage']:.1f}%")
                                            st.metric("Water Severity", f"{result['water_severity']:.0f}/100")
                                        
                                        st.metric("Combined Severity", f"{result['combined_severity']:.0f}/100")
                            
                            # PDF Export for Video Analysis
                            st.divider()
                            
                            def generate_video_analysis_pdf():
                                try:
                                    from reportlab.lib.pagesizes import letter
                                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
                                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                                    from reportlab.lib.units import inch
                                    from reportlab.lib import colors
                                    
                                    pdf_buffer = io.BytesIO()
                                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
                                    story = []
                                    styles = getSampleStyleSheet()
                                    
                                    # Title
                                    title_style = ParagraphStyle(
                                        'CustomTitle',
                                        parent=styles['Heading1'],
                                        fontSize=24,
                                        textColor=colors.HexColor("#0B1220"),
                                        spaceAfter=12
                                    )
                                    story.append(Paragraph("Video Frame Analysis Report", title_style))
                                    story.append(Spacer(1, 0.2*inch))
                                    
                                    # Summary
                                    total_cracks = sum(r['cracks_detected'] for r in video_analysis_results)
                                    frames_with_water = sum(1 for r in video_analysis_results if r['water_coverage'] > 0)
                                    avg_water_cov = sum(r['water_coverage'] for r in video_analysis_results) / len(video_analysis_results) if video_analysis_results else 0
                                    avg_severity = sum(r['combined_severity'] for r in video_analysis_results) / len(video_analysis_results) if video_analysis_results else 0
                                    
                                    summary_data = [
                                        ["Metric", "Value"],
                                        ["Total Frames Analyzed", str(len(video_analysis_results))],
                                        ["Total Cracks", str(total_cracks)],
                                        ["Frames with Water Damage", str(frames_with_water)],
                                        ["Avg Water Damage Coverage", f"{avg_water_cov:.1f}%"],
                                        ["Avg Severity Score", f"{avg_severity:.0f}/100"],
                                    ]
                                    
                                    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
                                    summary_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#60A5FA")),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                                    ]))
                                    
                                    story.append(summary_table)
                                    story.append(Spacer(1, 0.3*inch))
                                    
                                    # Detailed Frame Analysis
                                    story.append(Paragraph("Detailed Frame Analysis", styles['Heading2']))
                                    story.append(Spacer(1, 0.1*inch))
                                    
                                    for frame_result in video_analysis_results:
                                        # Frame header with severity indicator
                                        severity_indicator = "🔴 CRITICAL" if frame_result['combined_severity'] > 75 else "🟠 HIGH" if frame_result['combined_severity'] > 60 else "🟡 MEDIUM" if frame_result['combined_severity'] > 40 else "🟢 OK"
                                        frame_title = f"Frame {frame_result['frame_number']} - {frame_result['timestamp']} - {severity_indicator}"
                                        story.append(Paragraph(frame_title, styles['Heading3']))
                                        
                                        # Frame metrics table
                                        metrics_data = []
                                        if enable_cracks_video:
                                            metrics_data.extend([
                                                ["Cracks Detected", str(frame_result['cracks_detected'])],
                                                ["Crack Severity", f"{frame_result['crack_severity']:.1f}/100"],
                                                ["Crack Coverage", f"{frame_result['crack_coverage']:.1f}%"],
                                            ])
                                        
                                        if enable_puddles_video:
                                            metrics_data.extend([
                                                ["Water Damage Coverage", f"{frame_result['water_coverage']:.1f}%"],
                                                ["Water Damage Severity", f"{frame_result['water_severity']:.1f}/100"],
                                            ])
                                        
                                        metrics_data.append(["Combined Severity", f"{frame_result['combined_severity']:.0f}/100"])
                                        
                                        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
                                        metrics_table.setStyle(TableStyle([
                                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0E7FF")),
                                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                                        ]))
                                        
                                        story.append(metrics_table)
                                        story.append(Spacer(1, 0.15*inch))
                                        
                                        # Frame Recommendations
                                        if frame_result['combined_severity'] > 40:
                                            story.append(Paragraph("Recommended Actions:", styles['Heading4']))
                                            
                                            frame_recs = []
                                            if enable_cracks_video and frame_result['cracks_detected'] > 0:
                                                crack_sev = frame_result['crack_severity']
                                                if crack_sev > 70:
                                                    frame_recs.append("• URGENT: Cracks detected with high severity. Structural assessment required immediately.")
                                                elif crack_sev > 40:
                                                    frame_recs.append("• Monitor cracks closely. Schedule professional repair assessment.")
                                                else:
                                                    frame_recs.append("• Minor cracks present. Seal to prevent water infiltration.")
                                            
                                            if enable_puddles_video and frame_result['water_coverage'] > 0:
                                                water_sev = frame_result['water_severity']
                                                if water_sev > 70:
                                                    frame_recs.append("• CRITICAL: Active water intrusion detected. Locate and repair water source immediately.")
                                                elif water_sev > 40:
                                                    frame_recs.append("• Water damage present. Identify moisture sources and implement remediation.")
                                                else:
                                                    frame_recs.append("• Minor water staining. Monitor for progression and consider preventive sealing.")
                                            
                                            for rec in frame_recs:
                                                story.append(Paragraph(rec, styles['Normal']))
                                            story.append(Spacer(1, 0.1*inch))
                                        
                                        # Frame image
                                        if frame_result['overlay_image'] is not None:
                                            frame_pil = Image.fromarray(frame_result['overlay_image'])
                                            frame_buffer = io.BytesIO()
                                            frame_pil.save(frame_buffer, format='PNG')
                                            frame_buffer.seek(0)
                                            
                                            story.append(RLImage(frame_buffer, width=5*inch, height=3.75*inch))
                                        
                                        if frame_result != video_analysis_results[-1]:
                                            story.append(PageBreak())
                                    
                                    # Build PDF
                                    doc.build(story)
                                    pdf_buffer.seek(0)
                                    
                                    return pdf_buffer.getvalue()
                                
                                except Exception as e:
                                    st.error(f"❌ PDF generation error: {str(e)}")
                                    return None
                            
                            # Download button
                            st.download_button(
                                label="📥 Generate & Download PDF Report",
                                data=generate_video_analysis_pdf(),
                                file_name=f"video_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
            else:
                st.info("👉 Use the timeline above to select frames for analysis")
            
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Error reading video: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(video_path):
                os.remove(video_path)


# ============================================================================
# TAB 3: LIVE STREAM (RTMP/UDP FEED ANALYSIS)
# ============================================================================

with tab_live:
    st.markdown("#### Live Stream Analysis")
    st.markdown("""
    Connect to an RTMP or UDP stream for real-time infrastructure monitoring. 
    Captures frames at intelligent intervals, analyzes each, and exports comprehensive reports.
    """)
    st.divider()
    
    # Initialize session state for live stream control
    if "stream_running" not in st.session_state:
        st.session_state["stream_running"] = False
    if "frame_analysis_results" not in st.session_state:
        st.session_state["frame_analysis_results"] = []
    
    # LIVE STREAM CONFIGURATION
    # ====================================================================
    st.markdown("**Stream Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stream_url = st.text_input(
            "Stream URL",
            value="rtmp://localhost:1935/live/test",
            help="RTMP (e.g., rtmp://localhost:1935/live/test) or UDP (e.g., udp://127.0.0.1:5000). RTMP and UDP both supported if OpenCV was built with FFMPEG (true on macOS via pip wheels). For demos, use rtmp://localhost:1935/live/test (DJI GO 4 Custom RTMP) or a UDP restream via ffmpeg."
        )
        
        sampling_fps = st.slider(
            "Sampling FPS (frames to process per second)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
        
        resize_width = st.slider(
            "Resize Width (pixels)",
            min_value=480,
            max_value=960,
            value=640,
            step=48
        )
    
    with col2:
        # Detection method selection
        st.markdown("**Detection Methods**")
        st.markdown("<small>Select at least one detection method to analyze the stream:</small>", unsafe_allow_html=True)
        
        enable_cracks = st.checkbox(
            "🔍 Enable Crack Detection",
            value=False,
            disabled=not (api_key and "xxxxx" not in CRACK_MODEL_ID),
            help="Uses AI model to detect structural cracks. Requires API key and valid model ID.",
            key="livestream_enable_cracks"
        )
        
        enable_puddles = st.checkbox(
            "💧 Enable Water Damage Detection",
            value=True,
            help="Detects water stains, puddles, and moisture damage using our puddle detection API.",
            key="livestream_enable_puddles"
        )
        
        # Validation: at least one must be selected
        if not enable_cracks and not enable_puddles:
            st.error("⚠️ **Please enable at least one detection method (Cracks or Water Damage)**")
        
        if enable_cracks and not (api_key and "xxxxx" not in CRACK_MODEL_ID):
            st.warning("⚠ Crack detection unavailable: API key not configured or model ID invalid.")
            enable_cracks = False
    
    # ====================================================================
    # STREAM CONTROLS
    # ====================================================================
    st.markdown("---")
    st.markdown("**Capture Controls**")
    
    col_start, col_stop = st.columns(2)
    
    with col_start:
        start_button = st.button(
            "Start Capture",
            type="primary",
            use_container_width=True,
            help="Begin continuous frame capture and analysis from the stream"
        )
    
    with col_stop:
        stop_button = st.button(
            "Stop Capture",
            use_container_width=True,
            help="Stop streaming and retain captured frames"
        )
    
    if start_button:
        st.session_state["stream_running"] = True
        st.session_state["frame_analysis_results"] = []  # Reset results
        st.session_state["capture_start_time"] = time.time()
        st.session_state["frames_captured"] = 0
        st.rerun()
    
    if stop_button:
        st.session_state["stream_running"] = False
        st.rerun()  # One final rerun to show results retained
    
    # ====================================================================
    # CONTINUOUS FRAME CAPTURE & ANALYSIS (ONE FRAME PER RERUN)
    # ====================================================================
    if st.session_state.get("stream_running", False):
        # Create placeholders for live updates
        preview_ph = st.empty()
        progress_ph = st.empty()
        status_ph = st.empty()
        
        # Get session data
        frame_results = st.session_state.get("frame_analysis_results", [])
        capture_start_time = st.session_state.get("capture_start_time", time.time())
        frames_captured = st.session_state.get("frames_captured", 0)
        capture_interval = 2  # Seconds between frame captures
        current_time = time.time()
        elapsed = current_time - capture_start_time
        
        try:
            # Check if we should capture a frame
            should_capture = (frames_captured == 0) or (elapsed >= (frames_captured * capture_interval))
            
            if should_capture:
                # Connect to RTMP stream
                status_ph.info(f"🔗 Connecting to stream and reading frame {frames_captured + 1}...")
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    st.error(
                        "❌ Could not open stream. Verify:\n"
                        "- Stream URL: `rtmp://localhost:1935/live/test`\n"
                        "- RTMP server is running\n"
                        "- OBS/stream source is streaming"
                    )
                    st.session_state["stream_running"] = False
                else:
                    # Read frame with retries
                    ret, frame = None, None
                    for attempt in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            break
                        time.sleep(0.2)
                    
                    cap.release()
                    
                    if not ret or frame is None:
                        status_ph.warning("⚠ Failed to read frame from stream.")
                        st.session_state["stream_running"] = False
                    else:
                        frames_captured += 1
                        frame_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Downscale frame
                        height, width = frame.shape[:2]
                        scale = resize_width / width
                        new_height = int(height * scale)
                        frame_resized = cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_LINEAR)
                        
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        frame_bgr = frame_resized
                        
                        # ====================================================
                        # PUDDLE DETECTION (Always uses Roboflow API)
                        # ====================================================
                        puddle_mask = None
                        puddle_cov_pct = 0.0
                        
                        if enable_puddles:
                            try:
                                if api_key and "xxxxx" not in PUD_MODEL_ID:
                                    puddle_result = infer_puddles_mask_from_rgb(frame_rgb, api_key)
                                    puddle_mask = puddle_result["mask"]
                                    puddle_cov_pct = float(puddle_result["coverage_pct"])
                                    if puddle_result["error"]:
                                        local_result = detect_puddles(frame_bgr, PUD_CFG)
                                        puddle_mask = local_result["mask"]
                                        puddle_cov_pct = float(local_result.get("coverage_pct", 0.0))
                                else:
                                    local_result = detect_puddles(frame_bgr, PUD_CFG)
                                    puddle_mask = local_result["mask"]
                                    puddle_cov_pct = float(local_result.get("coverage_pct", 0.0))
                            except Exception as e:
                                status_ph.error(f"Water damage detection error: {str(e)}")
                                puddle_cov_pct = 0.0
                        
                        # ====================================================
                        # CRACK DETECTION
                        # ====================================================
                        crack_predictions = []
                        crack_severity_score = 0.0
                        crack_coverage_pct = 0.0
                        
                        if enable_cracks and api_key and "xxxxx" not in CRACK_MODEL_ID:
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                                    cv2.imwrite(tmp.name, frame_resized)
                                    temp_path = tmp.name
                                
                                client = InferenceHTTPClient(
                                    api_url="https://serverless.roboflow.com",
                                    api_key=api_key
                                )
                                result = client.infer(
                                    temp_path,
                                    model_id=CRACK_MODEL_ID,
                                    confidence=CRACK_CONF
                                )
                                
                                crack_predictions = result.get("predictions", [])
                                crack_severity_score, crack_coverage_pct, _ = calculate_severity(
                                    crack_predictions,
                                    image_width=frame_resized.shape[1],
                                    image_height=frame_resized.shape[0]
                                )
                                
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            
                            except Exception as e:
                                status_ph.error(f"Crack detection error: {str(e)}")
                        
                        # ====================================================
                        # BUILD OVERLAY
                        # ====================================================
                        try:
                            overlay = frame_rgb.copy()
                            
                            if crack_predictions:
                                overlay_pil = Image.fromarray(overlay)
                                draw = ImageDraw.Draw(overlay_pil, "RGBA")
                                
                                for pred in crack_predictions:
                                    if "points" in pred:
                                        points = pred["points"]
                                        if isinstance(points, list) and len(points) > 0:
                                            coords = [(p.get("x", 0), p.get("y", 0)) for p in points]
                                            if len(coords) >= 3:
                                                draw.polygon(coords, fill=(0, 255, 0, 80), outline=(0, 255, 0, 255), width=2)
                                    elif "x" in pred and "y" in pred and "width" in pred and "height" in pred:
                                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                                        draw.rectangle([x, y, x + w, y + h], fill=(0, 255, 0, 80), outline=(0, 255, 0, 255), width=2)
                                
                                overlay = np.array(overlay_pil)
                            
                            if puddle_mask is not None:
                                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                                overlay_bgr = draw_puddles_overlay(overlay_bgr, puddle_mask, alpha=0.4)
                                overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                        
                        except Exception as e:
                            status_ph.error(f"Overlay error: {str(e)}")
                            overlay = frame_rgb
                        
                        # ====================================================
                        # COMPUTE METRICS & STORE RESULT
                        # ====================================================
                        puddle_severity = int(min(puddle_cov_pct * 5, 100))
                        combined_severity = int(round(0.6 * puddle_severity + 0.4 * crack_severity_score))
                        
                        frame_data = {
                            "frame_number": frames_captured,
                            "timestamp": frame_timestamp,
                            "elapsed_seconds": round(elapsed, 1),
                            "puddle_coverage_pct": puddle_cov_pct,
                            "puddle_severity": puddle_severity,
                            "cracks_detected": len(crack_predictions),
                            "crack_severity": crack_severity_score,
                            "combined_severity": combined_severity,
                            "overlay_image": overlay  # RGB array
                        }
                        
                        frame_results.append(frame_data)
                        st.session_state["frame_analysis_results"] = frame_results
                        st.session_state["frames_captured"] = frames_captured
                        
                        # ====================================================
                        # UPDATE DISPLAY - LIVE PREVIEW & METRICS
                        # ====================================================
                        preview_ph.image(overlay, channels="RGB", use_column_width=True, caption=f"Frame {frames_captured} captured at {frame_timestamp}")
                        
                        # Show progress info
                        total_cracks = sum(f['cracks_detected'] for f in frame_results)
                        frames_with_water = sum(1 for f in frame_results if f['puddle_coverage_pct'] > 0)
                        progress_info = f"""
                        **Live Capture Progress**
                        - Frames Captured: {frames_captured}
                        - Elapsed Time: {round(elapsed, 1)}s
                        - Total Cracks Found: {total_cracks}
                        - Frames with Water Damage: {frames_with_water}
                        """
                        progress_ph.markdown(progress_info)
                        
                        # Show latest frame metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"Frame {frames_captured}: Water %", f"{puddle_cov_pct:.1f}%")
                        with col2:
                            st.metric(f"Frame {frames_captured}: Cracks", len(crack_predictions))
                        with col3:
                            st.metric(f"Frame {frames_captured}: Water Severity", f"{puddle_severity}/100")
                        with col4:
                            st.metric(f"Frame {frames_captured}: Combined Severity", f"{combined_severity}/100")
                        
                        status_ph.success(f"✓ Frame {frames_captured} analyzed. Auto-capturing next frame in {capture_interval}s...")
                        
                        # Trigger next rerun after capture_interval seconds
                        time.sleep(1)  # Brief delay before triggering next rerun
                        st.rerun()
            else:
                # Not time to capture yet, show current results and rerun after 1 second
                status_ph.info(f"⏳ Next frame in {round((frames_captured * capture_interval) - elapsed)}s...")
                progress_ph.markdown(f"""
                **Live Capture Progress**
                - Frames Captured: {frames_captured}
                - Elapsed Time: {round(elapsed, 1)}s
                - Total Cracks Found: {sum(f['cracks_detected'] for f in frame_results)}
                - Frames with Water Damage: {sum(1 for f in frame_results if f['puddle_coverage_pct'] > 0)}
                """)
                
                if len(frame_results) > 0:
                    latest = frame_results[-1]
                    preview_ph.image(latest['overlay_image'], channels="RGB", use_column_width=True, caption=f"Latest: Frame {latest['frame_number']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"Frame {latest['frame_number']}: Water %", f"{latest['puddle_coverage_pct']:.1f}%")
                    with col2:
                        st.metric(f"Frame {latest['frame_number']}: Cracks", latest['cracks_detected'])
                    with col3:
                        st.metric(f"Frame {latest['frame_number']}: Water Severity", f"{latest['puddle_severity']}/100")
                    with col4:
                        st.metric(f"Frame {latest['frame_number']}: Combined Severity", f"{latest['combined_severity']}/100")
                
                # Trigger rerun after 1 second to check if it's time to capture
                time.sleep(1)
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Stream error: {str(e)}")
            st.session_state["stream_running"] = False
    
    # ====================================================================
    # DISPLAY RETAINED RESULTS AFTER STOP & PDF EXPORT
    # ====================================================================
    elif len(st.session_state.get("frame_analysis_results", [])) > 0:
        frame_results = st.session_state["frame_analysis_results"]
        
        st.divider()
        st.markdown("<h3 style='color: #0B1220;'>Live Capture Results</h3>", unsafe_allow_html=True)
        
        # Show summary with professional metrics
        total_cracks = sum(f['cracks_detected'] for f in frame_results)
        frames_with_water = sum(1 for f in frame_results if f['puddle_coverage_pct'] > 0)
        avg_water_cov = sum(f['puddle_coverage_pct'] for f in frame_results) / len(frame_results) if frame_results else 0
        avg_severity = sum(f['combined_severity'] for f in frame_results) / len(frame_results) if frame_results else 0
        
        # Create professional summary cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E3A5F 0%, #0B1220 100%); 
                        border-left: 4px solid #60A5FA; 
                        padding: 20px; border-radius: 8px; text-align: center;'>
                <div style='font-size: 28px; font-weight: bold; color: #60A5FA;'>{len(frame_results)}</div>
                <div style='font-size: 12px; color: #93C5FD; margin-top: 5px;'>Frames Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E3A5F 0%, #0B1220 100%); 
                        border-left: 4px solid #60A5FA; 
                        padding: 20px; border-radius: 8px; text-align: center;'>
                <div style='font-size: 28px; font-weight: bold; color: #60A5FA;'>{total_cracks}</div>
                <div style='font-size: 12px; color: #93C5FD; margin-top: 5px;'>Cracks Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E3A5F 0%, #0B1220 100%); 
                        border-left: 4px solid #60A5FA; 
                        padding: 20px; border-radius: 8px; text-align: center;'>
                <div style='font-size: 28px; font-weight: bold; color: #60A5FA;'>{frames_with_water}</div>
                <div style='font-size: 12px; color: #93C5FD; margin-top: 5px;'>Water Damage Instances</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m4:
            severity_color = "#E0F2FE" if avg_severity > 75 else "#BFDBFE" if avg_severity > 50 else "#93C5FD"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E3A5F 0%, #0B1220 100%); 
                        border-left: 4px solid #60A5FA; 
                        padding: 20px; border-radius: 8px; text-align: center;'>
                <div style='font-size: 28px; font-weight: bold; color: {severity_color};'>{avg_severity:.0f}</div>
                <div style='font-size: 12px; color: #93C5FD; margin-top: 5px;'>Avg Severity</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show all frames in expandable sections
        st.markdown("<h4 style='color: #0B1220; margin-top: 30px;'>Frame-by-Frame Analysis</h4>", unsafe_allow_html=True)
        
        for idx, frame_data in enumerate(frame_results, 1):
            severity = frame_data['combined_severity']
            severity_badge = "High" if severity > 75 else "Medium" if severity > 50 else "Low"
            
            with st.expander(f"Frame {frame_data['frame_number']} • {severity_badge} • Severity {severity}/100", expanded=False):
                col_img, col_metrics = st.columns([2, 1])
                
                with col_img:
                    st.image(frame_data['overlay_image'], channels="RGB", use_column_width=True, caption=f"Capture: {frame_data['timestamp']}")
                
                with col_metrics:
                    st.markdown("""
                    <div style='background: #F8FAFC; padding: 15px; border-radius: 8px;'>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Water Damage Coverage", f"{frame_data['puddle_coverage_pct']:.1f}%")
                    st.metric("Water Damage Severity", f"{frame_data['puddle_severity']}/100")
                    st.metric("Cracks Detected", frame_data['cracks_detected'])
                    st.metric("Crack Severity", f"{frame_data['crack_severity']:.1f}/100")
                    st.metric("Combined Severity", f"{frame_data['combined_severity']}/100")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # PDF Export Button
        st.divider()
        
        # Initialize PDF generation in session state if needed
        if "pdf_buffer" not in st.session_state:
            st.session_state.pdf_buffer = None
        
        def generate_pdf():
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
                story = []
                styles = getSampleStyleSheet()
                
                # Title Page
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor("#0B1220"),
                    spaceAfter=12
                )
                story.append(Paragraph("Live Stream Inspection Report", title_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Session Summary
                total_cracks = sum(f['cracks_detected'] for f in frame_results)
                frames_with_water = sum(1 for f in frame_results if f['puddle_coverage_pct'] > 0)
                avg_water_cov = sum(f['puddle_coverage_pct'] for f in frame_results) / len(frame_results) if frame_results else 0
                avg_severity = sum(f['combined_severity'] for f in frame_results) / len(frame_results) if frame_results else 0
                
                summary_data = [
                    ["Metric", "Value"],
                    ["Total Frames", str(len(frame_results))],
                    ["Total Cracks", str(total_cracks)],
                    ["Frames with Water Damage", str(frames_with_water)],
                    ["Avg Water Damage Coverage", f"{avg_water_cov:.1f}%"],
                    ["Avg Severity Score", f"{avg_severity:.0f}/100"],
                ]
                
                summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#60A5FA")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                
                story.append(summary_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Problematic Frames Table (Quick Navigation)
                problematic_frames = [f for f in frame_results if f['combined_severity'] > 40]
                if problematic_frames:
                    story.append(Paragraph("Frames Requiring Attention", styles['Heading3']))
                    story.append(Spacer(1, 0.1*inch))
                    
                    issue_data = [["Frame", "Severity", "Issues Found"]]
                    for frame in problematic_frames:
                        issues = []
                        if frame['cracks_detected'] > 0:
                            issues.append(f"{frame['cracks_detected']} crack(s)")
                        if frame['puddle_coverage_pct'] > 0:
                            issues.append(f"{frame['puddle_coverage_pct']:.1f}% water damage")
                        
                        severity_label = "CRITICAL" if frame['combined_severity'] > 75 else "HIGH" if frame['combined_severity'] > 60 else "MEDIUM"
                        issue_data.append([
                            f"Frame {frame['frame_number']}",
                            f"{severity_label} ({frame['combined_severity']}/100)",
                            ", ".join(issues)
                        ])
                    
                    issues_table = Table(issue_data, colWidths=[1.2*inch, 1.5*inch, 2.8*inch])
                    issues_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F59E0B")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FEF3C7"), colors.white]),
                    ]))
                    story.append(issues_table)
                    story.append(PageBreak())
                
                # Frame-by-Frame Analysis
                story.append(Paragraph("Detailed Frame Analysis", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                for frame_data in frame_results:
                    # Frame header with severity indicator
                    severity_indicator = "🔴 CRITICAL" if frame_data['combined_severity'] > 75 else "🟠 HIGH" if frame_data['combined_severity'] > 60 else "🟡 MEDIUM" if frame_data['combined_severity'] > 40 else "🟢 OK"
                    frame_title = f"Frame {frame_data['frame_number']} - {frame_data['timestamp']} - {severity_indicator}"
                    story.append(Paragraph(frame_title, styles['Heading3']))
                    
                    # Frame metrics table
                    metrics_data = [
                        ["Water Damage Coverage", f"{frame_data['puddle_coverage_pct']:.1f}%"],
                        ["Water Damage Severity", f"{frame_data['puddle_severity']}/100"],
                        ["Cracks Detected", str(frame_data['cracks_detected'])],
                        ["Crack Severity", f"{frame_data['crack_severity']:.1f}/100"],
                        ["Combined Severity", f"{frame_data['combined_severity']}/100"],
                    ]
                    
                    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0E7FF")),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ]))
                    
                    story.append(metrics_table)
                    story.append(Spacer(1, 0.15*inch))
                    
                    # Frame Recommendations
                    if frame_data['combined_severity'] > 40:
                        story.append(Paragraph("Recommended Actions:", styles['Heading4']))
                        
                        frame_recs = []
                        if frame_data['cracks_detected'] > 0:
                            crack_sev = frame_data['crack_severity']
                            if crack_sev > 70:
                                frame_recs.append("• URGENT: Cracks detected with high severity. Structural assessment required immediately.")
                            elif crack_sev > 40:
                                frame_recs.append("• Monitor cracks closely. Schedule professional repair assessment.")
                            else:
                                frame_recs.append("• Minor cracks present. Seal to prevent water infiltration.")
                        
                        if frame_data['puddle_coverage_pct'] > 0:
                            water_sev = frame_data['puddle_severity']
                            if water_sev > 70:
                                frame_recs.append("• CRITICAL: Active water intrusion detected. Locate and repair water source immediately.")
                            elif water_sev > 40:
                                frame_recs.append("• Water damage present. Identify moisture sources and implement remediation.")
                            else:
                                frame_recs.append("• Minor water staining. Monitor for progression and consider preventive sealing.")
                        
                        for rec in frame_recs:
                            story.append(Paragraph(rec, styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
                    
                    # Frame image
                    if frame_data['overlay_image'] is not None:
                        frame_pil = Image.fromarray(frame_data['overlay_image'])
                        frame_buffer = io.BytesIO()
                        frame_pil.save(frame_buffer, format='PNG')
                        frame_buffer.seek(0)
                        
                        story.append(RLImage(frame_buffer, width=5*inch, height=3.75*inch))
                    
                    if frame_data != frame_results[-1]:
                        story.append(PageBreak())
                
                # Build PDF
                doc.build(story)
                pdf_buffer.seek(0)
                
                return pdf_buffer.getvalue()
            
            except Exception as e:
                st.error(f"❌ PDF generation error: {str(e)}")
                return None
        
        # Single download button that generates and downloads PDF
        st.download_button(
            label="📥 Generate & Download PDF Report",
            data=generate_pdf(),
            file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **📊 Features**
    - 🤖 AI-Powered Detection
    - 📸 Image & Video Analysis
    - 🔴 Live Stream Support
    - 📄 PDF Reports
    """)

with footer_col2:
    st.markdown("""
    **🛠️ Technology**
    - Roboflow Models
    - Computer Vision (OpenCV)
    - Python & Streamlit
    - Local Processing
    """)

with footer_col3:
    st.markdown("""
    **📞 About**
    **Aerius** v1.0.0
    
    Infrastructure inspection
    platform powered by AI
    
    Built with ❤️ for
    business users
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF;'>"
    "© 2025 Aerius. All rights reserved. "
    "| "
    "<a href='#' style='color: #60A5FA; text-decoration: none;'>Documentation</a> "
    "| "
    "<a href='#' style='color: #60A5FA; text-decoration: none;'>Support</a>"
    "</div>",
    unsafe_allow_html=True
)

