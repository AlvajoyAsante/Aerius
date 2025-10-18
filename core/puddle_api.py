"""
Roboflow Puddle Detection API module.

Purpose: Detect puddles using a trained Roboflow instance-segmentation model.
"""

from typing import Dict
import os
import tempfile
import numpy as np
import cv2
from PIL import Image as PILImage
from inference_sdk import InferenceHTTPClient

# --- Roboflow Puddle model config (constants only) ---
PUDDLE_ROBOFLOW_MODEL_ID = "puddle-new/1"  # TODO: replace with actual model_id (e.g., "puddle-uznq9/2")
PUDDLE_CONFIDENCE = 0.40                      # start a bit higher to reduce glare/shadow FPs
PUDDLE_MIN_AREA_FRAC = 0.002                  # ignore masks < 0.2% of image
TARGET_RESIZE_WIDTH = 640
# -------------------------------------------------------


def infer_puddles_mask_from_rgb(image_rgb: np.ndarray, api_key: str = None) -> Dict:
    """
    Detect puddles using Roboflow puddle model (instance segmentation).
    
    Args:
        image_rgb: RGB image ndarray from PIL
        api_key: Roboflow API key (if None, attempts to get from env)
    
    Returns:
        Dict with keys:
        - "mask": binary mask (bool ndarray, same HxW as input image)
        - "coverage_pct": puddle coverage as percentage (0..100)
        - "predictions": raw API predictions (for debugging)
        - "error": error message if inference failed (None on success)
    
    If ROBOFLOW_API_KEY or PUDDLE_ROBOFLOW_MODEL_ID is missing/placeholder,
    returns an empty mask result.
    """
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY", "")
    
    if not api_key or not api_key.strip():
        return {
            "mask": np.zeros(image_rgb.shape[:2], dtype=bool),
            "coverage_pct": 0.0,
            "predictions": [],
            "error": "API key not set"
        }
    
    # Check if model ID is placeholder
    if "xxxxx" in PUDDLE_ROBOFLOW_MODEL_ID:
        return {
            "mask": np.zeros(image_rgb.shape[:2], dtype=bool),
            "coverage_pct": 0.0,
            "predictions": [],
            "error": "Puddle model ID not configured"
        }
    
    original_height, original_width = image_rgb.shape[:2]
    
    try:
        # Resize image to target width, keeping aspect ratio
        scale = TARGET_RESIZE_WIDTH / original_width
        resized_height = int(original_height * scale)
        resized_image_rgb = cv2.resize(image_rgb, (TARGET_RESIZE_WIDTH, resized_height), interpolation=cv2.INTER_LINEAR)
        
        # Save to temp file (Roboflow API requires file path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Convert RGB to BGR for cv2.imwrite
            resized_image_bgr = cv2.cvtColor(resized_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp.name, resized_image_bgr)
            temp_path = tmp.name
        
        try:
            # Call Roboflow API
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key
            )
            result = client.infer(
                temp_path,
                model_id=PUDDLE_ROBOFLOW_MODEL_ID
            )
            
            predictions = result.get("predictions", [])
            
            # Rasterize polygons to mask at resized size
            resized_mask = np.zeros((resized_height, TARGET_RESIZE_WIDTH), dtype=np.uint8)
            
            if predictions:
                for pred in predictions:
                    # Get segmentation points if available
                    points = pred.get("points", [])
                    if points:
                        # Convert points to numpy array for polylines
                        pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)
                        
                        # Calculate area fraction to filter tiny masks
                        area = cv2.contourArea(pts)
                        area_frac = area / (resized_height * TARGET_RESIZE_WIDTH)
                        
                        if area_frac >= PUDDLE_MIN_AREA_FRAC:
                            # Draw filled polygon on mask
                            cv2.fillPoly(resized_mask, [pts], 255)
            
            # Upsample mask back to original image size
            if resized_mask.max() > 0:
                # Use nearest neighbor to preserve binary mask
                mask_original = cv2.resize(resized_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                mask_binary = mask_original > 127
            else:
                mask_binary = np.zeros((original_height, original_width), dtype=bool)
            
            # Calculate coverage
            coverage_pct = (np.count_nonzero(mask_binary) / (original_height * original_width) * 100)
            
            return {
                "mask": mask_binary,
                "coverage_pct": float(coverage_pct),
                "predictions": predictions,
                "error": None
            }
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return {
            "mask": np.zeros((original_height, original_width), dtype=bool),
            "coverage_pct": 0.0,
            "predictions": [],
            "error": str(e)
        }
