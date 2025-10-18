"""
Local puddle detection module.

Purpose: Detect standing water/puddles in video frames using local computer vision.
Operates in offline mode (no API calls).
Used as fallback when Roboflow puddle model is not configured.
"""

import cv2
import numpy as np
from typing import Dict

# Default configuration for local puddle detection
DEFAULT_CFG = {
    "use_heuristic": True,
}


def detect_puddles(frame_bgr: np.ndarray, cfg: Dict = None) -> Dict:
    """
    Detect puddles in a BGR frame using edge detection and dark smooth regions.
    Puddles are characterized by:
    - Low edges/texture (smooth water surface)
    - Darker than surrounding areas
    - Distinct boundaries
    
    This is a fallback local CV method when Roboflow puddle model is not available.
    
    Args:
        frame_bgr: BGR image ndarray (from cv2 or cv2.cvtColor)
        cfg: Config dict
    
    Returns:
        Dict with keys:
        - "mask": binary mask (bool ndarray, True where puddles detected)
        - "area_px": total puddle area in pixels
        - "coverage_pct": puddle coverage as percentage of frame
        - "confidence": confidence score (0..1), based on area prominence
    """
    
    if cfg is None:
        cfg = DEFAULT_CFG
    
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "mask": np.zeros((1, 1), dtype=bool),
            "area_px": 0,
            "coverage_pct": 0.0,
            "confidence": 0.0,
        }
    
    height, width = frame_bgr.shape[:2]
    total_area = height * width
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # ================================================================
    # Detect smooth, dark regions (puddle characteristics)
    # ================================================================
    
    # Step 1: Find edges using Canny (puddles have distinct boundaries)
    edges = cv2.Canny(gray, 50, 150)
    
    # Step 2: Dilate edges slightly to mark boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Step 3: Create mask of LOW edge density (smooth regions)
    # Use morphological opening on inverted edges to find smooth areas
    smooth_mask = cv2.morphologyEx(255 - edges_dilated, cv2.MORPH_OPEN, kernel)
    smooth_mask = smooth_mask > 200
    
    # Step 4: Dark regions (puddles are typically darker)
    dark_mask = gray < 120
    
    # Step 5: Combine: smooth AND dark = likely puddle
    puddle_candidate = smooth_mask & dark_mask
    
    # Step 6: Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    puddle_mask = cv2.morphologyEx(puddle_candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    puddle_mask = cv2.morphologyEx(puddle_mask, cv2.MORPH_OPEN, kernel)
    puddle_mask = puddle_mask > 0
    
    # Step 7: Remove very small noise (min area threshold)
    min_area = 100
    puddle_mask = _remove_small_components(puddle_mask, min_area)
    
    # Calculate metrics
    puddle_area = np.count_nonzero(puddle_mask)
    coverage_pct = (puddle_area / total_area * 100) if total_area > 0 else 0.0
    
    # Confidence: scale with coverage, but be conservative
    confidence = min(coverage_pct / 20.0, 1.0)  # Max confidence at 20% coverage
    
    return {
        "mask": puddle_mask,
        "area_px": int(puddle_area),
        "coverage_pct": float(coverage_pct),
        "confidence": float(confidence),
    }


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove small connected components from binary mask."""
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # Count pixels in each component
    result = np.zeros_like(mask)
    for label in range(1, num_labels):
        component = (labels == label)
        if np.count_nonzero(component) >= min_size:
            result |= component
    
    return result
