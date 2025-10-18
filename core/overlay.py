"""
Visualization overlay module.

Purpose: Overlay detected puddles and crack masks on frames with color coding,
labels, and score annotations.
"""

import cv2
import numpy as np
from typing import Tuple


def draw_puddles_overlay(frame_bgr: np.ndarray, puddle_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Blend a puddle mask (blue overlay) onto a BGR frame.
    
    Args:
        frame_bgr: BGR image ndarray
        puddle_mask: Binary mask (bool or uint8) where True/255 = puddle
        alpha: Blend factor (0..1); higher = more opaque puddle overlay
    
    Returns:
        BGR image ndarray with blue puddle overlay blended in
    """
    
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    
    # Create a blue overlay (BGR: blue channel high)
    overlay = frame_bgr.copy()
    
    # Puddle color in BGR: bright blue
    puddle_color_bgr = (255, 200, 100)  # (B, G, R) - aqua/cyan
    
    # Apply blue tint where puddle mask is True
    puddle_bool = puddle_mask > 0 if puddle_mask.dtype != bool else puddle_mask
    
    for c in range(3):  # For each color channel
        overlay[puddle_bool, c] = (
            overlay[puddle_bool, c] * (1 - alpha) + puddle_color_bgr[c] * alpha
        ).astype(np.uint8)
    
    return overlay
