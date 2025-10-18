"""
Defect scoring module.

Purpose: Score puddles and cracks on a 0–100 scale based on:
- Area coverage
- Depth/severity (inferred from color/texture)
- Temporal persistence across frames
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_severity(predictions: List[Dict], image_width: int, image_height: int) -> Tuple[int, float, str]:
    """
    Calculate severity score (0-100) for cracks based on Roboflow predictions.
    
    Args:
        predictions: List of Roboflow prediction dicts with 'points', 'confidence', etc.
        image_width: Width of the image (pixels)
        image_height: Height of the image (pixels)
    
    Returns:
        Tuple of (severity_score: 0-100, coverage_percent: 0-100, recommendation: str)
    """
    
    if not predictions:
        return 0, 0.0, "No defects detected. Structure appears sound."
    
    total_image_area = image_width * image_height
    
    # Calculate total defect area from polygon predictions
    total_defect_area = 0
    confidences = []
    
    for pred in predictions:
        confidences.append(pred.get("confidence", 0))
        
        if "points" in pred:
            points = pred["points"]
            if isinstance(points, list) and len(points) >= 3:
                # Calculate polygon area using the shoelace formula
                coords = [(p.get("x", 0), p.get("y", 0)) for p in points]
                area = _polygon_area(coords)
                total_defect_area += max(area, 0)  # Ensure non-negative
    
    # Calculate coverage percentage
    coverage_percent = (total_defect_area / total_image_area) * 100 if total_image_area > 0 else 0
    coverage_percent = min(coverage_percent, 100.0)  # Cap at 100%
    
    # Calculate average confidence (model's confidence in detections)
    avg_confidence = np.mean(confidences) * 100 if confidences else 0
    
    # Defect count factor (more defects = higher severity)
    defect_count = len(predictions)
    defect_factor = min(defect_count * 10, 30)  # Max 30 points from defect count
    
    # Area factor (0-40 points based on coverage)
    area_factor = (coverage_percent / 100) * 40
    
    # Confidence factor (0-30 points - higher confidence = higher severity)
    confidence_factor = (avg_confidence / 100) * 30
    
    # Combine into severity score (0-100)
    severity_score = int(defect_factor + area_factor + confidence_factor)
    severity_score = max(0, min(severity_score, 100))  # Clamp to 0-100
    
    # Generate recommendation based on severity
    if severity_score < 20:
        recommendation = "Monitor regularly. No immediate action needed."
    elif severity_score < 40:
        recommendation = "Schedule inspection within 6 months."
    elif severity_score < 60:
        recommendation = "Plan repairs within 3 months."
    elif severity_score < 80:
        recommendation = "Urgent repair needed within 1 month."
    else:
        recommendation = "Critical. Immediate professional inspection required."
    
    return severity_score, coverage_percent, recommendation


def _polygon_area(coords: List[Tuple[float, float]]) -> float:
    """Calculate polygon area using shoelace formula."""
    if len(coords) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    
    return abs(area) / 2.0
