"""
Aerius core module for structural crack and puddle detection.

This module provides tools for analyzing phone/drone videos of buildings/structures
to detect puddles and cracks using a hybrid approach:
- Local CV for puddle detection
- Roboflow API for crack masks
- Temporal merge and scoring
"""
