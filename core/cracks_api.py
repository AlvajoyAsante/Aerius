"""
Roboflow crack detection API module.

Purpose: Send video frames to Roboflow API for structural crack detection.
Works on buildings, walls, and other structures. Includes result caching by video hash 
to avoid redundant API calls.

TODO: Implement Roboflow API client with error handling
TODO: Implement frame batch submission to Roboflow
TODO: Implement response parsing (segmentation masks)
TODO: Implement cache storage by video hash (see cache.py)
TODO: Handle API rate limits and retries
"""
# --- Roboflow Inference config (constants only; no logic) ---
ROBOFLOW_MODEL_ID = "concrete-crack-ver.1/2" # crack detection model
ROBOFLOW_CONFIDENCE = 0.35           # default confidence; will tune later
TARGET_RESIZE_WIDTH = 640            # frames sent to API
BATCH_SIZE = 15                      # images per request
# --------------------------------------------------------------

