#!/usr/bin/env python3
"""
Sanity test script for Roboflow inference API (≤40 lines).

Usage:
    export ROBOFLOW_API_KEY=your_key
    python scripts/test_roboflow_infer.py path/to/image.jpg
"""

import os
import sys
from inference_sdk import InferenceHTTPClient
from core.cracks_api import ROBOFLOW_MODEL_ID


def main():
    # Check args
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_roboflow_infer.py <image_path>")
        print("Example: export ROBOFLOW_API_KEY=key && python scripts/test_roboflow_infer.py image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Validate file
    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    # Get API key
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not set")
        sys.exit(1)
    
    print(f"Image: {image_path}")
    print(f"Model: {ROBOFLOW_MODEL_ID}")
    print()
    
    # Infer
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        result = client.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
        
        print(f"✓ Response keys: {list(result.keys())}")
        print(f"✓ Predictions: {len(result.get('predictions', []))}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
