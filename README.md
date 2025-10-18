# Aerius - Crack & Puddle Detection Tool

Aerius is a Streamlit application that analyzes phone or drone videos to detect and report cracks and puddles in buildings, walls, and other structures.

## Project Overview

**Aerius** leverages a hybrid approach to structural defect detection:
- **Roboflow API (Cracks)**: AI-powered crack detection using instance segmentation
- **Roboflow API (Puddles)**: AI-powered puddle detection (if model is configured), with local CV fallback
- **Temporal Merge**: Combines results across frames for consistent detection
- **Intelligent Scoring**: Generates a 0–100 severity score based on area, depth, and persistence (60% puddle weight + 40% crack weight)
- **Visual Overlays**: Renders annotated frames with color-coded defects (green for cracks, blue for puddles)
- **PDF Report**: Exports a professional one-page PDF summary with key metrics

## Features

- **Dual-API Detection**: Cracks + Puddles via Roboflow APIs
- **Graceful Fallback**: If puddle API not configured, uses local OpenCV heuristics
- **API Result Caching**: Caches Roboflow responses by video hash (no redundant calls)
- **Batch Processing**: Handles up to 100 frames per batch
- **Cross-Platform**: Supports macOS, Linux, and Windows

## Quickstart

### Prerequisites
- Python 3.9+

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd aerius
   ```

2. Create and activate a virtual environment:

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy and configure secrets:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Edit `.streamlit/secrets.toml` and add your Roboflow API key:
   ```toml
   ROBOFLOW_API_KEY="your_key_here"
   ```

### Configuration

**Crack Model** (required):
- Update `ROBOFLOW_MODEL_ID` in `core/cracks_api.py` with your crack detection model ID from Roboflow

**Puddle Model** (optional):
- Update `PUDDLE_ROBOFLOW_MODEL_ID` in `core/puddle_api.py` with your puddle detection model ID
- If not configured, the app will fall back to local OpenCV-based puddle detection

### Run the App

1. Make sure your venv is activated and dependencies are installed (see above).

2. Ensure `.streamlit/secrets.toml` contains your `ROBOFLOW_API_KEY`.

3. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
   The app will open at `http://localhost:8501` by default.

4. **Using the app:**
   - **Image Test (Cracks + Puddles)**: Upload an image and click "Run Crack Inference" to see cracks (green overlay) and puddles (blue overlay) detected in one combined view with combined severity score.
   - **Video Scaffold**: Upload a short video to see metadata (FPS, frame count, duration) and the first frame preview.

   *(Optional for remote access):*
   ```bash
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```
   Then open `http://<your-machine-ip>:8501` from another device on the same network.

### Roboflow Sanity Test

Before running the full app, test the Roboflow API integration directly:

1. Set your API key:
   ```bash
   export ROBOFLOW_API_KEY=your_actual_key_here
   ```

2. **Important**: Ensure the model IDs are set:
   - Crack model: Replace `ROBOFLOW_MODEL_ID = "crack-xxxxx/1"` in `core/cracks_api.py` with your exact model ID
   - Puddle model (optional): Replace `PUDDLE_ROBOFLOW_MODEL_ID = "puddle-xxxxx/1"` in `core/puddle_api.py` with your model ID (find it at Roboflow → Hosted API section)

3. Run the CLI test script:
   ```bash
   python scripts/test_roboflow_infer.py samples/any_crack_image.jpg
   ```
   This will call the Roboflow API and print the response keys and prediction count.

````

## Notes

- **Puddles-Only Mode**: Run offline by disabling the Roboflow API (local CV only)
- **API Caching**: Results are cached by video hash in `samples/fixtures/` to avoid redundant API calls
- **Batch Limits**: Each analysis processes a maximum of 100 frames per batch for performance
- **Environment Variables**: Load API keys from `.env` via `python-dotenv` or Streamlit secrets

## Project Structure

```
aerius/
├── streamlit_app.py          # Main Streamlit app entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── .env.example             # Example environment configuration
├── .streamlit/
│   ├── config.toml          # Streamlit UI configuration
│   └── secrets.toml.example # Example API secrets
├── core/                    # Core analysis modules
│   ├── __init__.py
│   ├── ingest.py           # Video frame extraction
│   ├── puddles.py          # Local puddle detection
│   ├── cracks_api.py       # Roboflow API integration
│   ├── skeletonize.py      # Crack thinning
│   ├── tracking.py         # Temporal defect tracking
│   ├── scoring.py          # Defect severity scoring
│   ├── overlay.py          # Frame annotation
│   ├── report.py           # PDF report generation
│   ├── cache.py            # API result caching
│   └── utils.py            # Utility functions
├── assets/                 # Application assets
│   ├── logo.png
│   └── demo_banner.png
├── samples/                # Example data and fixtures
│   ├── puddle_demo.mp4
│   ├── crack_demo.mp4
│   └── fixtures/
│       └── crack_api_cached.json
└── tests/                  # Test notebooks
    ├── test_puddles.ipynb
    └── test_cracks_api.ipynb
```

## Development

TODO: Add development guidelines, testing instructions, and contribution workflow

## License

See LICENSE file for details.