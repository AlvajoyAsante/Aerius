# Getting Started with Aerius

**⏱️ Setup time: ~5 minutes**

## What is Aerius?

Aerius is a **Streamlit web app** that detects cracks and puddles in buildings/structures using:
- **Roboflow API** for AI-powered crack detection
- **Local Computer Vision** for puddle detection
- **Real-time visualization** with polygon overlays on images/videos

## Step 1: Get a Roboflow API Key (2 min)

1. Go to [Roboflow](https://roboflow.com) and sign up (free account)
2. Create a new project or use an existing one
3. Click **"API"** in the left sidebar
4. Copy your **API Key** (looks like: `C7mQNur0kpn2kosoidWQ`)
5. Keep this safe—you'll need it in Step 3

> **Note:** You also need a **Model ID** from Roboflow. It looks like `crack-bphdr/1`. Ask Bryan for the exact model ID.

## Step 2: Install & Setup (2 min)

### Clone the repo (if you haven't already)
```bash
git clone https://github.com/AlvajoyAsante/Aerius.git
cd Aerius
```

### Create a virtual environment
**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```bash
python -m venv .venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Configure Your API Key (1 min)

### Option A: Using `.streamlit/secrets.toml` (Recommended)
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your Roboflow API key:
```toml
ROBOFLOW_API_KEY="your_api_key_here"
```

### Option B: Using environment variable
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

## Step 4: Run the App! (1 min)

Make sure your virtual environment is activated, then:

```bash
streamlit run streamlit_app.py
```

The app will open at **http://localhost:8501** 🎉

## Using the App

### 📷 Image Tab
1. Upload a crack photo (JPG, PNG)
2. Click **"🚀 Run Crack Inference"**
3. See predictions with green polygon overlays

### 🎬 Video Tab
1. Upload a video file (MP4)
2. View metadata (FPS, frame count, duration)
3. See first frame preview

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ROBOFLOW_API_KEY not set` | Check `.streamlit/secrets.toml` or set environment variable (Step 3) |
| `InferenceHTTPClient error` | Verify your API key is correct and has quotes around it |
| `App won't start` | Ensure venv is activated: `source .venv/bin/activate` (macOS/Linux) |
| `Model ID not found` | Ask Bryan—it needs to be set in `core/cracks_api.py` |

## Quick Test (Optional)

Test the Roboflow API without the web app:

```bash
# Set API key
export ROBOFLOW_API_KEY="your_api_key_here"

# Run the CLI test
python scripts/test_roboflow_infer.py samples/test_crack.jpg
```

If you see prediction data printed, the API is working! ✅

## Next Steps

Once the app is running:
1. Test with the sample crack image
2. Try uploading your own images/videos
3. Check the **Video Scaffold** tab to understand video metadata extraction
4. Explore the code in `core/` to understand the pipeline

## Questions?

Ask Bryan or check the main [README.md](README.md) for more details on project architecture and features.

Happy detecting! 🔍

