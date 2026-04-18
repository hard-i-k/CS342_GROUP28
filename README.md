# Temporal Consistency and Real-Time Enhancement Pipeline

## Overview

Phase 1 -> https://drive.google.com/file/d/1SDA5RNtWHG1N4UrdVXZaT_OqOwC7r2kJ/view
Phase 2 -> https://drive.google.com/file/d/1VL-B72b7G66invMCkU1C9h6pnyC6ERPt/view

This project is a Streamlit-based computer vision application for analyzing and improving compressed video quality. It combines frame enhancement, optical flow based temporal smoothing, and quality metric reporting to compare compressed, enhanced, and temporally stabilized video outputs.

The application supports two workflows:

1. Video analysis mode for uploaded MP4 files
2. Real-time mode for webcam-based enhancement and monitoring

## Key Features

- Upload and analyze MP4 videos up to 30 seconds in length
- Simulate visible compression artifacts using JPEG-based degradation
- Enhance individual frames to improve visual sharpness
- Apply optical flow based temporal smoothing to reduce flicker
- Compare compressed, enhanced, and smoothed outputs side by side
- Visualize motion patterns through optical flow previews
- Measure temporal consistency with PSNR, SSIM, sharpness, contrast, and flicker metrics
- Run a lightweight real-time webcam pipeline with live performance reporting

## Technology Stack

- Python
- Streamlit
- OpenCV
- NumPy
- SciPy
- scikit-image
- Matplotlib
- Pandas
- Pillow
- streamlit-webrtc

## Project Structure

```text
IMAGE_PROJECT/
|-- app.py
|-- requirements.txt
|-- modules/
|   |-- frame_enhancer.py
|   |-- optical_flow_smoother.py
|   |-- realtime_pipeline.py
|   |-- temporal_metrics.py
|   |-- video_processor.py
|-- utils/
|   |-- plot_utils.py
|   |-- video_utils.py
```

## How It Works

### Video Analysis Mode

The uploaded video is read frame by frame, resized to a fixed resolution, and artificially compressed to simulate low-quality transmission. Each frame is then enhanced, after which temporal smoothing is applied using optical flow to align motion between neighboring frames. The app generates preview videos, GIFs, optical flow visualizations, and a metrics dashboard summarizing quality changes across the pipeline.

### Real-Time Mode

The webcam pipeline processes lower-resolution frames to keep runtime lightweight on a standard CPU. It enhances live frames, applies temporal smoothing, and continuously updates quality and performance statistics such as FPS, processing time, sharpness change, contrast change, and PSNR.

## Installation

### 1. Clone or open the project

Make sure you are inside the project folder:

```powershell
cd c:\Users\hadik\Desktop\IMAGE_PROJECT
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app with:

```powershell
streamlit run app.py
```

After launch, open the local Streamlit URL shown in the terminal.

## Usage

### Video Analysis

1. Open the app in your browser
2. Select `Video Analysis Mode`
3. Upload an MP4 video
4. Adjust enhancement strength, temporal blend alpha, and compression severity
5. Click `Run Analysis`
6. Review the comparison tabs for video previews, optical flow, metrics, and summary tables

### Real-Time Processing

1. Select `Real-Time Mode`
2. Allow camera access in the browser
3. Start the webcam stream
4. Monitor live visual output and the quality report panel

If `streamlit-webrtc` is unavailable in your environment, the app falls back to snapshot processing using camera capture.

## Metrics Reported

The application reports several quality indicators, including:

- Inter-frame difference
- Temporal SSIM
- Flickering index
- PSNR
- Sharpness trends
- Contrast trends
- Brightness comparison
- Real-time FPS and processing latency


