from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = (256, 256)
MAX_DURATION_SECONDS = 30


def _simulate_compression(frame, compression_quality):
    # JPEG encode-decode acts as a simple stand-in for strong video compression artifacts.
    encode_success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(compression_quality)],
    )
    if not encode_success:
        return frame.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return frame.copy()
    return decoded


def load_and_compress_video(video_path, compression_quality=10):
    """Read a video, resize frames, and simulate strong compression artifacts."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(fps * MAX_DURATION_SECONDS)

    original_frames = []
    compressed_frames = []
    frame_count = 0

    while frame_count < max_frames:
        success, frame = capture.read()
        if not success:
            break
        # Every frame is normalized to a fixed size so processing and metrics stay consistent.
        resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        compressed = _simulate_compression(resized, compression_quality)
        original_frames.append(resized)
        compressed_frames.append(compressed)
        frame_count += 1

    capture.release()

    if not original_frames:
        raise ValueError("The uploaded video did not contain any readable frames.")

    metadata = {
        "fps": float(fps),
        "frame_count": len(original_frames),
        "size": TARGET_SIZE,
        "max_duration_seconds": MAX_DURATION_SECONDS,
    }
    return original_frames, compressed_frames, metadata
