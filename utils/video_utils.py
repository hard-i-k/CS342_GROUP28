import io
import tempfile
from pathlib import Path

import cv2
from PIL import Image


def create_gif_bytes(frames, fps=8):
    """Convert BGR frames into a compact animated GIF for Streamlit preview."""
    if not frames:
        return None

    duration_ms = int(1000 / max(fps, 1))
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames[:60]]
    buffer = io.BytesIO()
    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    buffer.seek(0)
    return buffer.getvalue()


def create_mp4_bytes(frames, fps=15):
    """Write a temporary MP4 preview and return its bytes for Streamlit video playback."""
    if not frames:
        return None

    height, width = frames[0].shape[:2]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.close()
    temp_path = Path(temp_file.name)

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

    data = temp_path.read_bytes()
    temp_path.unlink(missing_ok=True)
    return data
