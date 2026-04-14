import numpy as np
import cv2


def _validate_frame(frame):
    assert frame.dtype == np.uint8, "Frame must be uint8"
    assert len(frame.shape) == 3 and frame.shape[2] == 3, "Frame must be BGR"


def enhance_frame(frame, strength=0.3):
    _validate_frame(frame)

    blurred = cv2.GaussianBlur(frame, (3, 3), 0.8)
    sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    enhanced = cv2.addWeighted(frame, 1 - strength, sharpened, strength, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def enhance_frames(frames, source_frames=None, enhancement_strength=1.0):
    enhanced_frames = []
    for idx, frame in enumerate(frames):
        _validate_frame(frame)
        _ = source_frames[idx] if source_frames is not None else frame
        enhanced_frames.append(enhance_frame(frame, strength=enhancement_strength))
    return enhanced_frames
