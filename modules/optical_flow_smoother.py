import cv2
import numpy as np


def _validate_frame(frame):
    assert frame.dtype == np.uint8, "Frame must be uint8"
    assert len(frame.shape) == 3 and frame.shape[2] == 3, "Frame must be BGR"


def _compute_flow(curr_frame, prev_frame):
    _validate_frame(curr_frame)
    _validate_frame(prev_frame)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=10,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    return np.clip(flow, -10, 10)


def compute_flow_visualization(flow, frame_shape):
    h, w = frame_shape[:2]
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 0] = angle * 180 / np.pi / 2
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def warp_frame_with_flow(previous_frame, flow):
    _validate_frame(previous_frame)
    h, w = flow.shape[:2]
    map_x = np.float32(np.tile(np.arange(w), (h, 1)))
    map_y = np.float32(np.tile(np.arange(h), (w, 1)).T)
    map_x_warped = map_x + flow[:, :, 0]
    map_y_warped = map_y + flow[:, :, 1]
    warped = cv2.remap(
        previous_frame,
        map_x_warped,
        map_y_warped,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def smooth_frame(curr_frame, prev_frame, alpha=0.08):
    _validate_frame(curr_frame)
    _validate_frame(prev_frame)

    diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
    if diff > 30:
        return curr_frame.copy()

    flow = _compute_flow(curr_frame, prev_frame)
    warped_prev = warp_frame_with_flow(prev_frame, flow)
    smoothed = cv2.addWeighted(curr_frame, 1 - alpha, warped_prev, alpha, 0)
    blurred = cv2.GaussianBlur(smoothed, (3, 3), 0.5)
    smoothed = cv2.addWeighted(smoothed, 1.2, blurred, -0.2, 0)
    smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    return smoothed


def smooth_frames_with_optical_flow(enhanced_frames, alpha=0.08):
    if not enhanced_frames:
        return [], []

    alpha = min(alpha, 0.08)
    smoothed_frames = [enhanced_frames[0].copy()]
    flow_visualizations = [np.zeros_like(enhanced_frames[0])]

    for idx in range(1, len(enhanced_frames)):
        previous_smoothed = smoothed_frames[-1]
        current_frame = enhanced_frames[idx]
        _validate_frame(current_frame)

        diff = np.mean(np.abs(current_frame.astype(float) - previous_smoothed.astype(float)))
        if diff > 30:
            smoothed_frame = current_frame.copy()
            flow_visualization = np.zeros_like(current_frame)
        else:
            try:
                flow = _compute_flow(current_frame, previous_smoothed)
                smoothed_frame = smooth_frame(current_frame, previous_smoothed, alpha=alpha)
                flow_visualization = compute_flow_visualization(flow, current_frame.shape)
            except cv2.error:
                flow_visualization = np.zeros_like(current_frame)
                smoothed_frame = current_frame.copy()

        smoothed_frames.append(smoothed_frame)
        flow_visualizations.append(flow_visualization)

    return smoothed_frames, flow_visualizations
