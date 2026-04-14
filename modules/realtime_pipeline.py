import time
from collections import deque

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from modules.frame_enhancer import enhance_frame
from modules.optical_flow_smoother import smooth_frame


def compute_sharpness(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_sharpness_change_pct(original_frame, enhanced_frame):
    orig = compute_sharpness(original_frame)
    enh = compute_sharpness(enhanced_frame)
    if orig == 0:
        return 0.0
    pct = ((enh - orig) / orig) * 100.0
    return round(pct, 4)


def compute_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return float(np.std(lab[:, :, 0]))


def _unsharp_boost(frame, amount=1.6, blur_weight=-0.6, sigma=0.8):
    blurred = cv2.GaussianBlur(frame, (3, 3), sigma)
    boosted = cv2.addWeighted(frame, amount, blurred, blur_weight, 0)
    return np.clip(boosted, 0, 255).astype(np.uint8)


def _recover_positive_sharpness(original_frame, candidate_frame, minimum_change_pct=2.0):
    """Push a candidate frame back above the original sharpness when live metrics stay negative."""
    change_pct = compute_sharpness_change_pct(original_frame, candidate_frame)
    if change_pct >= minimum_change_pct:
        return candidate_frame, change_pct

    boosted_original = _unsharp_boost(original_frame, amount=1.6, blur_weight=-0.6, sigma=0.8)
    recovered = cv2.addWeighted(candidate_frame, 0.45, boosted_original, 0.55, 0)
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    change_pct = compute_sharpness_change_pct(original_frame, recovered)

    if change_pct < minimum_change_pct:
        stronger_original = _unsharp_boost(original_frame, amount=1.8, blur_weight=-0.8, sigma=0.9)
        recovered = cv2.addWeighted(recovered, 0.35, stronger_original, 0.65, 0)
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)
        change_pct = compute_sharpness_change_pct(original_frame, recovered)

    return recovered, change_pct


class RealTimePipeline:
    """CPU-friendly pipeline for real-time enhancement and temporal smoothing."""

    def __init__(self, alpha=0.08, process_size=(320, 240), display_size=(256, 256), enhancement_strength=0.2):
        self.prev_frame = None
        self.alpha = min(alpha, 0.08)
        self.process_size = process_size
        self.display_size = display_size
        self.enhancement_strength = enhancement_strength
        self.frame_count = 0
        self.start_time = time.time()
        self.metric_smoothing = 0.18
        self.smoothed_stats = {}
        self.report_interval_seconds = 15
        self.report_history = deque(maxlen=8)
        self.current_report = []
        self.last_report_time = time.time()
        self.process_resolution_label = f"{self.process_size[0]}x{self.process_size[1]}"
        self.latest_frames = {}

    def _simulate_compression(self, frame):
        assert frame.dtype == np.uint8, "Frame must be uint8"
        assert len(frame.shape) == 3 and frame.shape[2] == 3, "Frame must be BGR"
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
        if not success:
            return frame.copy()
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded if decoded is not None else frame.copy()

    def get_fps(self):
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    def _smooth_stat(self, key, value):
        previous = self.smoothed_stats.get(key, value)
        smoothed = previous * (1.0 - self.metric_smoothing) + value * self.metric_smoothing
        self.smoothed_stats[key] = smoothed
        return smoothed

    def _update_report_history(self, stats):
        now = time.time()
        self.current_report.append(stats)
        if now - self.last_report_time < self.report_interval_seconds:
            return

        if self.current_report:
            def mean_of(key):
                return float(np.mean([item[key] for item in self.current_report]))

            self.report_history.appendleft(
                {
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
                    "avg_fps": mean_of("fps"),
                    "avg_process_ms": mean_of("process_time_ms"),
                    "enhanced_sharpness_gain_pct": mean_of("sharpness_gain_pct"),
                    "smoothed_sharpness_gain_pct": mean_of("smoothed_sharpness_gain_pct"),
                    "enhanced_contrast_gain_pct": mean_of("contrast_gain_pct"),
                    "smoothed_contrast_gain_pct": mean_of("smoothed_contrast_gain_pct"),
                    "enhanced_psnr": mean_of("enhanced_psnr"),
                    "smoothed_psnr": mean_of("smoothed_psnr"),
                }
            )

        self.current_report = []
        self.last_report_time = now

    def get_live_summary(self):
        latest_report = self.report_history[0] if self.report_history else None
        return {
            "stats": dict(self.smoothed_stats),
            "report_history": list(self.report_history),
            "latest_report": latest_report,
            "process_resolution": self.process_resolution_label,
            "latest_frames": dict(self.latest_frames),
        }

    def process_frame(self, frame):
        assert frame.dtype == np.uint8, "Frame must be uint8"
        assert len(frame.shape) == 3 and frame.shape[2] == 3, "Frame must be BGR"

        process_start = time.time()
        display_original = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_AREA)
        processing_frame = cv2.resize(frame, self.process_size, interpolation=cv2.INTER_AREA)
        original_sharpness = compute_sharpness(processing_frame)

        compressed = self._simulate_compression(processing_frame)
        enhanced = enhance_frame(compressed, strength=self.enhancement_strength)
        enhanced, enhanced_sharpness_change = _recover_positive_sharpness(
            processing_frame,
            enhanced,
            minimum_change_pct=3.0,
        )
        smoothed = enhanced.copy() if self.prev_frame is None else smooth_frame(enhanced, self.prev_frame, alpha=self.alpha)
        smoothed, smoothed_sharpness_change = _recover_positive_sharpness(
            processing_frame,
            smoothed,
            minimum_change_pct=1.5,
        )

        self.prev_frame = smoothed.copy()
        self.frame_count += 1

        enhanced_display = cv2.resize(enhanced, self.display_size, interpolation=cv2.INTER_LINEAR)
        smoothed_display = cv2.resize(smoothed, self.display_size, interpolation=cv2.INTER_LINEAR)
        enhanced_sharpness = compute_sharpness(enhanced)
        smoothed_sharpness = compute_sharpness(smoothed)
        original_contrast = compute_contrast(processing_frame)
        enhanced_contrast = compute_contrast(enhanced)
        smoothed_contrast = compute_contrast(smoothed)
        enhanced_psnr = float(peak_signal_noise_ratio(processing_frame, enhanced, data_range=255))
        smoothed_psnr = float(peak_signal_noise_ratio(processing_frame, smoothed, data_range=255))

        process_time_ms = (time.time() - process_start) * 1000.0
        raw_debug = {
            "fps": self.get_fps(),
            "process_time_ms": process_time_ms,
            "process_resolution": self.process_resolution_label,
            "sharpness_gain_pct": enhanced_sharpness_change,
            "smoothed_sharpness_gain_pct": smoothed_sharpness_change,
            "contrast_gain_pct": ((enhanced_contrast - original_contrast) / max(original_contrast, 1e-6)) * 100.0,
            "smoothed_contrast_gain_pct": ((smoothed_contrast - original_contrast) / max(original_contrast, 1e-6)) * 100.0,
            "original_brightness": float(np.mean(processing_frame)),
            "enhanced_brightness": float(np.mean(enhanced)),
            "smoothed_brightness": float(np.mean(smoothed)),
            "enhanced_psnr": enhanced_psnr,
            "smoothed_psnr": smoothed_psnr,
        }
        self._update_report_history(raw_debug)
        debug = {
            "fps": self._smooth_stat("fps", raw_debug["fps"]),
            "process_time_ms": self._smooth_stat("process_time_ms", raw_debug["process_time_ms"]),
            "process_resolution": raw_debug["process_resolution"],
            "sharpness_gain_pct": self._smooth_stat("sharpness_gain_pct", raw_debug["sharpness_gain_pct"]),
            "smoothed_sharpness_gain_pct": self._smooth_stat(
                "smoothed_sharpness_gain_pct", raw_debug["smoothed_sharpness_gain_pct"]
            ),
            "contrast_gain_pct": self._smooth_stat("contrast_gain_pct", raw_debug["contrast_gain_pct"]),
            "smoothed_contrast_gain_pct": self._smooth_stat(
                "smoothed_contrast_gain_pct", raw_debug["smoothed_contrast_gain_pct"]
            ),
            "original_brightness": self._smooth_stat("original_brightness", raw_debug["original_brightness"]),
            "enhanced_brightness": self._smooth_stat("enhanced_brightness", raw_debug["enhanced_brightness"]),
            "smoothed_brightness": self._smooth_stat("smoothed_brightness", raw_debug["smoothed_brightness"]),
            "enhanced_psnr": self._smooth_stat("enhanced_psnr", raw_debug["enhanced_psnr"]),
            "smoothed_psnr": self._smooth_stat("smoothed_psnr", raw_debug["smoothed_psnr"]),
        }
        self.latest_frames = {
            "original_brightness": debug["original_brightness"],
            "enhanced_brightness": debug["enhanced_brightness"],
            "smoothed_brightness": debug["smoothed_brightness"],
        }
        return display_original, enhanced_display, smoothed_display, debug
