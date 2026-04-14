import matplotlib
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_float(frame):
    return frame.astype(np.float32)


def inter_frame_difference_scores(frames):
    scores = []
    for idx in range(1, len(frames)):
        diff = np.abs(_to_float(frames[idx]) - _to_float(frames[idx - 1]))
        scores.append(float(np.mean(diff)))
    return scores


def temporal_ssim_scores(frames):
    scores = []
    for idx in range(1, len(frames)):
        score = structural_similarity(
            frames[idx - 1],
            frames[idx],
            channel_axis=2,
            data_range=255,
        )
        scores.append(float(score))
    return scores


def psnr_scores(reference_frames, compared_frames):
    scores = []
    for reference, compared in zip(reference_frames, compared_frames):
        scores.append(float(peak_signal_noise_ratio(reference, compared, data_range=255)))
    return scores


def sharpness_scores(frames):
    scores = []
    for frame in frames:
        gray = frame.mean(axis=2).astype(np.float32)
        scores.append(float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1])))
    return scores


def contrast_scores(frames):
    scores = []
    for frame in frames:
        scores.append(float(np.std(frame.mean(axis=2))))
    return scores


def _plot_line_chart(x_values, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, values, color in series:
        ax.plot(x_values[: len(values)], values, label=label, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_bar_chart(labels, values, colors, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def compute_temporal_metrics(original_frames, compressed_frames, enhanced_frames, smoothed_frames):
    compressed_diff = inter_frame_difference_scores(compressed_frames)
    enhanced_diff = inter_frame_difference_scores(enhanced_frames)
    smoothed_diff = inter_frame_difference_scores(smoothed_frames)

    compressed_tssim = temporal_ssim_scores(compressed_frames)
    enhanced_tssim = temporal_ssim_scores(enhanced_frames)
    smoothed_tssim = temporal_ssim_scores(smoothed_frames)

    enhanced_psnr = psnr_scores(original_frames, enhanced_frames)
    smoothed_psnr = psnr_scores(original_frames, smoothed_frames)
    original_sharpness = sharpness_scores(original_frames)
    compressed_sharpness = sharpness_scores(compressed_frames)
    enhanced_sharpness = sharpness_scores(enhanced_frames)
    smoothed_sharpness = sharpness_scores(smoothed_frames)
    original_contrast = contrast_scores(original_frames)
    compressed_contrast = contrast_scores(compressed_frames)
    enhanced_contrast = contrast_scores(enhanced_frames)
    smoothed_contrast = contrast_scores(smoothed_frames)

    x_diff = list(range(1, len(compressed_diff) + 1))
    x_psnr = list(range(len(enhanced_psnr)))
    x_frame = list(range(len(original_frames)))

    inter_frame_diff_fig = _plot_line_chart(
        x_diff,
        [
            ("Compressed", compressed_diff, "#ff4d4d"),
            ("Enhanced", enhanced_diff, "#ffb347"),
            ("Smoothed", smoothed_diff, "#55d66b"),
        ],
        "Inter-frame Difference Over Time",
        "Mean Absolute Difference",
    )

    temporal_ssim_fig = _plot_line_chart(
        x_diff,
        [
            ("Compressed", compressed_tssim, "#ff4d4d"),
            ("Enhanced", enhanced_tssim, "#ffb347"),
            ("Smoothed", smoothed_tssim, "#55d66b"),
        ],
        "Temporal SSIM Over Time",
        "SSIM",
    )

    flickering_indices = [
        float(np.std(compressed_diff)) if compressed_diff else 0.0,
        float(np.std(enhanced_diff)) if enhanced_diff else 0.0,
        float(np.std(smoothed_diff)) if smoothed_diff else 0.0,
    ]
    flickering_index_fig = _plot_bar_chart(
        ["Compressed", "Enhanced", "Smoothed"],
        flickering_indices,
        ["#ff4d4d", "#ffb347", "#55d66b"],
        "Flickering Index Comparison",
        "Std. Dev. of Inter-frame Difference",
    )

    psnr_fig = _plot_line_chart(
        x_psnr,
        [
            ("Enhanced", enhanced_psnr, "#ffb347"),
            ("Smoothed", smoothed_psnr, "#55d66b"),
        ],
        "PSNR Stability Over Time",
        "PSNR (dB)",
    )

    sharpness_fig = _plot_line_chart(
        x_frame,
        [
            ("Original", original_sharpness, "#7fb3ff"),
            ("Compressed", compressed_sharpness, "#ff4d4d"),
            ("Enhanced", enhanced_sharpness, "#ffb347"),
            ("Smoothed", smoothed_sharpness, "#55d66b"),
        ],
        "Sharpness Over Time",
        "Gradient Variance",
    )

    contrast_fig = _plot_line_chart(
        x_frame,
        [
            ("Original", original_contrast, "#7fb3ff"),
            ("Compressed", compressed_contrast, "#ff4d4d"),
            ("Enhanced", enhanced_contrast, "#ffb347"),
            ("Smoothed", smoothed_contrast, "#55d66b"),
        ],
        "Contrast Over Time",
        "Luminance Std. Dev.",
    )

    enhanced_flicker = float(np.mean(enhanced_diff)) if enhanced_diff else 0.0
    smoothed_flicker = float(np.mean(smoothed_diff)) if smoothed_diff else 0.0
    enhanced_tssim_mean = float(np.mean(enhanced_tssim)) if enhanced_tssim else 0.0
    smoothed_tssim_mean = float(np.mean(smoothed_tssim)) if smoothed_tssim else 0.0

    flickering_reduction_pct = (
        ((enhanced_flicker - smoothed_flicker) / enhanced_flicker) * 100.0 if enhanced_flicker else 0.0
    )
    temporal_ssim_improvement_pct = (
        ((smoothed_tssim_mean - enhanced_tssim_mean) / enhanced_tssim_mean) * 100.0
        if enhanced_tssim_mean
        else 0.0
    )

    summary = {
        "flickering_reduction_pct": flickering_reduction_pct,
        "temporal_ssim_improvement_pct": temporal_ssim_improvement_pct,
        "average_smoothed_psnr": float(np.mean(smoothed_psnr)) if smoothed_psnr else 0.0,
        "enhanced_sharpness_gain_pct": (
            ((float(np.mean(enhanced_sharpness)) - float(np.mean(compressed_sharpness))) / max(float(np.mean(compressed_sharpness)), 1e-6))
            * 100.0
        ),
        "smoothed_sharpness_gain_pct": (
            ((float(np.mean(smoothed_sharpness)) - float(np.mean(compressed_sharpness))) / max(float(np.mean(compressed_sharpness)), 1e-6))
            * 100.0
        ),
        "enhanced_contrast_gain_pct": (
            ((float(np.mean(enhanced_contrast)) - float(np.mean(compressed_contrast))) / max(float(np.mean(compressed_contrast)), 1e-6))
            * 100.0
        ),
        "smoothed_contrast_gain_pct": (
            ((float(np.mean(smoothed_contrast)) - float(np.mean(compressed_contrast))) / max(float(np.mean(compressed_contrast)), 1e-6))
            * 100.0
        ),
    }

    summary_table = pd.DataFrame(
        {
            "Metric": [
                "Mean Inter-frame Difference",
                "Mean Temporal SSIM",
                "Flickering Index",
                "Average PSNR",
                "Average Sharpness",
                "Average Contrast",
            ],
            "Compressed": [
                float(np.mean(compressed_diff)) if compressed_diff else 0.0,
                float(np.mean(compressed_tssim)) if compressed_tssim else 0.0,
                flickering_indices[0],
                np.nan,
                float(np.mean(compressed_sharpness)),
                float(np.mean(compressed_contrast)),
            ],
            "Enhanced": [
                enhanced_flicker,
                enhanced_tssim_mean,
                flickering_indices[1],
                float(np.mean(enhanced_psnr)) if enhanced_psnr else 0.0,
                float(np.mean(enhanced_sharpness)),
                float(np.mean(enhanced_contrast)),
            ],
            "Smoothed": [
                smoothed_flicker,
                smoothed_tssim_mean,
                flickering_indices[2],
                summary["average_smoothed_psnr"],
                float(np.mean(smoothed_sharpness)),
                float(np.mean(smoothed_contrast)),
            ],
        }
    )

    return {
        "inter_frame_diff_fig": inter_frame_diff_fig,
        "temporal_ssim_fig": temporal_ssim_fig,
        "flickering_index_fig": flickering_index_fig,
        "psnr_fig": psnr_fig,
        "sharpness_fig": sharpness_fig,
        "contrast_fig": contrast_fig,
        "summary": summary,
        "summary_table": summary_table,
    }
