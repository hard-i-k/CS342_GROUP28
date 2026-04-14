import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st

from modules.frame_enhancer import enhance_frames
from modules.optical_flow_smoother import smooth_frames_with_optical_flow
from modules.realtime_pipeline import RealTimePipeline
from modules.temporal_metrics import compute_temporal_metrics
from modules.video_processor import load_and_compress_video
from utils.plot_utils import apply_plot_style
from utils.video_utils import create_gif_bytes, create_mp4_bytes

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


st.set_page_config(
    page_title="Temporal Consistency & Real-Time Enhancement Pipeline",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def run_full_analysis(video_bytes, alpha, compression_quality, enhancement_strength):
    # Heavy work is cached so rerunning the same settings is fast inside Streamlit.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_bytes)
        temp_path = tmp_file.name

    try:
        original_frames, compressed_frames, metadata = load_and_compress_video(
            temp_path,
            compression_quality=compression_quality,
        )
        enhanced_frames = enhance_frames(
            compressed_frames,
            source_frames=original_frames,
            enhancement_strength=enhancement_strength,
        )
        smoothed_frames, flow_visualizations = smooth_frames_with_optical_flow(
            enhanced_frames,
            alpha=alpha,
        )
        metrics = compute_temporal_metrics(
            original_frames=original_frames,
            compressed_frames=compressed_frames,
            enhanced_frames=enhanced_frames,
            smoothed_frames=smoothed_frames,
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return {
        "original_frames": original_frames,
        "compressed_frames": compressed_frames,
        "enhanced_frames": enhanced_frames,
        "smoothed_frames": smoothed_frames,
        "flow_visualizations": flow_visualizations,
        "metrics": metrics,
        "metadata": metadata,
    }


def render_frame_strip(title, frames, border_color):
    st.markdown(f"**{title}**")
    if not frames:
        st.info("No frames available.")
        return
    sample_indices = np.linspace(0, len(frames) - 1, num=min(4, len(frames)), dtype=int)
    cols = st.columns(len(sample_indices))
    for col, idx in zip(cols, sample_indices):
        frame = frames[idx]
        bordered = frame.copy()
        cv2.rectangle(
            bordered,
            (0, 0),
            (bordered.shape[1] - 1, bordered.shape[0] - 1),
            border_color,
            3,
        )
        col.image(cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB), caption=f"Frame {idx}")


def render_video_tab(results):
    col1, col2, col3 = st.columns(3)
    with col1:
        render_frame_strip("Compressed", results["compressed_frames"], (40, 40, 220))
        st.image(create_gif_bytes(results["compressed_frames"]), caption="Compressed GIF")
        st.video(create_mp4_bytes(results["compressed_frames"], results["metadata"]["fps"]))
    with col2:
        render_frame_strip("Enhanced", results["enhanced_frames"], (0, 180, 255))
        st.image(create_gif_bytes(results["enhanced_frames"]), caption="Enhanced GIF")
        st.video(create_mp4_bytes(results["enhanced_frames"], results["metadata"]["fps"]))
    with col3:
        render_frame_strip("Temporally Smoothed", results["smoothed_frames"], (0, 200, 120))
        st.image(create_gif_bytes(results["smoothed_frames"]), caption="Smoothed GIF")
        st.video(create_mp4_bytes(results["smoothed_frames"], results["metadata"]["fps"]))

    sample_index = 0
    brightness_col1, brightness_col2, brightness_col3 = st.columns(3)
    brightness_col1.metric(
        "Original Brightness",
        f"{np.mean(results['original_frames'][sample_index]):.0f}",
    )
    brightness_col2.metric(
        "Enhanced Brightness",
        f"{np.mean(results['enhanced_frames'][sample_index]):.0f}",
    )
    brightness_col3.metric(
        "Smoothed Brightness",
        f"{np.mean(results['smoothed_frames'][sample_index]):.0f}",
    )


def render_live_report_visuals(history):
    if not history:
        st.info("Waiting for enough live data to build graphs and heatmaps.")
        return

    history_df = pd.DataFrame(history).iloc[::-1].reset_index(drop=True)

    fig_line, ax_line = plt.subplots(figsize=(10, 4))
    ax_line.plot(
        history_df["timestamp"],
        history_df["enhanced_sharpness_gain_pct"],
        color="#ffb347",
        linewidth=2,
        label="Enhanced Sharpness Change",
    )
    ax_line.plot(
        history_df["timestamp"],
        history_df["smoothed_sharpness_gain_pct"],
        color="#55d66b",
        linewidth=2,
        label="Smoothed Sharpness Change",
    )
    ax_line.plot(
        history_df["timestamp"],
        history_df["enhanced_contrast_gain_pct"],
        color="#ffd166",
        linewidth=2,
        linestyle="--",
        label="Enhanced Contrast Change",
    )
    ax_line.plot(
        history_df["timestamp"],
        history_df["smoothed_contrast_gain_pct"],
        color="#8bd3a8",
        linewidth=2,
        linestyle="--",
        label="Smoothed Contrast Change",
    )
    ax_line.axhline(0, color="#cccccc", linewidth=1, alpha=0.5)
    ax_line.set_title("Live Quality Trends")
    ax_line.set_xlabel("Time")
    ax_line.set_ylabel("Percent Change")
    ax_line.tick_params(axis="x", rotation=30)
    ax_line.grid(alpha=0.25)
    ax_line.legend()
    fig_line.tight_layout()
    st.pyplot(fig_line, use_container_width=True)

    heatmap_df = history_df[
        [
            "enhanced_sharpness_gain_pct",
            "smoothed_sharpness_gain_pct",
            "enhanced_contrast_gain_pct",
            "smoothed_contrast_gain_pct",
            "avg_fps",
            "avg_process_ms",
        ]
    ].copy()
    heatmap_df.columns = [
        "Enhanced Sharpness",
        "Smoothed Sharpness",
        "Enhanced Contrast",
        "Smoothed Contrast",
        "FPS",
        "Process ms",
    ]
    heatmap_df["Enhanced PSNR"] = history_df["enhanced_psnr"]
    heatmap_df["Smoothed PSNR"] = history_df["smoothed_psnr"]
    heatmap_df = heatmap_df[
        [
            "Enhanced Sharpness",
            "Smoothed Sharpness",
            "Enhanced Contrast",
            "Smoothed Contrast",
            "Enhanced PSNR",
            "Smoothed PSNR",
            "FPS",
            "Process ms",
        ]
    ]

    row_names = list(heatmap_df.columns)
    row_arrays = []
    for row_name in row_names:
        values = heatmap_df[row_name].to_numpy(dtype=float)
        if row_name in {"Enhanced Sharpness", "Smoothed Sharpness", "Enhanced Contrast", "Smoothed Contrast"}:
            cmap = plt.get_cmap("RdYlGn")
            norm = mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
        elif row_name in {"Enhanced PSNR", "Smoothed PSNR"}:
            cmap = plt.get_cmap("RdYlGn")
            norm = mcolors.Normalize(vmin=20, vmax=40)
        elif row_name == "FPS":
            cmap = plt.get_cmap("RdYlGn")
            norm = mcolors.Normalize(vmin=5, vmax=30)
        else:
            cmap = plt.get_cmap("RdYlGn_r")
            norm = mcolors.Normalize(vmin=30, vmax=200)
        row_arrays.append(cmap(norm(values)))

    heatmap_rgba = np.stack(row_arrays, axis=0)

    fig_heat, ax_heat = plt.subplots(figsize=(10, 5.4))
    ax_heat.imshow(heatmap_rgba, aspect="auto")
    ax_heat.set_title("Live Report Heatmap")
    ax_heat.set_xticks(range(len(history_df)))
    ax_heat.set_xticklabels(history_df["timestamp"], rotation=30, ha="right")
    ax_heat.set_yticks(range(len(row_names)))
    ax_heat.set_yticklabels(row_names)
    for row in range(len(row_names)):
        for col in range(len(history_df)):
            value = heatmap_df.iloc[col, row]
            ax_heat.text(col, row, f"{value:.1f}", ha="center", va="center", color="black", fontsize=8)
    fig_heat.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)


def render_flow_tab(flow_frames):
    st.caption("Color encodes motion direction while brightness encodes motion magnitude.")
    if not flow_frames:
        st.warning("Optical flow visualizations are not available for this video.")
        return
    preview_indices = np.linspace(0, len(flow_frames) - 1, num=min(6, len(flow_frames)), dtype=int)
    cols = st.columns(3)
    for idx, frame_idx in enumerate(preview_indices):
        cols[idx % 3].image(
            cv2.cvtColor(flow_frames[frame_idx], cv2.COLOR_BGR2RGB),
            caption=f"Flow {frame_idx}",
        )


def render_metrics_tab(metrics):
    st.pyplot(metrics["inter_frame_diff_fig"], use_container_width=True)
    st.pyplot(metrics["temporal_ssim_fig"], use_container_width=True)
    st.pyplot(metrics["flickering_index_fig"], use_container_width=True)
    st.pyplot(metrics["psnr_fig"], use_container_width=True)
    st.pyplot(metrics["sharpness_fig"], use_container_width=True)
    st.pyplot(metrics["contrast_fig"], use_container_width=True)


def render_summary_tab(metrics, metadata):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Flickering Reduced By", f"{metrics['summary']['flickering_reduction_pct']:.2f}%")
    col2.metric(
        "Temporal Consistency Improved By",
        f"{metrics['summary']['temporal_ssim_improvement_pct']:.2f}%",
    )
    col3.metric("Average PSNR", f"{metrics['summary']['average_smoothed_psnr']:.2f} dB")
    col4.metric("Enhanced Sharpness Change", f"{metrics['summary']['enhanced_sharpness_gain_pct']:.2f}%")
    col5.metric("Smoothed Contrast Change", f"{metrics['summary']['smoothed_contrast_gain_pct']:.2f}%")

    st.markdown("### Processing Summary")
    st.write(
        f"Processed `{metadata['frame_count']}` frames at `{metadata['fps']:.2f}` FPS "
        f"with resolution `{metadata['size'][0]}x{metadata['size'][1]}`."
    )
    st.write(
        "Independent enhancement improves sharpness but introduces frame-to-frame variation. "
        "Optical flow smoothing stabilizes texture by motion-aligning the previous frame before blending."
    )
    st.write("Optical flow is computed at half resolution and then scaled up to improve analysis speed.")
    st.info('If optical flow is slow say "optimize optical flow to use half resolution for speed"')
    st.dataframe(metrics["summary_table"], use_container_width=True)


class StreamlitWebRTCProcessor(VideoProcessorBase):
    def __init__(self, enhancement_strength=0.2):
        self.pipeline = RealTimePipeline(enhancement_strength=enhancement_strength)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        original, enhanced, smoothed, debug = self.pipeline.process_frame(image)
        panels = [original.copy(), enhanced.copy(), smoothed.copy()]
        labels = ["Original", "Enhanced", "Smoothed"]

        for panel, label in zip(panels, labels):
            cv2.rectangle(panel, (0, 0), (150, 36), (20, 20, 20), thickness=-1)
            cv2.rectangle(panel, (0, 0), (255, 255), (0, 170, 255), thickness=2)
            cv2.putText(
                panel,
                label,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        stitched = np.hstack(panels)
        return av.VideoFrame.from_ndarray(stitched, format="bgr24")


def render_realtime_mode():
    enhancement_strength = st.sidebar.slider(
        "Enhancement Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="0 = no enhancement, 1 = full enhancement",
    )
    st.subheader("Real-Time Webcam Mode")
    st.write(
        "This lightweight pipeline uses low-resolution processing, cached flow updates, and lighter enhancement "
        "to keep the webcam stream responsive on a normal CPU."
    )
    st.write("For speed, live optical flow runs on smaller frames and is reused briefly between updates.")
    st.caption('If optical flow is slow say "optimize optical flow to use half resolution for speed"')

    if WEBRTC_AVAILABLE:
        st.info("WebRTC is available. Click start below to launch the webcam stream.")
        st.caption("The live stream stays clean, while performance and quality statistics are shown below it.")
        ctx = webrtc_streamer(
            key="temporal-realtime-demo",
            video_processor_factory=lambda: StreamlitWebRTCProcessor(enhancement_strength=enhancement_strength),
            media_stream_constraints={"video": True, "audio": False},
        )
        stats_placeholder = st.empty()
        report_placeholder = st.empty()

        if ctx.state.playing:
            for _ in range(120):
                processor = ctx.video_processor
                if processor is not None:
                    summary = processor.pipeline.get_live_summary()
                    stats = summary["stats"]
                    latest_report = summary["latest_report"]

                    with stats_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Stable FPS", f"{stats.get('fps', 0.0):.2f}")
                        col2.metric("Stable Processing Time", f"{stats.get('process_time_ms', 0.0):.1f} ms")
                        col3.metric("Enhanced Sharpness Change", f"{stats.get('sharpness_gain_pct', 0.0):+.1f}%")
                        col4.metric("Smoothed Sharpness Change", f"{stats.get('smoothed_sharpness_gain_pct', 0.0):+.1f}%")

                        col5, col6, col7 = st.columns(3)
                        col5.metric("Enhanced Contrast Change", f"{stats.get('contrast_gain_pct', 0.0):+.1f}%")
                        col6.metric(
                            "Smoothed Contrast Change",
                            f"{stats.get('smoothed_contrast_gain_pct', 0.0):+.1f}%",
                        )
                        col7.metric("Processing Resolution", summary.get("process_resolution", "192x192"))

                        bright1, bright2, bright3, psnr1, psnr2 = st.columns(5)
                        latest_frames = summary.get("latest_frames", {})
                        bright1.metric("Original Brightness", f"{latest_frames.get('original_brightness', 0.0):.0f}")
                        bright2.metric("Enhanced Brightness", f"{latest_frames.get('enhanced_brightness', 0.0):.0f}")
                        bright3.metric("Smoothed Brightness", f"{latest_frames.get('smoothed_brightness', 0.0):.0f}")
                        psnr1.metric("Enhanced PSNR vs Original", f"{stats.get('enhanced_psnr', 0.0):.1f} dB")
                        psnr2.metric("Smoothed PSNR vs Original", f"{stats.get('smoothed_psnr', 0.0):.1f} dB")

                    with report_placeholder.container():
                        st.markdown("### 15-Second Quality Report")
                        if latest_report is not None:
                            rep1, rep2, rep3 = st.columns(3)
                            rep1.metric("Average FPS", f"{latest_report['avg_fps']:.2f}")
                            rep2.metric("Enhanced Sharpness Change", f"{latest_report['enhanced_sharpness_gain_pct']:+.1f}%")
                            rep3.metric("Smoothed Sharpness Change", f"{latest_report['smoothed_sharpness_gain_pct']:+.1f}%")
                            st.caption(
                                "A small negative sharpness change is not automatically a bug. It means the subtle "
                                "pipeline was slightly softer than the original during that window. It becomes a concern "
                                "only if it stays consistently negative and also looks visibly worse."
                            )
                        history = summary["report_history"]
                        if history:
                            render_live_report_visuals(history)
                            history_df = pd.DataFrame(history).rename(
                                columns={
                                    "timestamp": "Time",
                                    "avg_fps": "Avg FPS",
                                    "avg_process_ms": "Avg Process (ms)",
                                    "enhanced_sharpness_gain_pct": "Enhanced Sharpness Change (%)",
                                    "smoothed_sharpness_gain_pct": "Smoothed Sharpness Change (%)",
                                    "enhanced_contrast_gain_pct": "Enhanced Contrast Change (%)",
                                    "smoothed_contrast_gain_pct": "Smoothed Contrast Change (%)",
                                    "enhanced_psnr": "Enhanced PSNR vs Original (dB)",
                                    "smoothed_psnr": "Smoothed PSNR vs Original (dB)",
                                }
                            )
                            st.dataframe(history_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Waiting for the first 15-second report window to complete.")
                time.sleep(1)
    else:
        st.warning(
            "`streamlit-webrtc` is not installed. Snapshot mode is available using `st.camera_input()`."
        )
        pipeline = RealTimePipeline(enhancement_strength=enhancement_strength)
        camera_image = st.camera_input("Capture a frame for snapshot processing")
        if camera_image is not None:
            bytes_data = camera_image.getvalue()
            raw_array = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(raw_array, cv2.IMREAD_COLOR)
            original, enhanced, smoothed, debug = pipeline.process_frame(frame)
            col1, col2, col3 = st.columns(3)
            col1.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original")
            col2.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="Enhanced")
            col3.image(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB), caption="Temporally Smoothed")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("FPS", f"{debug['fps']:.2f}")
            metric_col2.metric("Processing Time", f"{debug['process_time_ms']:.1f} ms")
            metric_col3.metric("Sharpness Change", f"{debug['sharpness_gain_pct']:+.1f}%")
            metric_col4.metric("Contrast Change", f"{debug['contrast_gain_pct']:+.1f}%")
            bright1, bright2, bright3 = st.columns(3)
            bright1.metric("Original Brightness", f"{debug['original_brightness']:.0f}")
            bright2.metric("Enhanced Brightness", f"{debug['enhanced_brightness']:.0f}")
            bright3.metric("Smoothed Brightness", f"{debug['smoothed_brightness']:.0f}")
            st.caption(f"Processing resolution: {debug['process_resolution']}")


def main():
    apply_plot_style()
    st.title("Temporal Consistency & Real-Time Enhancement Pipeline")
    st.caption("Implementing future work extensions of the Keyframe-Based GAN videoconferencing paper")

    mode = st.radio(
        "Choose Mode",
        ["Video Analysis Mode", "Real-Time Mode"],
        horizontal=True,
    )

    if mode == "Video Analysis Mode":
        enhancement_strength = st.sidebar.slider(
            "Enhancement Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="0 = no enhancement, 1 = full enhancement",
        )
        st.subheader("Video Upload Mode")
        uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
        alpha = st.slider("Temporal Blend Alpha", min_value=0.0, max_value=0.08, value=0.08, step=0.01)
        compression_quality = st.slider(
            "Compression Severity (JPEG Quality)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        if uploaded_file is not None and st.button("Run Analysis", type="primary"):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            try:
                progress_text.write("Reading video and simulating compression artifacts...")
                progress_bar.progress(25)
                results = run_full_analysis(
                    uploaded_file.getvalue(),
                    alpha,
                    compression_quality,
                    enhancement_strength,
                )
                progress_text.write("Preparing previews and temporal metrics...")
                progress_bar.progress(75)

                tabs = st.tabs(
                    ["Video Comparison", "Optical Flow", "Temporal Metrics", "Summary"]
                )
                with tabs[0]:
                    render_video_tab(results)
                with tabs[1]:
                    render_flow_tab(results["flow_visualizations"])
                with tabs[2]:
                    render_metrics_tab(results["metrics"])
                with tabs[3]:
                    render_summary_tab(results["metrics"], results["metadata"])

                progress_bar.progress(100)
                progress_text.success("Analysis complete.")
            except Exception as exc:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
        elif uploaded_file is None:
            st.info("Upload a short face video of up to 30 seconds to begin analysis.")
    else:
        render_realtime_mode()


if __name__ == "__main__":
    main()
