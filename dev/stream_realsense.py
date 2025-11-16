
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BASE_SENSOR_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record aligned RGB and depth streams from an Intel RealSense camera.")
    parser.add_argument("--fps", type=int, default=30, help="Target recording rate. Must evenly divide 30.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Recording duration in seconds.")
    parser.add_argument("--output-dir", type=str, default="/home/ripl/workspace/lerobot/dev/scratch", help="Directory where MP4 files are written.")
    parser.add_argument("--depth-max-m", type=float, default=4.0, help="Depth range mapped to white in the depth video.")
    return parser.parse_args()


def depth_to_bgr(depth_image: np.ndarray, max_depth_units: int) -> np.ndarray:
    scale = depth_image.astype(np.float32) / float(max_depth_units)
    normalized = np.clip(scale, 0.0, 1.0)
    depth_uint8 = (normalized * 255.0).astype(np.uint8)
    return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("FPS must be positive.")
    if BASE_SENSOR_FPS % args.fps != 0:
        raise ValueError(f"FPS must evenly divide {BASE_SENSOR_FPS}.")
    if args.seconds <= 0:
        raise ValueError("Recording duration must be positive.")
    if args.depth_max_m <= 0:
        raise ValueError("Depth range must be positive.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_path = output_dir / f"color_{args.fps}hz_{timestamp}.mp4"
    depth_path = output_dir / f"depth_{args.fps}hz_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_writer = cv2.VideoWriter(str(color_path), fourcc, args.fps, (FRAME_WIDTH, FRAME_HEIGHT))
    depth_writer = cv2.VideoWriter(str(depth_path), fourcc, args.fps, (FRAME_WIDTH, FRAME_HEIGHT))

    if not color_writer.isOpened():
        raise RuntimeError(f"Unable to open color writer for {color_path}.")
    if not depth_writer.isOpened():
        raise RuntimeError(f"Unable to open depth writer for {depth_path}.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.rgb8, BASE_SENSOR_FPS)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, BASE_SENSOR_FPS)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    max_depth_units = int(args.depth_max_m / depth_scale)
    if max_depth_units <= 0:
        raise ValueError("Depth range is too small for the current sensor scale.")

    align = rs.align(rs.stream.color)
    frames_needed = int(args.fps * args.seconds)
    stride = BASE_SENSOR_FPS // args.fps
    frame_index = 0
    recorded_frames = 0

    while recorded_frames < frames_needed:
        frameset = pipeline.wait_for_frames()
        aligned_frames = align.process(frameset)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to retrieve frames from the camera.")

        if frame_index % stride == 0:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            depth_bgr = depth_to_bgr(depth_image, max_depth_units)
            color_writer.write(color_bgr)
            depth_writer.write(depth_bgr)
            recorded_frames += 1
        frame_index += 1

    color_writer.release()
    depth_writer.release()
    pipeline.stop()

    print(f"Color video saved to {color_path}")
    print(f"Depth video saved to {depth_path}")


if __name__ == "__main__":
    main()

