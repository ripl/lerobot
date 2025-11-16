import argparse
import time
import threading
from pathlib import Path

import h5py
import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


class _RealSenseReader:
    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.align = rs.align(rs.stream.color)

    def start(self) -> None:
        self.pipeline.start(self.config)

    def stop(self) -> None:
        self.pipeline.stop()

    def read(self) -> np.ndarray:
        frameset = self.pipeline.wait_for_frames()
        aligned = self.align.process(frameset)
        color_frame = aligned.get_color_frame()
        color_rgb = np.asanyarray(color_frame.get_data())
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        return color_bgr


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        commands = np.asarray(handle["command_position"], dtype=np.float32)
        raw_keys = handle["command_position"].attrs["command_keys"]
        command_keys = [
            key.decode("utf-8") if isinstance(key, (bytes, np.bytes_)) else str(key)
            for key in raw_keys
        ]
        color_group = handle.get("realsense_color_bgr_uint8")
        images: dict[str, np.ndarray] = {}
        for serial in color_group.keys():
            images[serial] = np.asarray(color_group[serial], dtype=np.uint8)
    return time_s, commands, command_keys, images


class LiveCaptureWorker(threading.Thread):
    def __init__(self, serials: list[str], width: int, height: int, target_fps: float) -> None:
        super().__init__(daemon=True)
        self.serials = serials
        self.width = width
        self.height = height
        self.target_interval = 1.0 / target_fps
        self.stop_event = threading.Event()
        self.frames: dict[str, list[np.ndarray]] = {serial: [] for serial in serials}
        self.timestamps: list[float] = []
        self.cameras = [_RealSenseReader(serial, width, height, int(round(target_fps))) for serial in serials]

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        for camera in self.cameras:
            camera.start()
        next_ts = time.perf_counter()
        start_time = time.perf_counter()
        while not self.stop_event.is_set():
            captures = {camera.serial: camera.read() for camera in self.cameras}
            for serial, frame in captures.items():
                self.frames[serial].append(frame.copy())
            self.timestamps.append(time.perf_counter() - start_time)
            next_ts += self.target_interval
            remaining = next_ts - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        for camera in self.cameras:
            camera.stop()


def _interpolate_live_indices(recorded_times: np.ndarray, live_times: list[float]) -> np.ndarray:
    live_times_arr = np.asarray(live_times, dtype=np.float64)
    indices = np.searchsorted(live_times_arr, recorded_times, side="left")
    indices = np.clip(indices, 0, len(live_times_arr) - 1)
    prev_indices = np.clip(indices - 1, 0, len(live_times_arr) - 1)
    choose_prev = (
        np.abs(live_times_arr[indices] - recorded_times)
        >= np.abs(live_times_arr[prev_indices] - recorded_times)
    )
    indices[choose_prev] = prev_indices[choose_prev]
    return indices


def save_two_by_two_video(
    recorded_images: dict[str, np.ndarray],
    live_frames: dict[str, list[np.ndarray]],
    live_times: list[float],
    output_path: Path,
    times: np.ndarray,
    control_fps: float,
) -> None:
    serials = sorted(recorded_images.keys())
    frame_count = min(len(recorded_images[serial]) for serial in serials)

    duration = times[-1] - times[0]
    if duration <= 0:
        fps = control_fps
    else:
        fps = (frame_count - 1) / duration
    if fps <= 0:
        fps = control_fps

    recorded_times = times[:frame_count]
    indices = _interpolate_live_indices(recorded_times, live_times)

    first_stream = recorded_images[serials[0]]
    height = first_stream.shape[1]
    width = first_stream.shape[2]
    combined_width = width * 2
    combined_height = height * 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (combined_width, combined_height),
    )
    for idx in range(frame_count):
        top_row = []
        bottom_row = []
        for serial in serials:
            recorded_frames = recorded_images[serial]
            live_sequence = live_frames[serial]
            recorded_frame = recorded_frames[idx]
            live_frame = live_sequence[indices[idx]]
            top_row.append(recorded_frame)
            bottom_row.append(live_frame)
        top_concat = np.hstack(top_row)
        bottom_concat = np.hstack(bottom_row)
        combined = np.vstack((top_concat, bottom_concat))
        writer.write(combined)
    writer.release()


def replay_episode(
    robot: SO101Follower,
    time_s: np.ndarray,
    commands: np.ndarray,
    command_keys: list[str],
    control_fps: float,
    recorded_streams: dict[str, np.ndarray],
    realsense_serials: list[str],
    video_output: Path,
) -> None:
    times = time_s - time_s[0]
    duration = times[-1]
    control_interval = 1.0 / control_fps
    num_steps = int(round(duration / control_interval)) + 1
    t_grid = np.linspace(0.0, duration, num_steps, dtype=np.float64)

    interp_commands = np.empty((num_steps, commands.shape[1]), dtype=np.float32)
    for joint_id, joint_series in enumerate(commands.T):
        interp_commands[:, joint_id] = np.interp(t_grid, times, joint_series)

    print(
        f"Replaying {num_steps} control steps over {duration:.2f} seconds "
        f"(control {control_fps:.2f} Hz).",
        flush=True,
    )

    sample_stream = next(iter(recorded_streams.values()))
    height, width = sample_stream.shape[1], sample_stream.shape[2]
    live_worker = LiveCaptureWorker(realsense_serials, width, height, target_fps=30.0)
    live_worker.start()

    start = time.perf_counter()
    for step_time, command_row in zip(t_grid, interp_commands, strict=False):
        target = start + step_time
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        action = {key: float(value) for key, value in zip(command_keys, command_row, strict=False)}
        robot.send_action(action)

    print("Replay complete.", flush=True)
    live_worker.stop()
    live_worker.join()
    save_two_by_two_video(
        recorded_streams,
        live_worker.frames,
        live_worker.timestamps,
        video_output,
        times,
        control_fps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a recorded joint-state episode.")
    parser.add_argument(
        "--episode",
        default="/home/ripl/workspace/lerobot/datasets/nov16_50/episode_0032.h5",
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Follower USB port (e.g. /dev/ttyACM0).")
    parser.add_argument("--follower-id", default="follower0", help="Follower calibration id.")
    parser.add_argument("--control-fps", type=float, default=210.0, help="Control loop frequency (Hz).")
    parser.add_argument(
        "--realsense-serial",
        action="append",
        default=["817612070096", "817412070743"],
        help="RealSense serial numbers captured in the episode (order sets video layout).",
    )
    parser.add_argument(
        "--video-output",
        default="/home/ripl/workspace/lerobot/outputs/replay_videos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episode).expanduser().resolve()
    time_s, commands, command_keys, recorded_images = load_episode(episode_path)

    ordered_images = {serial: recorded_images[serial] for serial in args.realsense_serial}

    config = SO101FollowerConfig(
        id=args.follower_id,
        port=args.port,
    )
    robot = SO101Follower(config)
    robot.connect()
    try:
        video_output = (
            Path(args.video_output).expanduser().resolve()
            / f"{episode_path.stem}.mp4"
        )
        replay_episode(
            robot,
            time_s,
            commands,
            command_keys,
            control_fps=args.control_fps,
            recorded_streams=ordered_images,
            realsense_serials=args.realsense_serial,
            video_output=video_output,
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

