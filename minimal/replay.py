import argparse
import time
import threading
from pathlib import Path

import h5py
import cv2
import numpy as np

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    with h5py.File(path, "r") as handle:
        time_s = np.asarray(handle["time_s"], dtype=np.float64)
        commands = np.asarray(handle["command_position"], dtype=np.float32)
        raw_keys = handle["command_position"].attrs["command_keys"]
        command_keys = [
            key.decode("utf-8") if isinstance(key, (bytes, np.bytes_)) else str(key)
            for key in raw_keys
        ]
        if "images_bgr_uint8" in handle:
            images = np.asarray(handle["images_bgr_uint8"], dtype=np.uint8)
        else:
            images = None
    return time_s, commands, command_keys, images


class LiveCaptureWorker(threading.Thread):
    def __init__(self, device: str, width: int, height: int, target_fps: float) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.width = width
        self.height = height
        self.target_interval = 1.0 / target_fps
        self.stop_event = threading.Event()
        self.frames: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.error: Exception | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            self.error = RuntimeError(f"Failed to open live camera {self.device}.")
            return
        if not capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")):
            self.error = RuntimeError("Failed to set live camera pixel format to MJPG.")
            capture.release()
            return
        if not capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width):
            self.error = RuntimeError("Failed to set live camera width.")
            capture.release()
            return
        if not capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height):
            self.error = RuntimeError("Failed to set live camera height.")
            capture.release()
            return
        if not capture.set(cv2.CAP_PROP_FPS, 30.0):
            self.error = RuntimeError("Failed to set live camera fps.")
            capture.release()
            return

        next_ts = time.perf_counter()
        try:
            start_time = time.perf_counter()
            while not self.stop_event.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    self.error = RuntimeError("Failed to read frame from live camera.")
                    break
                if frame.shape[1] != self.width or frame.shape[0] != self.height or frame.shape[2] != 3:
                    self.error = RuntimeError("Live camera frame has unexpected dimensions.")
                    break
                self.frames.append(frame.copy())
                self.timestamps.append(time.perf_counter() - start_time)
                next_ts += self.target_interval
                remaining = next_ts - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            capture.release()


def save_side_by_side_video(
    recorded_images: np.ndarray,
    live_frames: list[np.ndarray],
    live_times: list[float],
    output_path: Path,
    times: np.ndarray,
    control_fps: float,
) -> None:
    frame_count = len(recorded_images)
    if frame_count == 0:
        raise RuntimeError("No recorded frames available to save the replay video.")
    if len(live_frames) == 0 or len(live_times) == 0:
        raise RuntimeError("Live capture did not produce any frames.")

    duration = times[-1] - times[0]
    if duration <= 0:
        fps = control_fps
    else:
        fps = (len(recorded_images) - 1) / duration
    if fps <= 0:
        fps = control_fps

    live_times_arr = np.asarray(live_times, dtype=np.float64)
    recorded_times = times[:frame_count]
    indices = np.searchsorted(live_times_arr, recorded_times, side="left")
    indices = np.clip(indices, 0, len(live_times_arr) - 1)
    prev_indices = np.clip(indices - 1, 0, len(live_times_arr) - 1)
    choose_prev = (
        np.abs(live_times_arr[indices] - recorded_times)
        >= np.abs(live_times_arr[prev_indices] - recorded_times)
    )
    indices[choose_prev] = prev_indices[choose_prev]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = recorded_images.shape[1]
    width = recorded_images.shape[2]
    combined_width = width * 2
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (combined_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for idx in range(frame_count):
            recorded_frame = recorded_images[idx]
            live_frame = live_frames[indices[idx]]
            if live_frame.shape[0] != height or live_frame.shape[1] != width:
                raise RuntimeError("Live frame size mismatch during video encoding.")
            combined = np.hstack((recorded_frame, live_frame))
            writer.write(combined)
    finally:
        writer.release()


def replay_episode(
    robot: SO101Follower,
    time_s: np.ndarray,
    commands: np.ndarray,
    command_keys: list[str],
    control_fps: float,
    images: np.ndarray,
    live_camera_device: str,
    video_output: Path,
) -> None:
    if images is None:
        raise ValueError("Episode does not contain recorded images; cannot save video.")

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

    height, width = images.shape[1], images.shape[2]
    live_worker = LiveCaptureWorker(live_camera_device, width, height, target_fps=30.0)
    live_worker.start()

    start = time.perf_counter()
    for idx, (step_time, command_row) in enumerate(zip(t_grid, interp_commands, strict=False)):
        target = start + step_time
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        action = {key: float(value) for key, value in zip(command_keys, command_row, strict=False)}
        robot.send_action(action)

    print("Replay complete.", flush=True)
    live_worker.stop()
    live_worker.join()
    if live_worker.error is not None:
        raise live_worker.error
    save_side_by_side_video(
        images,
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
        default="/home/tianchongj/workspace/lerobot/outputs/joint_records/episode_0000.h5",
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Follower USB port (e.g. /dev/ttyACM0).")
    parser.add_argument("--follower-id", default="follower0", help="Follower calibration id.")
    parser.add_argument("--control-fps", type=float, default=210.0, help="Control loop frequency (Hz).")
    parser.add_argument("--live-camera", default="/dev/video4", help="Live camera device for replay visualization.")
    parser.add_argument(
        "--video-output",
        default="/home/tianchongj/workspace/lerobot/outputs/replay_videos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episode).expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    time_s, commands, command_keys, images = load_episode(episode_path)

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
            images=images,
            live_camera_device=args.live_camera,
            video_output=video_output,
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

