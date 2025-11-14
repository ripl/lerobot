import argparse
import contextlib
import queue
import select
import sys
import termios
import time
import threading
import tty
from pathlib import Path

import h5py
import cv2
import numpy as np
from queue import Empty

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


class ControlWorker(threading.Thread):
    def __init__(
        self,
        robot: SO101Follower,
        teleop: SO101Leader,
        control_interval_s: float,
        record_stride: int,
        snapshot_queue: queue.Queue,
        stop_event: threading.Event,
        joint_keys: list[str],
    ) -> None:
        super().__init__(daemon=True)
        self.robot = robot
        self.teleop = teleop
        self.control_interval = control_interval_s
        self.record_stride = record_stride
        self.snapshot_queue = snapshot_queue
        self.stop_event = stop_event
        self.joint_keys = joint_keys
        self.lock = threading.Lock()
        self.recording = False
        self.record_start_step = 0
        self.step = 0
        self.start_time = time.perf_counter()

    def run(self) -> None:
        self.start_time = time.perf_counter()
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            command = self.teleop.get_action()
            self.robot.send_action(command)
            observation = self.robot.get_observation()

            self.step += 1
            current_step = self.step

            with self.lock:
                recording = self.recording
                record_start = self.record_start_step

            if recording:
                relative = current_step - record_start
                if relative >= 0 and relative % self.record_stride == 0:
                    timestamp = time.perf_counter() - self.start_time
                    command_snapshot = {key: float(value) for key, value in command.items()}
                    observation_snapshot = {
                        key: float(observation[key]) for key in self.joint_keys
                    }
                    self.snapshot_queue.put((timestamp, command_snapshot, observation_snapshot))

            elapsed = time.perf_counter() - loop_start
            remaining = self.control_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def enable_recording(self) -> None:
        with self.lock:
            self.recording = True
            self.record_start_step = self.step + 1

    def disable_recording(self) -> None:
        with self.lock:
            self.recording = False


class JointRecorder:
    def __init__(
        self,
        robot: SO101Follower,
        teleop: SO101Leader,
        control_interval_s: float,
        record_stride: int,
        record_frames: int,
        output_dir: Path,
        camera_device: str,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.record_stride = record_stride
        self.record_frames = record_frames
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_width = 2592
        self.camera_height = 1944
        self.camera_fps = 30.0
        self.camera = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera device {camera_device}.")
        if not self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")):
            raise RuntimeError("Failed to set camera pixel format to MJPG.")
        if not self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width):
            raise RuntimeError("Failed to set camera width.")
        if not self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height):
            raise RuntimeError("Failed to set camera height.")
        if not self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps):
            raise RuntimeError("Failed to set camera fps.")

        sample = self.robot.get_observation()
        self.joint_keys = [
            key for key, value in sample.items() if isinstance(value, (int, float))
        ]
        command_sample = self.teleop.get_action()
        self.command_keys = sorted(command_sample.keys())

        self.snapshot_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.record_start_time: float | None = None
        self.control_worker = ControlWorker(
            robot=self.robot,
            teleop=self.teleop,
            control_interval_s=control_interval_s,
            record_stride=record_stride,
            snapshot_queue=self.snapshot_queue,
            stop_event=self.stop_event,
            joint_keys=self.joint_keys,
        )
        self.control_worker.start()

        self.episode_index = 0
        self.record_buffer: dict[str, np.ndarray] | None = None
        self.frames_collected = 0
        self.recording_active = False
        self.awaiting_decision = False

    def run(self) -> None:
        print("Controls: press 's' to record, 'y'/'n' to keep or discard, 'q' to quit.")
        print("Press 's' to start a new episode.")
        with _raw_mode(sys.stdin):
            while not self.stop_event.is_set():
                key = _poll_key()
                if key:
                    key = key.lower()
                    if key == "q":
                        print("\nQuit requested.")
                        self.control_worker.disable_recording()
                        self._cancel_recording()
                        self.stop_event.set()
                        break
                    if self.awaiting_decision and key in {"y", "n"}:
                        self._complete_episode(save=(key == "y"))
                        continue
                    if key == "s" and not self.recording_active and not self.awaiting_decision:
                        self._start_recording()
                        continue
                self._collect_snapshots()
                time.sleep(0.001)
        print("Recorder stopped.")

    def close(self) -> None:
        self._cancel_recording()
        self.stop_event.set()
        self.control_worker.disable_recording()
        self.control_worker.join()
        self.camera.release()

    def _start_recording(self) -> None:
        self._drain_snapshot_queue()
        self.record_start_time = None
        self.record_buffer = {
            "time": np.zeros(self.record_frames, dtype=np.float64),
            "joints": np.zeros((self.record_frames, len(self.joint_keys)), dtype=np.float32),
            "commands": np.zeros((self.record_frames, len(self.command_keys)), dtype=np.float32),
            "images": np.zeros(
                (self.record_frames, self.camera_height, self.camera_width, 3),
                dtype=np.uint8,
            ),
        }
        self.frames_collected = 0
        self.recording_active = True
        self.control_worker.enable_recording()
        print("\nRecording episode...", flush=True)

    def _collect_snapshots(self) -> None:
        if not self.recording_active or self.record_buffer is None:
            return

        while True:
            try:
                timestamp, command_snapshot, observation_snapshot = self.snapshot_queue.get_nowait()
            except Empty:
                break

            idx = self.frames_collected
            if idx >= self.record_frames:
                continue

            if self.record_start_time is None:
                self.record_start_time = timestamp
            self.record_buffer["time"][idx] = timestamp - self.record_start_time
            self.record_buffer["commands"][idx] = np.array(
                [command_snapshot[key] for key in self.command_keys],
                dtype=np.float32,
            )
            for joint_id, joint_key in enumerate(self.joint_keys):
                self.record_buffer["joints"][idx, joint_id] = float(observation_snapshot[joint_key])
            self.record_buffer["images"][idx] = self._capture_image()
            self.frames_collected += 1

        if self.recording_active and self.frames_collected >= self.record_frames:
            self.control_worker.disable_recording()
            self.recording_active = False
            self.awaiting_decision = True
            print("\nEpisode finished. Save episode? [y/n] ", end="", flush=True)

    def _complete_episode(self, *, save: bool) -> None:
        if self.record_buffer is None:
            return

        if save:
            episode = {
                "time": self.record_buffer["time"][: self.frames_collected],
                "joints": self.record_buffer["joints"][: self.frames_collected],
                "commands": self.record_buffer["commands"][: self.frames_collected],
                "images": self.record_buffer["images"][: self.frames_collected],
            }
            self._save_episode(episode)
        else:
            print("\nEpisode discarded.")

        self.record_buffer = None
        self.awaiting_decision = False
        self.frames_collected = 0
        self.record_start_time = None
        self._drain_snapshot_queue()
        print("Ready. Press 's' to start a new episode.")

    def _capture_image(self) -> np.ndarray:
        success, frame = self.camera.read()
        if not success or frame is None:
            raise RuntimeError("Failed to read frame from camera.")
        if frame.shape[0] != self.camera_height or frame.shape[1] != self.camera_width or frame.shape[2] != 3:
            raise RuntimeError("Camera frame has unexpected dimensions.")
        return frame

    def _save_episode(self, episode: dict[str, np.ndarray]) -> None:
        episode_path = self.output_dir / f"episode_{self.episode_index:04d}.h5"
        self.episode_index += 1
        with h5py.File(episode_path, "w") as handle:
            handle.create_dataset("time_s", data=episode["time"])
            joint_dataset = handle.create_dataset("joint_position", data=episode["joints"])
            command_dataset = handle.create_dataset("command_position", data=episode["commands"])
            images_dataset = handle.create_dataset("images_bgr_uint8", data=episode["images"])
            joint_dataset.attrs["joint_keys"] = np.array(
                self.joint_keys, dtype=h5py.string_dtype("utf-8")
            )
            command_dataset.attrs["command_keys"] = np.array(
                self.command_keys, dtype=h5py.string_dtype("utf-8")
            )
            images_dataset.attrs["height"] = self.camera_height
            images_dataset.attrs["width"] = self.camera_width
            images_dataset.attrs["channels"] = 3
        print(f"\nSaved episode to {episode_path}", flush=True)

    def _drain_snapshot_queue(self) -> None:
        while True:
            try:
                self.snapshot_queue.get_nowait()
            except Empty:
                break

    def _cancel_recording(self) -> None:
        self.recording_active = False
        self.awaiting_decision = False
        self.record_buffer = None
        self.frames_collected = 0
        self.record_start_time = None
        self._drain_snapshot_queue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal joint state recorder.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Follower USB port (e.g. /dev/ttyACM0).")
    parser.add_argument("--control-fps", type=float, default=210.0, help="Control loop frequency (Hz).")
    parser.add_argument("--record-fps", type=float, default=30.0, help="Recording frequency (Hz).")
    parser.add_argument("--episode-length", type=float, default=10.0, help="Episode length in seconds.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("/home/tianchongj/workspace/lerobot/outputs/joint_records").resolve()),
        help="Directory for recorded episodes.",
    )
    parser.add_argument("--follower-id", default="follower0", help="Follower calibration id.")
    parser.add_argument("--leader-id", default="leader0", help="Leader calibration id.")
    parser.add_argument("--teleop-port", default="/dev/ttyACM1", help="Leader USB port (e.g. /dev/ttyACM1).")
    parser.add_argument(
        "--camera-device",
        default="/dev/video4",
        help="Video device path (e.g. /dev/video4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    control_rate = int(round(args.control_fps))
    record_rate = int(round(args.record_fps))
    if control_rate <= 0 or record_rate <= 0:
        raise ValueError("Frequencies must be positive.")

    if control_rate % record_rate != 0:
        raise ValueError("control-fps must be a multiple of record-fps.")

    control_interval = 1.0 / control_rate
    record_stride = control_rate // record_rate
    record_frames = max(1, int(round(args.episode_length * record_rate)))

    follower_config = SO101FollowerConfig(
        id=args.follower_id,
        port=args.port,
    )
    leader_config = SO101LeaderConfig(
        id=args.leader_id,
        port=args.teleop_port,
    )
    robot = SO101Follower(follower_config)
    teleop = SO101Leader(leader_config)
    robot.connect()
    teleop.connect()
    recorder: JointRecorder | None = None
    try:
        recorder = JointRecorder(
            robot=robot,
            teleop=teleop,
            control_interval_s=control_interval,
            record_stride=record_stride,
            record_frames=record_frames,
            output_dir=output_dir,
            camera_device=args.camera_device,
        )
        recorder.run()
    finally:
        if recorder is not None:
            recorder.close()
        robot.disconnect()
        teleop.disconnect()


@contextlib.contextmanager
def _raw_mode(stream):
    fd = stream.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _poll_key() -> str:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        return sys.stdin.read(1)
    return ""


if __name__ == "__main__":
    main()

