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
import pyrealsense2 as rs
from queue import Empty

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


class _RealSenseCamera:
    def __init__(self, serial_number: str, width: int, height: int, fps: int) -> None:
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile: rs.pipeline_profile | None = None
        self.align = rs.align(rs.stream.color)
        self.depth_scale: float = 0.0

    def start(self) -> None:
        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def stop(self) -> None:
        self.pipeline.stop()

    def capture_frames(self) -> tuple[np.ndarray, np.ndarray]:
        frameset = self.pipeline.wait_for_frames()
        aligned = self.align.process(frameset)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture RealSense frames.")
        color_rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        if color_bgr.shape[0] != self.height or color_bgr.shape[1] != self.width or color_bgr.shape[2] != 3:
            raise RuntimeError("RealSense color frame has unexpected dimensions.")
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            raise RuntimeError("RealSense depth frame has unexpected dimensions.")
        return color_bgr, depth.astype(np.uint16)


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
        realsense_serials: list[str],
        stream_width: int,
        stream_height: int,
        stream_fps: int,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.record_stride = record_stride
        self.record_frames = record_frames
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_width = stream_width
        self.camera_height = stream_height
        self.camera_fps = stream_fps
        self.realsense_cameras = [
            _RealSenseCamera(serial_number=serial, width=stream_width, height=stream_height, fps=stream_fps)
            for serial in realsense_serials
        ]
        for camera in self.realsense_cameras:
            camera.start()
        self.camera_ids = [camera.serial_number for camera in self.realsense_cameras]
        self.camera_lookup = {camera.serial_number: camera for camera in self.realsense_cameras}

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
        self.record_buffer: dict[str, object] | None = None
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
        for camera in self.realsense_cameras:
            camera.stop()

    def _start_recording(self) -> None:
        self._drain_snapshot_queue()
        self.record_start_time = None
        self.record_buffer = {
            "time": np.zeros(self.record_frames, dtype=np.float64),
            "joints": np.zeros((self.record_frames, len(self.joint_keys)), dtype=np.float32),
            "commands": np.zeros((self.record_frames, len(self.command_keys)), dtype=np.float32),
            "color": {
                camera_id: np.zeros(
                    (self.record_frames, self.camera_height, self.camera_width, 3),
                    dtype=np.uint8,
                )
                for camera_id in self.camera_ids
            },
            "depth": {
                camera_id: np.zeros(
                    (self.record_frames, self.camera_height, self.camera_width),
                    dtype=np.uint16,
                )
                for camera_id in self.camera_ids
            },
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
            color_frames, depth_frames = self._capture_realsense_frames()
            for camera_id in self.camera_ids:
                self.record_buffer["color"][camera_id][idx] = color_frames[camera_id]
                self.record_buffer["depth"][camera_id][idx] = depth_frames[camera_id]
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
                "color": {
                    camera_id: self.record_buffer["color"][camera_id][: self.frames_collected]
                    for camera_id in self.camera_ids
                },
                "depth": {
                    camera_id: self.record_buffer["depth"][camera_id][: self.frames_collected]
                    for camera_id in self.camera_ids
                },
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

    def _capture_realsense_frames(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        color_frames: dict[str, np.ndarray] = {}
        depth_frames: dict[str, np.ndarray] = {}
        for camera in self.realsense_cameras:
            color, depth = camera.capture_frames()
            color_frames[camera.serial_number] = color
            depth_frames[camera.serial_number] = depth
        return color_frames, depth_frames

    def _save_episode(self, episode: dict[str, object]) -> None:
        episode_path = self.output_dir / f"episode_{self.episode_index:04d}.h5"
        self.episode_index += 1
        with h5py.File(episode_path, "w") as handle:
            handle.create_dataset("time_s", data=episode["time"])
            joint_dataset = handle.create_dataset("joint_position", data=episode["joints"])
            command_dataset = handle.create_dataset("command_position", data=episode["commands"])
            joint_dataset.attrs["joint_keys"] = np.array(
                self.joint_keys, dtype=h5py.string_dtype("utf-8")
            )
            command_dataset.attrs["command_keys"] = np.array(
                self.command_keys, dtype=h5py.string_dtype("utf-8")
            )
            color_group = handle.create_group("realsense_color_bgr_uint8")
            depth_group = handle.create_group("realsense_depth_z16_uint16")
            for camera_id in self.camera_ids:
                color_dataset = color_group.create_dataset(camera_id, data=episode["color"][camera_id])
                depth_dataset = depth_group.create_dataset(camera_id, data=episode["depth"][camera_id])
                camera = self.camera_lookup[camera_id]
                color_dataset.attrs["height"] = self.camera_height
                color_dataset.attrs["width"] = self.camera_width
                color_dataset.attrs["channels"] = 3
                color_dataset.attrs["fps"] = self.camera_fps
                depth_dataset.attrs["height"] = self.camera_height
                depth_dataset.attrs["width"] = self.camera_width
                depth_dataset.attrs["units"] = "meters"
                depth_dataset.attrs["scale"] = camera.depth_scale
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
    parser.add_argument("--record-fps", type=float, default=10.0, help="Recording frequency (Hz).")
    parser.add_argument("--stream-fps", type=float, default=30.0, help="RealSense streaming frequency (Hz).")
    parser.add_argument("--stream-width", type=int, default=1280, help="RealSense color/depth width.")
    parser.add_argument("--stream-height", type=int, default=720, help="RealSense color/depth height.")
    parser.add_argument("--episode-length", type=float, default=8.0, help="Episode length in seconds.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("/home/ripl/workspace/lerobot/datasets/nov14_50").resolve()),
        help="Directory for recorded episodes.",
    )
    parser.add_argument("--follower-id", default="follower0", help="Follower calibration id.")
    parser.add_argument("--leader-id", default="leader0", help="Leader calibration id.")
    parser.add_argument("--teleop-port", default="/dev/ttyACM1", help="Leader USB port (e.g. /dev/ttyACM1).")
    parser.add_argument(
        "--realsense-serial",
        action="append",
        default=["817612070096", "817412070743"],
        help="RealSense serial number. Provide multiple times for multi-camera capture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    control_rate = int(round(args.control_fps))
    record_rate = int(round(args.record_fps))
    stream_rate = int(round(args.stream_fps))
    stream_width = int(args.stream_width)
    stream_height = int(args.stream_height)

    if not args.realsense_serial:
        raise ValueError("At least one --realsense-serial must be provided.")
    if control_rate % record_rate != 0:
        raise ValueError("control-fps must be a multiple of record-fps.")
    if stream_rate % record_rate != 0:
        raise ValueError("stream-fps must be a multiple of record-fps.")

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
            realsense_serials=args.realsense_serial,
            stream_width=stream_width,
            stream_height=stream_height,
            stream_fps=stream_rate,
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

