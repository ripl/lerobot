import argparse
import contextlib
import multiprocessing as mp
import queue
import select
import sys
import termios
import time
import threading
import tty
from enum import Enum
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
        self.fx: float = 0.0
        self.fy: float = 0.0
        self.ppx: float = 0.0
        self.ppy: float = 0.0
        self.intrinsics_coeffs: np.ndarray | None = None

    def start(self) -> None:
        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx = float(intr.fx)
        self.fy = float(intr.fy)
        self.ppx = float(intr.ppx)
        self.ppy = float(intr.ppy)
        self.intrinsics_coeffs = np.asarray(intr.coeffs, dtype=np.float32)

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


def _camera_worker(
    cmd_conn,
    reply_conn,
    serials: list[str],
    sizes: list[tuple[int, int]],
    fps: int,
) -> None:
    assert len(serials) == len(sizes)
    cameras: dict[str, _RealSenseCamera] = {}
    for serial, size in zip(serials, sizes):
        width = int(size[0])
        height = int(size[1])
        cam = _RealSenseCamera(serial_number=serial, width=width, height=height, fps=int(fps))
        cam.start()
        cameras[serial] = cam

    intrinsics: dict[str, dict[str, object]] = {}
    for serial, cam in cameras.items():
        intrinsics[serial] = {
            "height": int(cam.height),
            "width": int(cam.width),
            "fx": float(cam.fx),
            "fy": float(cam.fy),
            "ppx": float(cam.ppx),
            "ppy": float(cam.ppy),
            "coeffs": cam.intrinsics_coeffs,
            "depth_scale": float(cam.depth_scale),
        }
    reply_conn.send(intrinsics)

    while True:
        cmd = cmd_conn.recv()
        if cmd == "close":
            break
        if cmd == "capture":
            frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for serial, cam in cameras.items():
                color, depth = cam.capture_frames()
                frames[serial] = (color, depth)
            reply_conn.send(frames)

    for cam in cameras.values():
        cam.stop()


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

    def run(self) -> None:
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            command = self.teleop.get_action()
            self.robot.send_action(command)
            observation = self.robot.get_observation()

            self.step += 1
            current_step = self.step

            # with self.lock:
            recording = self.recording
            record_start = self.record_start_step

            if recording:
                relative = current_step - record_start
                if relative >= 0 and relative % self.record_stride == 0:
                    command_snapshot = {key: float(value) for key, value in command.items()}
                    observation_snapshot = {
                        key: float(observation[key]) for key in self.joint_keys
                    }
                    self.snapshot_queue.put((command_snapshot, observation_snapshot))

            elapsed = time.perf_counter() - loop_start
            remaining = self.control_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def enable_recording(self) -> None:
        # with self.lock:
        self.recording = True
        self.record_start_step = self.step + 1

    def disable_recording(self) -> None:
        # with self.lock:
        self.recording = False


class RecorderState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    AWAITING_DECISION = "awaiting_decision"


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
        sizes: list,
        stream_fps: int,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.record_stride = record_stride
        self.record_frames = record_frames
        self.record_dt = control_interval_s * record_stride
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_fps = stream_fps
        assert len(realsense_serials) == len(sizes)
        self.camera_ids = list(realsense_serials)

        parent_cmd, child_cmd = mp.Pipe()
        parent_reply, child_reply = mp.Pipe()
        self._camera_cmd = parent_cmd
        self._camera_reply = parent_reply
        self._camera_proc = mp.Process(
            target=_camera_worker,
            args=(child_cmd, child_reply, realsense_serials, [(int(s[0]), int(s[1])) for s in sizes], int(stream_fps)),
        )
        self._camera_proc.start()

        intrinsics: dict[str, dict[str, object]] = self._camera_reply.recv()
        self.camera_shapes = {
            serial: (int(info["height"]), int(info["width"])) for serial, info in intrinsics.items()
        }
        self.camera_fx = {serial: float(info["fx"]) for serial, info in intrinsics.items()}
        self.camera_fy = {serial: float(info["fy"]) for serial, info in intrinsics.items()}
        self.camera_ppx = {serial: float(info["ppx"]) for serial, info in intrinsics.items()}
        self.camera_ppy = {serial: float(info["ppy"]) for serial, info in intrinsics.items()}
        self.camera_coeffs = {serial: info["coeffs"] for serial, info in intrinsics.items()}
        self.camera_depth_scale = {serial: float(info["depth_scale"]) for serial, info in intrinsics.items()}

        sample = self.robot.get_observation()
        self.joint_keys = [
            key for key, value in sample.items() if isinstance(value, (int, float))
        ]
        command_sample = self.teleop.get_action()
        self.command_keys = sorted(command_sample.keys())

        self.snapshot_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
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
        self.state = RecorderState.IDLE

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
                    if self.state is RecorderState.AWAITING_DECISION and key in {"y", "n"}:
                        self._complete_episode(save=(key == "y"))
                        continue
                    if key == "s" and self.state is RecorderState.IDLE:
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
        self._camera_cmd.send("close")
        self._camera_proc.join()

    def _start_recording(self) -> None:
        self._drain_snapshot_queue()
        self.record_buffer = {
            "time": np.zeros(self.record_frames, dtype=np.float64),
            "joints": np.zeros((self.record_frames, len(self.joint_keys)), dtype=np.float32),
            "commands": np.zeros((self.record_frames, len(self.command_keys)), dtype=np.float32),
            "color": {
                camera_id: np.zeros(
                    (
                        self.record_frames,
                        self.camera_shapes[camera_id][0],
                        self.camera_shapes[camera_id][1],
                        3,
                    ),
                    dtype=np.uint8,
                )
                for camera_id in self.camera_ids
            },
            "depth": {
                camera_id: np.zeros(
                    (
                        self.record_frames,
                        self.camera_shapes[camera_id][0],
                        self.camera_shapes[camera_id][1],
                    ),
                    dtype=np.uint16,
                )
                for camera_id in self.camera_ids
            },
        }
        self.frames_collected = 0
        self.state = RecorderState.RECORDING
        self.control_worker.enable_recording()
        print("\nRecording episode...", flush=True)

    def _collect_snapshots(self) -> None:
        if self.state is not RecorderState.RECORDING or self.record_buffer is None:
            return

        while True:
            try:
                command_snapshot, observation_snapshot = self.snapshot_queue.get_nowait()
            except Empty:
                break

            idx = self.frames_collected
            if idx >= self.record_frames:
                continue

            self.record_buffer["time"][idx] = idx * self.record_dt
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

        if self.state is RecorderState.RECORDING and self.frames_collected >= self.record_frames:
            self.control_worker.disable_recording()
            self.state = RecorderState.AWAITING_DECISION
            print("\nEpisode finished. Save episode? [y/n] ", end="", flush=True)

    def _complete_episode(self, *, save: bool) -> None:
        if self.record_buffer is None:
            return

        if save:
            self._save_episode(self.frames_collected)
        else:
            print("\nEpisode discarded.")

        self.record_buffer = None
        self.frames_collected = 0
        self.state = RecorderState.IDLE
        self._drain_snapshot_queue()
        print("Ready. Press 's' to start a new episode.")

    def _capture_realsense_frames(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self._camera_cmd.send("capture")
        frames: dict[str, tuple[np.ndarray, np.ndarray]] = self._camera_reply.recv()
        color_frames: dict[str, np.ndarray] = {}
        depth_frames: dict[str, np.ndarray] = {}
        for camera_id in self.camera_ids:
            color, depth = frames[camera_id]
            color_frames[camera_id] = color
            depth_frames[camera_id] = depth
        return color_frames, depth_frames

    def _save_episode(self, frame_count: int) -> None:
        episode_path = self.output_dir / f"episode_{self.episode_index:04d}.h5"
        self.episode_index += 1
        buffer = self.record_buffer
        with h5py.File(episode_path, "w") as handle:
            handle.create_dataset("time_s", data=buffer["time"][:frame_count])
            joint_dataset = handle.create_dataset("joint_position", data=buffer["joints"][:frame_count])
            command_dataset = handle.create_dataset("command_position", data=buffer["commands"][:frame_count])
            joint_dataset.attrs["joint_keys"] = np.array(
                self.joint_keys, dtype=h5py.string_dtype("utf-8")
            )
            command_dataset.attrs["command_keys"] = np.array(
                self.command_keys, dtype=h5py.string_dtype("utf-8")
            )
            color_group = handle.create_group("realsense_color_bgr_uint8")
            depth_group = handle.create_group("realsense_depth_z16_uint16")
            for camera_id in self.camera_ids:
                color_dataset = color_group.create_dataset(
                    camera_id, data=buffer["color"][camera_id][:frame_count]
                )
                depth_dataset = depth_group.create_dataset(
                    camera_id, data=buffer["depth"][camera_id][:frame_count]
                )
                color_dataset.attrs["height"] = self.camera_shapes[camera_id][0]
                color_dataset.attrs["width"] = self.camera_shapes[camera_id][1]
                color_dataset.attrs["channels"] = 3
                color_dataset.attrs["fps"] = self.camera_fps
                color_dataset.attrs["fx"] = self.camera_fx[camera_id]
                color_dataset.attrs["fy"] = self.camera_fy[camera_id]
                color_dataset.attrs["ppx"] = self.camera_ppx[camera_id]
                color_dataset.attrs["ppy"] = self.camera_ppy[camera_id]
                coeffs = self.camera_coeffs[camera_id]
                if coeffs is not None:
                    color_dataset.attrs["intrinsics_coeffs"] = coeffs
                depth_dataset.attrs["height"] = self.camera_shapes[camera_id][0]
                depth_dataset.attrs["width"] = self.camera_shapes[camera_id][1]
                depth_dataset.attrs["units"] = "meters"
                depth_dataset.attrs["scale"] = self.camera_depth_scale[camera_id]
        print(f"\nSaved episode to {episode_path}", flush=True)

    def _drain_snapshot_queue(self) -> None:
        while True:
            try:
                self.snapshot_queue.get_nowait()
            except Empty:
                break

    def _cancel_recording(self) -> None:
        self.state = RecorderState.IDLE
        self.record_buffer = None
        self.frames_collected = 0
        self._drain_snapshot_queue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal joint state recorder.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Follower USB port (e.g. /dev/ttyACM0).")
    parser.add_argument("--control-fps", type=float, default=210.0, help="Control loop frequency (Hz).")
    parser.add_argument("--record-fps", type=float, default=10.0, help="Recording frequency (Hz).")
    parser.add_argument("--stream-fps", type=float, default=30.0, help="RealSense streaming frequency (Hz).")
    parser.add_argument("--episode-length", type=float, default=10.0, help="Episode length in seconds.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("/home/ripl/workspace/ProjectionPolicy/datasets/nov20").resolve()),
        help="Directory for recorded episodes.",
    )
    parser.add_argument("--follower-id", default="follower0", help="Follower calibration id.")
    parser.add_argument("--leader-id", default="leader0", help="Leader calibration id.")
    parser.add_argument("--teleop-port", default="/dev/ttyACM1", help="Leader USB port (e.g. /dev/ttyACM1).")
    parser.add_argument(
        "--realsense-serial",
        action="append",
        # D455 (213522251004) has a larger baseline / FOV; the others are D435 units.
        default=["817612070096", "817412070743", "213522251004"],
        help="RealSense serial number. Provide multiple times for multi-camera capture.",
    )
    parser.add_argument("--size",
        action="append",
        default=[[1280, 720], [1280, 720], [640, 480]],
        help="RealSense color/depth size. Provide multiple times for multi-camera capture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    control_rate = int(round(args.control_fps))
    record_rate = int(round(args.record_fps))
    stream_rate = int(round(args.stream_fps))
    sizes = [(int(size[0]), int(size[1])) for size in args.size]

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
            sizes=sizes,
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

