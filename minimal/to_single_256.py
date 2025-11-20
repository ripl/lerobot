import argparse
from pathlib import Path
import glob
import h5py
import numpy as np
import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge RealSense episodes to a single HDF5 with 256x256 images.")
    parser.add_argument(
        "--input-dir",
        default=str(Path("/home/ripl/workspace/ProjectionPolicy/datasets/nov16_50").resolve()),
        help="Directory containing episode_*.h5 files.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("/home/ripl/workspace/ProjectionPolicy/datasets/nov16_50_256.hdf5").resolve()),
        help="Output HDF5 path.",
    )
    parser.add_argument("--size", type=int, default=256, help="Output image size (default: 256).")
    return parser.parse_args()


def resize_rgb(frames_bgr: np.ndarray, size: int) -> np.ndarray:
    t = int(frames_bgr.shape[0])
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    for i in range(t):
        bgr = frames_bgr[i]
        small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        out[i] = rgb
    return out


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = int(args.size)

    episode_paths = sorted(glob.glob(str(in_dir / "episode_*.h5")))
    assert len(episode_paths) > 0, f"No episode_*.h5 files found in {in_dir}"

    with h5py.File(str(out_path), "w") as out_f:
        for epi_idx, epi_fp in enumerate(tqdm(episode_paths, desc="episodes")):
            with h5py.File(str(Path(epi_fp).resolve()), "r") as ep:
                joints = ep["joint_position"][:].astype(np.float32)
                actions = ep["command_position"][:].astype(np.float32)
                color_group = ep["realsense_color_bgr_uint8"]
                cam_ids = sorted(list(color_group.keys()))

            demo = out_f.create_group(f"demo_{epi_idx}")
            demo.create_dataset("joints", data=joints, compression=None)
            demo.create_dataset("actions", data=actions, compression=None)
            obs = demo.create_group("obs")
            for cam_id in tqdm(cam_ids, desc="cameras", leave=False):
                with h5py.File(str(Path(epi_fp).resolve()), "r") as ep2:
                    frames_bgr = ep2["realsense_color_bgr_uint8"][cam_id][:]
                rgb = resize_rgb(frames_bgr, size)
                obs.create_dataset(f"{cam_id}_image", data=rgb, compression=None)


if __name__ == "__main__":
    main()


