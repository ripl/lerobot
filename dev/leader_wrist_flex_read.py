from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
import time
from lerobot.motors.feetech import OperatingMode

def main() -> None:
    cfg = SO101LeaderConfig(
        port="/dev/ttyACM0",
        id="leader0",
    )

    leader = SO101Leader(cfg)
    leader.connect(calibrate=False)
    leader.bus.write("Operating_Mode", "wrist_flex", OperatingMode.POSITION.value)
    while True:
        pos = leader.bus.read("Present_Position", "wrist_flex", normalize=False)
        print(pos)
        time.sleep(0.1)

    leader.disconnect()


if __name__ == "__main__":
    main()


