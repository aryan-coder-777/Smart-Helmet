from smbus2 import SMBus
import json
import time
import math
from collections import deque

# =========================
# MPU6500 SETTINGS
# =========================

MPU_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

bus = SMBus(1)

# Wake MPU
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

# =========================
# LOAD CALIBRATION
# =========================

with open("imu_calibration.json", "r") as f:
    calib = json.load(f)

print("Calibration loaded.")
print(json.dumps(calib, indent=4))


# =========================
# HELPER FUNCTIONS
# =========================

def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)

    value = (high << 8) | low

    if value >= 0x8000:
        value -= 65536

    return value


def get_orientation():

    ax = read_word(ACCEL_XOUT_H) - calib["ax_offset"]
    ay = read_word(ACCEL_XOUT_H + 2) - calib["ay_offset"]
    az = read_word(ACCEL_XOUT_H + 4) - calib["az_offset"]

    pitch = math.degrees(
        math.atan2(
            -ax,
            math.sqrt(ay**2 + az**2)
        )
    )

    roll = math.degrees(
        math.atan2(ay, az)
    )

    orientation = "LEVEL"

    # NOTE:
    # Swap UP/DOWN if required.

    if az < 0:
        orientation = "UPSIDE-DOWN"

    elif pitch > 30:
        orientation = "UP"

    elif pitch < -30:
        orientation = "DOWN"

    elif roll > 30:
        orientation = "LEFT TILT"

    elif roll < -30:
        orientation = "RIGHT TILT"

    return pitch, roll, orientation


# =========================
# GESTURE DETECTION
# =========================

history = deque(maxlen=3)

last_orientation = "LEVEL"

print("\nReady for gestures!\n")

while True:

    pitch, roll, orientation = get_orientation()

    print(
        f"Pitch={pitch:6.1f}° "
        f"Roll={roll:6.1f}° "
        f"| {orientation}"
    )

    # Record only meaningful changes
    if (orientation != last_orientation and
            orientation != "LEVEL"):

        history.append(orientation)

        print("History:", list(history))

        # -----------------
        # NOD DETECTION
        # -----------------

        if list(history)[-2:] == ["DOWN", "UP"]:
            print("\n>>> NOD DETECTED <<<\n")
            history.clear()

        elif list(history)[-2:] == ["UP", "DOWN"]:
            print("\n>>> REVERSE NOD DETECTED <<<\n")
            history.clear()

        # -----------------
        # SHAKE DETECTION
        # -----------------

        elif list(history)[-2:] == [
            "LEFT TILT",
            "RIGHT TILT"
        ]:

            print("\n>>> HEAD SHAKE DETECTED <<<\n")
            history.clear()

        elif list(history)[-2:] == [
            "RIGHT TILT",
            "LEFT TILT"
        ]:

            print("\n>>> HEAD SHAKE DETECTED <<<\n")
            history.clear()

        last_orientation = orientation

    # Reset once helmet returns to LEVEL
    if orientation == "LEVEL":
        last_orientation = "LEVEL"

    time.sleep(0.1)
