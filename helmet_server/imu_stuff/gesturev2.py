from smbus2 import SMBus
import json
import time
import math
from collections import deque

# ====================================
# MPU6500 SETTINGS
# ====================================

MPU_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

bus = SMBus(1)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

# ====================================
# LOAD CALIBRATION
# ====================================

with open("imu_calibration.json", "r") as f:
    calib = json.load(f)

print("Calibration loaded.")
print(json.dumps(calib, indent=4))


# ====================================
# HELPER FUNCTIONS
# ====================================

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


# ====================================
# GESTURE SETTINGS
# ====================================

history = deque(maxlen=3)

last_orientation = "LEVEL"

gesture_start = None

GESTURE_TIMEOUT = 1.5      # seconds
LEVEL_RESET_TIME = 2.0     # seconds

level_start = None

print("\nReady for gestures!\n")

# ====================================
# MAIN LOOP
# ====================================

while True:

    pitch, roll, orientation = get_orientation()

    print(
        f"Pitch={pitch:6.1f}° "
        f"Roll={roll:6.1f}° "
        f"| {orientation}"
    )

    # --------------------------------
    # LEVEL HOLD DETECTION
    # --------------------------------

    if orientation == "LEVEL":

        if level_start is None:
            level_start = time.time()

        elif time.time() - level_start > LEVEL_RESET_TIME:

            if len(history) > 0:
                print("\nLEVEL HELD -> Clearing history\n")

            history.clear()
            gesture_start = None

    else:
        level_start = None

    # --------------------------------
    # RECORD NEW ORIENTATION
    # --------------------------------

    if (orientation != "LEVEL" and
            orientation != last_orientation):

        history.append(orientation)

        print("History:", list(history))

        last_orientation = orientation

        # First movement in gesture
        if len(history) == 1:
            gesture_start = time.time()

    # Allow future gestures after returning to level
    if orientation == "LEVEL":
        last_orientation = "LEVEL"

    # --------------------------------
    # GESTURE TIMEOUT
    # --------------------------------

    if gesture_start is not None:

        if time.time() - gesture_start > GESTURE_TIMEOUT:

            print("\nGesture timeout -> Clearing history\n")

            history.clear()
            gesture_start = None

    # --------------------------------
    # GESTURE RECOGNITION
    # --------------------------------

    h = list(history)

    if h[-2:] == ["DOWN", "UP"]:
        print("\n>>> NOD DETECTED <<<\n")

        history.clear()
        gesture_start = None

    elif h[-2:] == ["UP", "DOWN"]:
        print("\n>>> REVERSE NOD DETECTED <<<\n")

        history.clear()
        gesture_start = None

    elif h[-2:] == ["LEFT TILT", "RIGHT TILT"]:
        print("\n>>> HEAD SHAKE DETECTED <<<\n")

        history.clear()
        gesture_start = None

    elif h[-2:] == ["RIGHT TILT", "LEFT TILT"]:
        print("\n>>> HEAD SHAKE DETECTED <<<\n")

        history.clear()
        gesture_start = None

    time.sleep(0.1)
