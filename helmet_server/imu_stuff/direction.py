from smbus2 import SMBus
import json
import time
import math

MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

bus = SMBus(1)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

with open("imu_calibration.json") as f:
    calib = json.load(f)


def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)

    value = (high << 8) | low

    if value >= 0x8000:
        value -= 65536

    return value


while True:

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

    # Determine orientation
    orientation = "LEVEL"

    # Upside down check first
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

    else:
        orientation = "LEVEL"

    print(
        f"Pitch = {pitch:6.2f}° | "
        f"Roll = {roll:6.2f}° | "
        f"Orientation = {orientation}"
    )

    time.sleep(0.1)
