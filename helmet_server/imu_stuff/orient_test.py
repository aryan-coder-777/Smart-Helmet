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

    # Add gravity back because we removed it during calibration
    az = read_word(ACCEL_XOUT_H + 4) - calib["az_offset"]

    roll = math.degrees(math.atan2(ay, az))

    pitch = math.degrees(
        math.atan2(
            -ax,
            math.sqrt(ay**2 + az**2)
        )
    )

    print(
        f"Pitch = {pitch:6.2f}°    "
        f"Roll = {roll:6.2f}°"
    )

    time.sleep(0.1)
