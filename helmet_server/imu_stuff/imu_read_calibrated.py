from smbus2 import SMBus
import time
import json

MPU_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# -------------------------------
# Load calibration
# -------------------------------
with open("imu_calibration.json", "r") as f:
    calib = json.load(f)

print("Loaded calibration:")
print(json.dumps(calib, indent=4))
print()

# -------------------------------
# Initialize MPU6500
# -------------------------------
bus = SMBus(1)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

time.sleep(0.1)


def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)

    value = (high << 8) | low

    if value >= 0x8000:
        value -= 65536

    return value


while True:

    # Raw readings
    ax = read_word(ACCEL_XOUT_H)
    ay = read_word(ACCEL_XOUT_H + 2)
    az = read_word(ACCEL_XOUT_H + 4)

    gx = read_word(GYRO_XOUT_H)
    gy = read_word(GYRO_XOUT_H + 2)
    gz = read_word(GYRO_XOUT_H + 4)

    # Apply calibration
    ax -= calib["ax_offset"]
    ay -= calib["ay_offset"]
    az -= calib["az_offset"]

    gx -= calib["gx_offset"]
    gy -= calib["gy_offset"]
    gz -= calib["gz_offset"]

    print(
        f"Accel: X={ax:8.2f} "
        f"Y={ay:8.2f} "
        f"Z={az:8.2f}"
    )

    print(
        f"Gyro : X={gx:8.2f} "
        f"Y={gy:8.2f} "
        f"Z={gz:8.2f}"
    )

    print("-" * 60)

    time.sleep(0.2)
