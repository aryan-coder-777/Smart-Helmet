from smbus2 import SMBus
import time
import json

MPU_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

bus = SMBus(1)

# Wake up MPU6500
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
time.sleep(0.1)


def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)

    value = (high << 8) | low

    if value >= 0x8000:
        value -= 65536

    return value


print("=" * 60)
print("IMU CALIBRATION")
print("Keep the helmet perfectly still!")
print("Starting in 5 seconds...")
print("=" * 60)

time.sleep(5)

N = 500

ax_sum = ay_sum = az_sum = 0
gx_sum = gy_sum = gz_sum = 0

for i in range(N):

    ax_sum += read_word(ACCEL_XOUT_H)
    ay_sum += read_word(ACCEL_XOUT_H + 2)
    az_sum += read_word(ACCEL_XOUT_H + 4)

    gx_sum += read_word(GYRO_XOUT_H)
    gy_sum += read_word(GYRO_XOUT_H + 2)
    gz_sum += read_word(GYRO_XOUT_H + 4)

    if i % 50 == 0:
        print(f"Collected {i}/{N} samples")

    time.sleep(0.01)

calibration = {
    "ax_offset": ax_sum / N,
    "ay_offset": ay_sum / N,

    # Remove gravity from Z
    "az_offset": (az_sum / N) - 16384,

    "gx_offset": gx_sum / N,
    "gy_offset": gy_sum / N,
    "gz_offset": gz_sum / N
}

with open("imu_calibration.json", "w") as f:
    json.dump(calibration, f, indent=4)

print("\nCalibration complete!")
print(json.dumps(calibration, indent=4))
print("\nSaved to imu_calibration.json")
