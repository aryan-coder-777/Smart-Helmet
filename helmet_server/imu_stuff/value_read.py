from smbus2 import SMBus
import time

MPU_ADDR = 0x68

# Registers
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

    value = (high << 8) + low

    if value >= 0x8000:
        value = -((65535 - value) + 1)

    return value


while True:

    ax = read_word(ACCEL_XOUT_H)
    ay = read_word(ACCEL_XOUT_H + 2)
    az = read_word(ACCEL_XOUT_H + 4)

    gx = read_word(GYRO_XOUT_H)
    gy = read_word(GYRO_XOUT_H + 2)
    gz = read_word(GYRO_XOUT_H + 4)

    print(
        f"Accel: X={ax:6d} "
        f"Y={ay:6d} "
        f"Z={az:6d}"
    )

    print(
        f"Gyro : X={gx:6d} "
        f"Y={gy:6d} "
        f"Z={gz:6d}"
    )

    print("-" * 60)

    time.sleep(0.2)
