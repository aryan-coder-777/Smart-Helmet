from smbus2 import SMBus
import json
import time
import math

# ====================================
# MPU SETTINGS
# ====================================

MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

bus = SMBus(1)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

# ====================================
# LOAD CALIBRATION
# ====================================

with open("imu_calibration.json", "r") as f:
    calib = json.load(f)


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


def get_sensor_data():

    ax = read_word(ACCEL_XOUT_H) - calib["ax_offset"]
    ay = read_word(ACCEL_XOUT_H + 2) - calib["ay_offset"]
    az = read_word(ACCEL_XOUT_H + 4) - calib["az_offset"]

    gx = read_word(GYRO_XOUT_H) - calib["gx_offset"]
    gy = read_word(GYRO_XOUT_H + 2) - calib["gy_offset"]
    gz = read_word(GYRO_XOUT_H + 4) - calib["gz_offset"]

    return ax, ay, az, gx, gy, gz


def get_orientation(ax, ay, az):

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

    return orientation, pitch, roll


# ====================================
# THRESHOLDS
# ====================================

# Impact threshold (~2.5g)
ACC_THRESHOLD = 25000

# Violent rotation threshold
GYRO_THRESHOLD = 500

# Motionless threshold
STILL_THRESHOLD = 150

MOTIONLESS_TIME = 3  # seconds


print("\nFall detection started...\n")

while True:

    ax, ay, az, gx, gy, gz = get_sensor_data()

    # Magnitudes
    acc_mag = math.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

    orientation, pitch, roll = get_orientation(ax, ay, az)

    print(
        f"Acc={acc_mag:8.0f} "
        f"Gyro={gyro_mag:8.0f} "
        f"{orientation}"
    )

    # =================================
    # POSSIBLE FALL
    # =================================

    if (acc_mag > ACC_THRESHOLD and
            gyro_mag > GYRO_THRESHOLD):

        print("\n!!! IMPACT DETECTED !!!")

        if orientation != "LEVEL":

            print("Abnormal orientation detected")
            print("Checking for motionlessness...")

            start = time.time()
            motionless = True

            while time.time() - start < MOTIONLESS_TIME:

                ax2, ay2, az2, gx2, gy2, gz2 = \
                    get_sensor_data()

                gyro_mag2 = math.sqrt(
                    gx2**2 + gy2**2 + gz2**2
                )

                if gyro_mag2 > STILL_THRESHOLD:
                    motionless = False
                    print("Movement detected.")
                    break

                time.sleep(0.1)

            if motionless:
                print("\n########################")
                print("### FALL DETECTED! ###")
                print("########################\n")

                # Prevent repeated triggers
                time.sleep(5)

    time.sleep(0.1)
