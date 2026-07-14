from smbus2 import SMBus
import json
import time
import math

# =====================================================
# MPU6500 SETUP
# =====================================================

MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

bus = SMBus(1)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

# =====================================================
# LOAD CALIBRATION
# =====================================================

with open("imu_calibration.json", "r") as f:
    calib = json.load(f)


# =====================================================
# HELPERS
# =====================================================

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


def get_pitch_roll(ax, ay, az):

    pitch = math.degrees(
        math.atan2(-ax, math.sqrt(ay**2 + az**2))
    )

    roll = math.degrees(
        math.atan2(ay, az)
    )

    return pitch, roll


def get_orientation(pitch, roll):

    if pitch > 30:
        return "UP"

    elif pitch < -30:
        return "DOWN"

    elif roll > 30:
        return "LEFT TILT"

    elif roll < -30:
        return "RIGHT TILT"

    else:
        return "LEVEL"


# =====================================================
# BMP280 HEIGHT FUNCTION
# Requires:
# pip install adafruit-circuitpython-bmp280
# =====================================================

import board
import adafruit_bmp280

i2c = board.I2C()
bmp = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x76)

# Set according to your local sea level pressure
bmp.sea_level_pressure = 1013.25


# =====================================================
# THRESHOLDS
# =====================================================

FREE_FALL_THRESHOLD = 8000      # ~0.5g
IMPACT_THRESHOLD = 28000        # ~1.7g

ORIENTATION_THRESHOLD = 30      # deg

HEIGHT_THRESHOLD = 0.25         # meters

STILL_GYRO_THRESHOLD = 250

STILL_RATIO = 0.80

POST_FALL_TIME = 5              # seconds


# =====================================================
# MAIN LOOP
# =====================================================

state = "NORMAL"

initial_height = bmp.altitude

print("\nFall Detector V2 Started\n")

while True:

    ax, ay, az, gx, gy, gz = get_sensor_data()

    acc_mag = math.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

    pitch, roll = get_pitch_roll(ax, ay, az)

    orientation = get_orientation(pitch, roll)

    current_height = bmp.altitude

    print(
        f"State={state:12s} "
        f"Acc={acc_mag:8.0f} "
        f"Gyro={gyro_mag:7.0f} "
        f"H={current_height:6.2f}m "
        f"{orientation}"
    )

    # =================================================
    # STATE MACHINE
    # =================================================

    if state == "NORMAL":

        if acc_mag < FREE_FALL_THRESHOLD:
            print("\nPossible free fall detected")
            free_fall_height = current_height
            state = "FREE_FALL"


    elif state == "FREE_FALL":

        if acc_mag > IMPACT_THRESHOLD:
            print("Impact detected!")
            impact_height = current_height
            state = "IMPACT"

        elif acc_mag > FREE_FALL_THRESHOLD:
            # False alarm
            state = "NORMAL"


    elif state == "IMPACT":

        if orientation != "LEVEL":

            print("Abnormal posture detected")
            time.sleep(0.8)
            print("Monitoring post-fall state...")
            still_samples = 0
            total_samples = 0

            start = time.time()

            while time.time() - start < POST_FALL_TIME:

                ax2, ay2, az2, gx2, gy2, gz2 = \
                    get_sensor_data()

                gyro_mag2 = math.sqrt(
                    gx2**2 +
                    gy2**2 +
                    gz2**2
                )

                if gyro_mag2 < STILL_GYRO_THRESHOLD:
                    still_samples += 1

                total_samples += 1

                time.sleep(0.1)

            still_ratio = still_samples / total_samples

            height_drop = abs(
                free_fall_height - impact_height
            )

            print(
                f"Still ratio = {still_ratio:.2f}"
            )

            print(
                f"Height change = {height_drop:.2f} m"
            )

            if (still_ratio > STILL_RATIO ):

                print("\n################################")
                print("######## FALL DETECTED #########")
                print("################################\n")

                time.sleep(5)

            else:
                print("False alarm.")

        state = "NORMAL"

    time.sleep(0.05)
