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

print("Calibration loaded.")
print(json.dumps(calib, indent=4))


# =====================================================
# HELPER FUNCTIONS
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
        math.atan2(-ax, math.sqrt(ay ** 2 + az ** 2))
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
# THRESHOLDS
# =====================================================

FREE_FALL_THRESHOLD = 5000      # ~0.3g
IMPACT_THRESHOLD = 30000

STILL_GYRO_THRESHOLD = 1000

SETTLING_TIME = 2               # seconds
OBSERVATION_TIME = 3            # seconds

STILL_RATIO_THRESHOLD = 0.5

# =====================================================
# FSM
# =====================================================

state = "NORMAL"

print("\nFall Detector V3 Started\n")

while True:

    ax, ay, az, gx, gy, gz = get_sensor_data()

    acc_mag = math.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

    pitch, roll = get_pitch_roll(ax, ay, az)

    orientation = get_orientation(pitch, roll)

    print(
        f"State={state:12s} "
        f"Acc={acc_mag:8.0f} "
        f"Gyro={gyro_mag:8.0f} "
        f"{orientation}"
    )

    # =================================================
    # NORMAL STATE
    # =================================================

    if state == "NORMAL":

        if acc_mag < FREE_FALL_THRESHOLD:

            print("\nPossible free fall detected")

            free_fall_time = time.time()

            state = "FREE_FALL"

    # =================================================
    # FREE FALL STATE
    # =================================================

    elif state == "FREE_FALL":

        # Free fall must be followed by impact
        # within 1 second

        if acc_mag > IMPACT_THRESHOLD:

            print("Impact detected!")

            state = "IMPACT"

        elif time.time() - free_fall_time > 1:

            print("False free-fall")

            state = "NORMAL"

    # =================================================
    # IMPACT STATE
    # =================================================

    elif state == "IMPACT":

        if orientation != "LEVEL":

            print("Abnormal posture detected")

            print(
                f"Allowing {SETTLING_TIME}s for settling..."
            )

            time.sleep(SETTLING_TIME)

            print(
                f"Monitoring for "
                f"{OBSERVATION_TIME}s..."
            )

            still_samples = 0
            total_samples = 0

            start = time.time()

            while time.time() - start < OBSERVATION_TIME:

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

            print(
                f"Still ratio = "
                f"{still_ratio:.2f}"
            )

            if still_ratio >= STILL_RATIO_THRESHOLD:

                print("\n")
                print("#" * 40)
                print("######## FALL DETECTED ########")
                print("#" * 40)
                print("\n")

                # Prevent repeated triggers
                time.sleep(5)

            else:

                print("False alarm")

        else:

            print("Returned to LEVEL quickly")

        state = "NORMAL"

    time.sleep(0.05)
