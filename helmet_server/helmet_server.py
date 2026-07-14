from flask import Flask, jsonify, Response, send_from_directory
import subprocess
import time
import os
import signal
import struct
from datetime import datetime
import smbus2 as smbus
from threading import Thread, Lock

app = Flask(__name__)

# Dynamic media directory based on script location (handles any user accounts)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media")

IMU_ADDR = 0x68
BMP_ADDR = 0x76
MAG_ADDR = 0x0C
i2c_lock = Lock()

# Find an available I2C bus (4 for Radxa Pin 27/28, 3 for Pin 3/5, 1 for Pi, 6/0 for fallbacks)
bus = None
for port in [4, 3, 1, 6, 0]:
    try:
        bus = smbus.SMBus(port)
        print(f"Successfully initialized I2C bus {port}")
        break
    except Exception:
        bus = None

if bus is None:
    print("[WARNING] No working I2C bus found. IMU sensor is disabled.")

record_process = None
record_file = None

imu_record_thread = None
imu_record_running = False
imu_record_file = None

# Gyroscope calibration offsets (will be updated dynamically if calibration succeeds)
gyro_offset_x = 203
gyro_offset_y = 143
gyro_offset_z = -45
calibration_status = "pending"

# Track whether the IMU has been initialized so /status does not re-run it every poll
imu_initialized = False
has_magnetometer = False

# Global camera stream frame
latest_frame = None
frame_id = 0

# BMP280 Calibration Data
dig_T1 = 0; dig_T2 = 0; dig_T3 = 0
dig_P1 = 0; dig_P2 = 0; dig_P3 = 0; dig_P4 = 0; dig_P5 = 0; dig_P6 = 0; dig_P7 = 0; dig_P8 = 0; dig_P9 = 0

# ── Posture Engine State ─────────────────────────────────────────────────────
import math
from collections import deque

posture_lock = Lock()
posture_state = {
    "orientation": "UNKNOWN",
    "pitch": 0.0,
    "roll": 0.0,
    "turn_direction": "STRAIGHT",     # STRAIGHT | LEFT TURN | RIGHT TURN
    "fall_state": "NORMAL",           # NORMAL | FREE_FALL | IMPACT | FALL_DETECTED
    "last_gesture": "None",           # NOD | REVERSE_NOD | HEAD_SHAKE | None
    "altitude_m": 0.0,
    "altitude_delta_m": 0.0,
    "acc_magnitude": 0.0,
    "yaw_rate_dps": 0.0,              # calibrated yaw rate in degrees/second
}

# Fall detection thresholds (raw ADC counts from MPU6500 at ±2g, 16384 LSB/g)
_FREE_FALL_THRESHOLD = 4000      # ~0.24g  — nearly weightless during free fall
_IMPACT_THRESHOLD    = 22000     # ~1.3g   — lowered: cushion impacts are softer
_STILL_GYRO_THRESH   = 5000      # ~38 deg/s — raised: tolerate post-bounce vibration
_SETTLING_TIME       = 2.0       # seconds to wait for bounces to die down
_OBSERVE_TIME        = 2.5       # seconds to judge stillness

# Turn direction via gyro Z integration
# We accumulate yaw angle and classify based on that,
# so the display HOLDS the direction after turning (not rate-based).
_YAW_TURN_ANGLE   = 30.0   # degrees past centre = LEFT/RIGHT
_YAW_STILL_THRESH = 400    # raw gz below this = head is stationary (~3 deg/s)
_YAW_DRIFT_DECAY  = 0.998  # per 50ms tick: slowly drift back when still (~0.2%/tick)

# Gesture detection
_gesture_history   = deque(maxlen=3)
_last_orientation  = "LEVEL"
_gesture_start     = None
_GESTURE_TIMEOUT   = 1.5
_level_start       = None
_LEVEL_RESET_TIME  = 2.0

# Altitude baseline (set on first good reading)
_alt_baseline      = None


def init_bmp280():
    global dig_T1, dig_T2, dig_T3, dig_P1, dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9
    if bus is None:
        return False
    try:
        chip_id = bus.read_byte_data(BMP_ADDR, 0xD0)
        if chip_id != 0x58:
            return False
        
        calib = bus.read_i2c_block_data(BMP_ADDR, 0x88, 24)
        
        dig_T1 = struct.unpack_from("<H", bytearray(calib), 0)[0]
        dig_T2 = struct.unpack_from("<h", bytearray(calib), 2)[0]
        dig_T3 = struct.unpack_from("<h", bytearray(calib), 4)[0]
        
        dig_P1 = struct.unpack_from("<H", bytearray(calib), 6)[0]
        dig_P2 = struct.unpack_from("<h", bytearray(calib), 8)[0]
        dig_P3 = struct.unpack_from("<h", bytearray(calib), 10)[0]
        dig_P4 = struct.unpack_from("<h", bytearray(calib), 12)[0]
        dig_P5 = struct.unpack_from("<h", bytearray(calib), 14)[0]
        dig_P6 = struct.unpack_from("<h", bytearray(calib), 16)[0]
        dig_P7 = struct.unpack_from("<h", bytearray(calib), 18)[0]
        dig_P8 = struct.unpack_from("<h", bytearray(calib), 20)[0]
        dig_P9 = struct.unpack_from("<h", bytearray(calib), 22)[0]
        
        # ctrl_meas (0xF4): normal mode, temp x1, press x16
        bus.write_byte_data(BMP_ADDR, 0xF4, 0x3F)
        # config (0xF5): standby 0.5ms, filter 16
        bus.write_byte_data(BMP_ADDR, 0xF5, 0x10)
        return True
    except Exception as e:
        print(f"[BMP280] Init failed: {e}")
        return False


def read_bmp280():
    if bus is None:
        return 0.0, 0.0
    try:
        data = bus.read_i2c_block_data(BMP_ADDR, 0xF7, 6)
        adc_P = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        adc_T = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
        
        # Temp Compensation
        var1 = ((((adc_T >> 3) - (dig_T1 << 1))) * dig_T2) >> 11
        var2 = (((((adc_T >> 4) - dig_T1) * ((adc_T >> 4) - dig_T1)) >> 12) * dig_T3) >> 14
        t_fine = var1 + var2
        temperature = (t_fine * 5 + 128) >> 8
        temp_c = temperature / 100.0
        
        # Pressure Compensation
        var1 = (t_fine >> 1) - 64000
        var2 = (((var1 >> 2) * (var1 >> 2)) >> 11) * dig_P6
        var2 = var2 + ((var1 * dig_P5) << 1)
        var2 = (var2 >> 2) + (dig_P4 << 16)
        var1 = (((dig_P3 * (((var1 >> 2) * (var1 >> 2)) >> 13)) >> 3) + (((dig_P2 * var1) >> 1)) >> 18)
        var1 = ((32768 + var1) * dig_P1) >> 15
        
        if var1 == 0:
            return temp_c, 0.0
            
        pressure = (((1048576 - adc_P) - (var2 >> 12))) * 3125
        if pressure < 0x80000000:
            pressure = (pressure << 1) // var1
        else:
            pressure = (pressure // var1) * 2
            
        var1 = (dig_P9 * (((pressure >> 3) * (pressure >> 3)) >> 13)) >> 12
        var2 = (((pressure >> 2)) * dig_P8) >> 13
        pressure = pressure + ((var1 + var2 + dig_P7) >> 4)
        
        # Altitude calculation
        altitude = 44330.0 * (1.0 - (pressure / 101325.0) ** (1.0 / 5.255))
        return temp_c, altitude
    except Exception as e:
        print(f"[BMP280] Read failed: {e}")
        return 0.0, 0.0


def init_ak8963():
    """NOTE: Bypass mode (reg 0x37) must already be enabled by init_imu() before calling this."""
    if bus is None:
        return False
    try:
        # Verify WhoAmI of magnetometer (bypass must already be open)
        mag_id = bus.read_byte_data(MAG_ADDR, 0x00)
        print(f"[AK8963] WHO_AM_I = {mag_id:#x}")
        if mag_id != 0x48:
            print(f"[AK8963] Unexpected WHO_AM_I: {mag_id:#x} (expected 0x48)")
            return False
        
        # Power Down mode
        bus.write_byte_data(MAG_ADDR, 0x0A, 0x00)
        time.sleep(0.01)
        
        # Set 16-bit output, continuous measurement mode 2 (100Hz)
        bus.write_byte_data(MAG_ADDR, 0x0A, 0x16)
        time.sleep(0.01)
        return True
    except Exception as e:
        print(f"[AK8963] Init failed: {e}")
        return False


def read_ak8963():
    if bus is None or not has_magnetometer:
        return {"x_ut": 0.0, "y_ut": 0.0, "z_ut": 0.0}
    try:
        data = bus.read_i2c_block_data(MAG_ADDR, 0x03, 7)
        st2 = data[6]
        if (st2 & 0x08):
            return {"x_ut": 0.0, "y_ut": 0.0, "z_ut": 0.0}
            
        mx = struct.unpack_from("<h", bytearray(data), 0)[0]
        my = struct.unpack_from("<h", bytearray(data), 2)[0]
        mz = struct.unpack_from("<h", bytearray(data), 4)[0]
        
        scale = 0.15
        return {
            "x_ut": round(mx * scale, 2),
            "y_ut": round(my * scale, 2),
            "z_ut": round(mz * scale, 2)
        }
    except Exception as e:
        print(f"[AK8963] Read failed: {e}")
        return {"x_ut": 0.0, "y_ut": 0.0, "z_ut": 0.0}


def init_imu():
    global imu_initialized
    if bus is None:
        return False
    # Only initialize once — calling this on every /status poll destroys bypass mode
    if imu_initialized:
        return True
    with i2c_lock:
        try:
            # Step 1: Hard reset the MPU9250
            bus.write_byte_data(IMU_ADDR, 0x6B, 0x80)  # Device reset
            time.sleep(0.1)

            # Step 2: Wake up, select best clock source
            bus.write_byte_data(IMU_ADDR, 0x6B, 0x01)
            time.sleep(0.1)

            # Step 3: Read and log WHO_AM_I
            whoami = bus.read_byte_data(IMU_ADDR, 0x75)
            print(f"[IMU] WHO_AM_I = {whoami:#x}")
            if whoami not in [0x71, 0x73, 0x70]:
                print(f"[IMU] Unexpected chip ID {whoami:#x} — sensor may not be connected")

            # Step 4: Disable I2C Master mode (CRITICAL — must be 0 before bypass)
            bus.write_byte_data(IMU_ADDR, 0x6A, 0x00)
            time.sleep(0.05)

            # Step 5: Enable I2C bypass so host can talk directly to AK8963
            bus.write_byte_data(IMU_ADDR, 0x37, 0x02)
            time.sleep(0.05)  # Give AK8963 time to appear on the bus

            # Step 6: Try to init AK8963 (bypass already enabled above)
            global has_magnetometer
            has_magnetometer = init_ak8963()
            if not has_magnetometer:
                print(f"[IMU] WHO_AM_I={whoami:#x}: Magnetometer not found at 0x0C. Module may be MPU6500 clone.")

            # Step 7: Init BMP280
            init_bmp280()

            imu_initialized = True
            return True
        except Exception as e:
            print(f"[IMU] Init failed with exception: {e}")
            return False


def read_word(reg):
    if bus is None:
        return 0
    with i2c_lock:
        try:
            high = bus.read_byte_data(IMU_ADDR, reg)
            low = bus.read_byte_data(IMU_ADDR, reg + 1)
            value = (high << 8) | low
            if value >= 32768:
                value -= 65536
            return value
        except Exception:
            return 0


def calibrate_gyro():
    global gyro_offset_x, gyro_offset_y, gyro_offset_z, calibration_status
    if bus is None:
        calibration_status = "not_available"
        return

    calibration_status = "calibrating"
    try:
        # Settle sensor (throw away initial samples)
        for _ in range(500):
            read_word(0x43)
            read_word(0x45)
            read_word(0x47)
            time.sleep(0.002)

        gx_sum = 0
        gy_sum = 0
        gz_sum = 0
        samples = 2000

        for _ in range(samples):
            gx_sum += read_word(0x43)
            gy_sum += read_word(0x45)
            gz_sum += read_word(0x47)
            time.sleep(0.002)

        gyro_offset_x = gx_sum / samples
        gyro_offset_y = gy_sum / samples
        gyro_offset_z = gz_sum / samples
        calibration_status = "done"
        print(f"Calibration done. Offsets: GX={gyro_offset_x:.2f}, GY={gyro_offset_y:.2f}, GZ={gyro_offset_z:.2f}")
    except Exception as e:
        calibration_status = f"failed: {e}"
        print(f"Calibration failed: {e}")


def read_imu():
    if bus is None:
        return {
            "accelerometer": {"x_g": 0.0, "y_g": 0.0, "z_g": 0.0},
            "gyroscope": {"x_dps": 0.0, "y_dps": 0.0, "z_dps": 0.0},
            "magnetometer": {"x_ut": 0.0, "y_ut": 0.0, "z_ut": 0.0},
            "altitude_m": 0.0,
            "temperature_c": 0.0,
            "timestamp": time.time()
        }

    accel_x = read_word(0x3B)
    accel_y = read_word(0x3D)
    accel_z = read_word(0x3F)
    temp_raw = read_word(0x41)

    # Apply calibration offsets
    gyro_x = read_word(0x43) - gyro_offset_x
    gyro_y = read_word(0x45) - gyro_offset_y
    gyro_z = read_word(0x47) - gyro_offset_z

    # Read magnetometer
    mag = {"x_ut": 0.0, "y_ut": 0.0, "z_ut": 0.0}
    with i2c_lock:
        try:
            mag = read_ak8963()
        except Exception:
            pass

    # Read BMP280 altitude
    bmp_temp = 0.0
    altitude = 0.0
    with i2c_lock:
        try:
            bmp_temp, altitude = read_bmp280()
        except Exception:
            pass

    temp_c = round((temp_raw / 333.87) + 21.0, 2)

    return {
        "accelerometer": {
            "x_g": round(accel_x / 16384.0, 3),
            "y_g": round(accel_y / 16384.0, 3),
            "z_g": round(accel_z / 16384.0, 3),
        },
        "gyroscope": {
            "x_dps": round(gyro_x / 131.0, 3),
            "y_dps": round(gyro_y / 131.0, 3),
            "z_dps": round(gyro_z / 131.0, 3),
        },
        "magnetometer": mag,
        "altitude_m": round(altitude, 2),
        "temperature_c": temp_c,
        "timestamp": time.time()
    }


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "device": "smart-helmet",
        "camera": "available",
        "imu": "available" if imu_initialized else "not_detected",  # Fixed: do NOT call init_imu() here
        "recording": record_process is not None,
        "record_file": record_file,
        "imu_recording": imu_record_running,
        "imu_record_file": imu_record_file,
        "gyro_calibration": {
            "status": calibration_status,
            "offsets": {
                "x": round(gyro_offset_x, 2),
                "y": round(gyro_offset_y, 2),
                "z": round(gyro_offset_z, 2)
            }
        }
    })


@app.route("/imu", methods=["GET"])
def imu():
    try:
        data = read_imu()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/imu/record/start", methods=["POST", "GET"])
def start_imu_recording():
    global imu_record_thread, imu_record_running, imu_record_file
    if imu_record_running:
        return jsonify({"success": False, "message": "IMU recording already active", "file": imu_record_file}), 400

    filename = datetime.now().strftime("imu_%Y%m%d_%H%M%S.csv")
    imu_record_file = os.path.join(MEDIA_DIR, filename)

    imu_record_running = True
    def log_worker():
        try:
            with open(imu_record_file, "w") as f:
                f.write("timestamp,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps,temp_c\n")
                while imu_record_running:
                    if init_imu():
                        try:
                            data = read_imu()
                            f.write(f"{data['timestamp']},{data['accelerometer']['x_g']},{data['accelerometer']['y_g']},{data['accelerometer']['z_g']},{data['gyroscope']['x_dps']},{data['gyroscope']['y_dps']},{data['gyroscope']['z_dps']},{data['temperature_c']}\n")
                            f.flush()
                        except Exception as e:
                            print(f"IMU read error inside worker: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"IMU logging worker failed: {e}")

    imu_record_thread = Thread(target=log_worker, daemon=True)
    imu_record_thread.start()
    return jsonify({"success": True, "message": "IMU recording started", "file": imu_record_file})


@app.route("/imu/record/stop", methods=["POST", "GET"])
def stop_imu_recording():
    global imu_record_running, imu_record_file
    if not imu_record_running:
        return jsonify({"success": False, "message": "IMU recording not active"}), 400

    imu_record_running = False
    finished_file = imu_record_file
    imu_record_file = None
    return jsonify({"success": True, "message": "IMU recording stopped", "file": finished_file})


@app.route("/media/list", methods=["GET"])
def list_media():
    try:
        files = []
        if os.path.exists(MEDIA_DIR):
            for f in sorted(os.listdir(MEDIA_DIR), reverse=True):
                if f.endswith(".csv") or f.endswith(".mp4") or f.endswith(".h264") or f.endswith(".jpg"):
                    file_path = os.path.join(MEDIA_DIR, f)
                    files.append({
                        "name": f,
                        "size": os.path.getsize(file_path),
                        "timestamp": os.path.getmtime(file_path)
                    })
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/media/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory(MEDIA_DIR, filename, as_attachment=True)


# ── Camera System ──────────────────────────────────────────────────────────

# Global recording and lock variables
recording_file_handle = None
recording_lock = Lock()
camera_lock = Lock()
latest_frame = None

# On-demand streaming state
active_clients = 0
active_clients_lock = Lock()
camera_thread = None
camera_running = False
preferred_camera_device = os.environ.get("SMARTHELMET_CAMERA_DEVICE", "")
stream_fps = float(os.environ.get("SMARTHELMET_STREAM_FPS", "20"))

def start_camera_thread():
    global camera_thread, camera_running
    with active_clients_lock:
        if not camera_running:
            camera_running = True
            camera_thread = Thread(target=camera_worker, daemon=True)
            camera_thread.start()
            print("[Camera] Background worker started.", flush=True)

def stop_camera_thread():
    global camera_running
    with active_clients_lock:
        if camera_running:
            camera_running = False
            print("[Camera] Background worker stopping...", flush=True)

def camera_worker():
    global latest_frame, frame_id, camera_running, recording_file_handle
    import fcntl
    import os
    
    with camera_lock:
        latest_frame = None
        frame_id = 0

    print("[CameraWorker] Starting background camera thread...", flush=True)

    def configure_usb_camera(dev):
        controls = [
            ["v4l2-ctl", "--device", dev, "-c", "exposure_dynamic_framerate=0"],
            ["v4l2-ctl", "--device", dev, "-c", "auto_exposure=1"],
            ["v4l2-ctl", "--device", dev, "-c", "exposure_time_absolute=156"],
            ["v4l2-ctl", "--device", dev, "-c", "gain=128"],
            ["v4l2-ctl", "--device", dev, "-c", "brightness=150"],
            ["v4l2-ctl", "--device", dev, "-c", "power_line_frequency=1"],
        ]
        for cmd in controls:
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            except Exception:
                pass

    def find_camera_devices():
        """Scan /dev/video0..7 and return capture-capable devices, MJPEG first."""
        candidates = []

        # If user pinned a device via env var, try it first
        if preferred_camera_device and os.path.exists(preferred_camera_device):
            candidates.append((-1, preferred_camera_device))

        for i in range(8):
            dev = f"/dev/video{i}"
            if not os.path.exists(dev):
                continue
            if dev == preferred_camera_device:
                continue
            try:
                res = subprocess.run(
                    ["v4l2-ctl", "--device", dev, "--list-formats-ext"],
                    capture_output=True, text=True, timeout=2
                )
                output = res.stdout + res.stderr

                # Skip metadata / output-only nodes
                if "No such" in output or "Failed" in output:
                    continue
                # Skip multiplanar (ISP nodes on Radxa)
                if "Video Capture Multiplanar" in output:
                    continue
                # Must advertise capture formats
                if "Pixel Format" not in output and "MJPG" not in output:
                    continue

                # Score: 0 = MJPEG (best), 1 = other format
                score = 0 if ("MJPG" in output or "Motion-JPEG" in output) else 1
                candidates.append((score, dev))
                print(f"[CameraWorker] Found candidate {dev} score={score}", flush=True)
            except Exception:
                pass

        candidates.sort()
        return [dev for _, dev in candidates]

    while camera_running:
        devices = find_camera_devices()
        if not devices:
            print("[CameraWorker] No USB camera device found, retrying in 2s...", flush=True)
            time.sleep(2)
            continue

        for dev in devices:
            if not camera_running:
                break

            configure_usb_camera(dev)

            strategies = [
                (
                    "v4l2-mjpeg-640x480-30",
                    [
                        "v4l2-ctl", "--device", dev,
                        "--set-fmt-video=width=640,height=480,pixelformat=MJPG",
                        "--set-parm=30",
                        "--stream-mmap=4",
                        "--stream-to=-",
                        "--stream-count=999999"
                    ]
                ),
                (
                    "mjpeg-copy-640x480-30",
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "v4l2", "-input_format", "mjpeg",
                        "-video_size", "640x480", "-framerate", "30", "-i", dev,
                        "-an", "-vcodec", "copy", "-f", "mjpeg", "pipe:1"
                    ]
                ),
                (
                    "mjpeg-copy-320x240-30",
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "v4l2", "-input_format", "mjpeg",
                        "-video_size", "320x240", "-framerate", "30", "-i", dev,
                        "-an", "-vcodec", "copy", "-f", "mjpeg", "pipe:1"
                    ]
                ),
                (
                    "fallback-convert-320x240-10",
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "v4l2", "-video_size", "320x240", "-framerate", "10", "-i", dev,
                        "-an", "-vf", "fps=10,scale=320:240", "-q:v", "7",
                        "-f", "mjpeg", "pipe:1"
                    ]
                ),
            ]

            for strategy_name, cmd in strategies:
                if not camera_running:
                    break

                print(f"[CameraWorker] Trying {strategy_name} on {dev}", flush=True)

                try:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"[CameraWorker] Failed to start {strategy_name} on {dev}: {e}", flush=True)
                    continue

                # Set the pipe to non-blocking
                try:
                    fd = process.stdout.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                except Exception as e:
                    print(f"[CameraWorker] Failed to set non-blocking: {e}", flush=True)

                buf = b""
                got_frame = False
                try:
                    while camera_running:
                        try:
                            chunk = process.stdout.read(131072)
                            if chunk is None:
                                time.sleep(0.01)
                                continue
                            if not chunk:
                                if process.poll() is not None:
                                    break
                                time.sleep(0.01)
                                continue
                            buf += chunk
                        except (BlockingIOError, IOError):
                            if process.poll() is not None:
                                break
                            time.sleep(0.01)
                            continue

                        last_jpg = None
                        start_idx = 0
                        while True:
                            start = buf.find(b"\xff\xd8", start_idx)
                            if start == -1:
                                break
                            end = buf.find(b"\xff\xd9", start + 2)
                            if end == -1:
                                buf = buf[start:]
                                break
                            last_jpg = buf[start:end + 2]
                            start_idx = end + 2

                        if last_jpg is not None:
                            got_frame = True
                            with camera_lock:
                                latest_frame = last_jpg
                                frame_id += 1

                            with recording_lock:
                                if recording_file_handle is not None:
                                    try:
                                        recording_file_handle.write(last_jpg)
                                    except Exception as e:
                                        print(f"[CameraWorker] Recording write error: {e}", flush=True)

                            if start_idx >= len(buf):
                                buf = b""
                            else:
                                buf = buf[start_idx:]

                except Exception as e:
                    print(f"[CameraWorker] Loop error in {strategy_name} on {dev}: {e}", flush=True)
                finally:
                    try:
                        process.terminate()
                        process.wait(timeout=1)
                    except Exception:
                        try: process.kill()
                        except: pass
                    print(f"[CameraWorker] {strategy_name} on {dev} terminated.", flush=True)

                if got_frame:
                    break
                if camera_running:
                    print(f"[CameraWorker] No frames from {strategy_name} on {dev}, trying next mode...", flush=True)

            if got_frame:
                break
            if camera_running:
                print(f"[CameraWorker] No usable stream from {dev}, trying next camera device...", flush=True)

        if camera_running:
            time.sleep(1)

    with camera_lock:
        latest_frame = None
    print("[CameraWorker] Thread exited.", flush=True)


def mjpeg_generator():
    global active_clients

    should_start_camera = False
    with active_clients_lock:
        active_clients += 1
        if active_clients == 1:
            should_start_camera = True

    if should_start_camera:
        start_camera_thread()

    try:
        for _ in range(15):
            with camera_lock:
                if latest_frame is not None:
                    break
            time.sleep(0.1)
            
        last_sent_frame_id = -1
        frame_delay = 1.0 / max(1.0, stream_fps)
        while True:
            time.sleep(frame_delay)
            with camera_lock:
                frame = latest_frame
                current_frame_id = frame_id
                
            if frame and current_frame_id != last_sent_frame_id:
                last_sent_frame_id = current_frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
    finally:
        should_stop_camera = False
        with active_clients_lock:
            active_clients -= 1
            if active_clients <= 0:
                active_clients = 0
                should_stop_camera = True

        if should_stop_camera:
            stop_camera_thread()


@app.route("/camera/photo", methods=["POST", "GET"])
def take_photo():
    filename = datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(MEDIA_DIR, filename)

    with camera_lock:
        frame = latest_frame

    if frame is None:
        print("[Camera] No frame available for photo. Starting camera temporarily...", flush=True)
        start_camera_thread()
        for _ in range(30):
            time.sleep(0.1)
            with camera_lock:
                frame = latest_frame
            if frame is not None:
                break
        
        should_stop_camera = False
        with active_clients_lock:
            if active_clients == 0:
                should_stop_camera = True
        if should_stop_camera:
            stop_camera_thread()

    if frame is not None:
        try:
            with open(path, "wb") as f:
                f.write(frame)
            return jsonify({"success": True, "file": path})
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to save photo: {e}"}), 500

    return jsonify({"success": False, "error": "No camera frame available"}), 500


@app.route("/camera/record/start", methods=["POST", "GET"])
def start_recording():
    global recording_file_handle, record_file
    with recording_lock:
        if recording_file_handle is not None:
            return jsonify({"success": False, "message": "Already recording"}), 400

        filename = datetime.now().strftime("video_%Y%m%d_%H%M%S.mjpg")
        record_file = os.path.join(MEDIA_DIR, filename)
        try:
            recording_file_handle = open(record_file, "wb")
            start_camera_thread()
            print(f"[Camera] Started recording to {record_file}", flush=True)
            return jsonify({"success": True, "message": "Recording started", "file": record_file})
        except Exception as e:
            return jsonify({"success": False, "message": f"Failed to start recording: {e}"}), 500


@app.route("/camera/record/stop", methods=["POST", "GET"])
def stop_recording():
    global recording_file_handle, record_file
    with recording_lock:
        if recording_file_handle is None:
            return jsonify({"success": False, "message": "Not recording"}), 400

        try:
            recording_file_handle.close()
        except Exception: pass
        recording_file_handle = None
        finished_file = record_file
        record_file = None
        
        should_stop_camera = False
        with active_clients_lock:
            if active_clients == 0:
                should_stop_camera = True
        if should_stop_camera:
            stop_camera_thread()
                
        print(f"[Camera] Stopped recording. Saved to {finished_file}", flush=True)
        return jsonify({"success": True, "message": "Recording stopped", "file": finished_file})


@app.route("/video", methods=["GET"])
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/ppe", methods=["GET"])
def video_ppe():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/debug/logs", methods=["GET"])
def get_debug_logs():
    try:
        output = subprocess.check_output("journalctl -u smarthelmet -n 2000 | grep -v '/imu'", shell=True, stderr=subprocess.STDOUT)
        return Response(output, mimetype="text/plain")
    except Exception as e:
        return f"Failed to get logs: {e}"


@app.route("/debug/i2c", methods=["GET"])
def get_debug_i2c():
    try:
        results = []
        # Run regular i2cdetect first
        i2c_scan = subprocess.check_output(["i2cdetect", "-y", "4"], text=True)
        results.append("=== Standard I2C Scan ===")
        results.append(i2c_scan)
        
        results.append("\n=== Bypass Scan Trials ===")
        if bus is not None:
            try:
                orig_6b = bus.read_byte_data(IMU_ADDR, 0x6B)
                orig_6a = bus.read_byte_data(IMU_ADDR, 0x6A)
                orig_37 = bus.read_byte_data(IMU_ADDR, 0x37)
                
                # Test multiple configurations to see if bypass opens
                for val_6b in [0x00, 0x01]:
                    for val_6a in [0x00]:
                        for val_37 in [0x02, 0x22, 0x12, 0x32]:
                            # Wake up and set clock
                            bus.write_byte_data(IMU_ADDR, 0x6B, val_6b)
                            time.sleep(0.01)
                            # Disable master mode
                            bus.write_byte_data(IMU_ADDR, 0x6A, val_6a)
                            time.sleep(0.01)
                            # Configure bypass pin
                            bus.write_byte_data(IMU_ADDR, 0x37, val_37)
                            time.sleep(0.02)
                            
                            # Attempt to read magnetometer address
                            try:
                                mag_val = bus.read_byte_data(MAG_ADDR, 0x00)
                                results.append(f"SUCCESS: 6B={val_6b:#x}, 6A={val_6a:#x}, 37={val_37:#x} -> MAG ID = {mag_val:#x}")
                            except Exception:
                                results.append(f"Failed: 6B={val_6b:#x}, 6A={val_6a:#x}, 37={val_37:#x}")
                                
                # Restore original register values
                bus.write_byte_data(IMU_ADDR, 0x6B, orig_6b)
                bus.write_byte_data(IMU_ADDR, 0x6A, orig_6a)
                bus.write_byte_data(IMU_ADDR, 0x37, orig_37)
            except Exception as e:
                results.append(f"Error during trials: {e}")
        else:
            results.append("I2C bus not initialized")
            
        return Response("\n".join(results), mimetype="text/plain")
    except Exception as e:
        return f"Failed: {e}"


# ── Posture Engine ───────────────────────────────────────────────────────────

def _get_accel_raw():
    """Read raw accel counts without scaling (needed for fall thresholds)."""
    if bus is None:
        return 0, 0, 0
    with i2c_lock:
        try:
            def rw(reg):
                hi = bus.read_byte_data(IMU_ADDR, reg)
                lo = bus.read_byte_data(IMU_ADDR, reg + 1)
                v  = (hi << 8) | lo
                return v - 65536 if v >= 0x8000 else v
            return rw(0x3B), rw(0x3D), rw(0x3F)
        except Exception:
            return 0, 0, 0


def _get_gyro_raw():
    if bus is None:
        return 0, 0, 0
    with i2c_lock:
        try:
            def rw(reg):
                hi = bus.read_byte_data(IMU_ADDR, reg)
                lo = bus.read_byte_data(IMU_ADDR, reg + 1)
                v  = (hi << 8) | lo
                return v - 65536 if v >= 0x8000 else v
            return rw(0x43), rw(0x45), rw(0x47)
        except Exception:
            return 0, 0, 0


def _calc_orientation(ax, ay, az):
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
    roll  = math.degrees(math.atan2(ay, az))

    if az < 0:
        orientation = "UPSIDE-DOWN"
    elif pitch > 30:
        orientation = "LOOKING UP"
    elif pitch < -30:
        orientation = "LOOKING DOWN"
    elif roll > 30:
        orientation = "LEFT TILT"
    elif roll < -30:
        orientation = "RIGHT TILT"
    else:
        orientation = "LEVEL"

    return pitch, roll, orientation


def posture_engine():
    global posture_state, _gesture_history, _last_orientation
    global _gesture_start, _level_start, _alt_baseline

    fall_state    = "NORMAL"
    free_fall_ts  = None
    impact_ts     = None
    yaw_angle     = 0.0    # accumulated degrees: + = left, - = right
    last_tick     = time.time()

    while True:
        try:
            ax, ay, az = _get_accel_raw()
            gx, gy, gz = _get_gyro_raw()

            acc_mag  = math.sqrt(ax**2 + ay**2 + az**2)
            gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

            pitch, roll, orientation = _calc_orientation(ax, ay, az)

            # ── Turn detection: integrate gyro Z to get absolute yaw angle ──────
            now_tick = time.time()
            dt       = now_tick - last_tick
            last_tick = now_tick

            gz_cal  = gz - gyro_offset_z
            yaw_dps = gz_cal / 131.0          # degrees per second

            # Integrate: add rotation this tick
            yaw_angle += yaw_dps * dt

            # Drift correction: when head is still, slowly pull back toward 0
            # This prevents infinite accumulation but holds the angle for ~30s
            if abs(gz_cal) < _YAW_STILL_THRESH:
                yaw_angle *= _YAW_DRIFT_DECAY

            # Classify based on accumulated angle
            if yaw_angle > _YAW_TURN_ANGLE:
                turn_direction = "LOOKING LEFT"
            elif yaw_angle < -_YAW_TURN_ANGLE:
                turn_direction = "LOOKING RIGHT"
            else:
                turn_direction = "FORWARD"

            # ── Fall FSM ─────────────────────────────────────────────────
            if fall_state == "NORMAL":
                if acc_mag < _FREE_FALL_THRESHOLD:
                    fall_state   = "FREE FALL"
                    free_fall_ts = time.time()

            elif fall_state == "FREE FALL":
                if acc_mag > _IMPACT_THRESHOLD:
                    fall_state = "IMPACT"
                    impact_ts  = time.time()
                elif time.time() - free_fall_ts > 1.0:
                    fall_state = "NORMAL"   # false free-fall

            elif fall_state == "IMPACT":
                elapsed = time.time() - impact_ts

                # Wait for bounces to settle (SETTLING_TIME)
                if elapsed > _SETTLING_TIME:
                    # Check if device has calmed down after the impact
                    # NOTE: orientation != LEVEL is intentionally NOT required —
                    # a fallen person can lie flat (LEVEL) and still need help.
                    if gyro_mag < _STILL_GYRO_THRESH:
                        # Device is calm after the impact → CONFIRMED FALL
                        fall_state = "FALL DETECTED"
                    elif elapsed > 5.0:
                        # Still very active after 5s → person moved/recovered
                        fall_state = "NORMAL"

            elif fall_state == "FALL DETECTED":
                # Hold FALL DETECTED for 20s so the SOS dialog has time to appear
                if time.time() - impact_ts > 20.0:
                    fall_state = "NORMAL"

            # ── Gesture engine ───────────────────────────────────────────
            now = time.time()

            if orientation == "LEVEL":
                if _level_start is None:
                    _level_start = now
                elif now - _level_start > _LEVEL_RESET_TIME:
                    _gesture_history.clear()
                    _gesture_start = None
                _last_orientation = "LEVEL"
            else:
                _level_start = None
                if orientation != _last_orientation:
                    _gesture_history.append(orientation)
                    _last_orientation = orientation
                    if len(_gesture_history) == 1:
                        _gesture_start = now

            if _gesture_start and now - _gesture_start > _GESTURE_TIMEOUT:
                _gesture_history.clear()
                _gesture_start = None

            last_gesture = "None"
            h = list(_gesture_history)
            if len(h) >= 2:
                if h[-2:] == ["LOOKING DOWN", "LOOKING UP"]:
                    last_gesture = "NOD"
                    _gesture_history.clear(); _gesture_start = None
                elif h[-2:] == ["LOOKING UP", "LOOKING DOWN"]:
                    last_gesture = "REVERSE NOD"
                    _gesture_history.clear(); _gesture_start = None
                elif h[-2:] in (["LEFT TILT", "RIGHT TILT"], ["RIGHT TILT", "LEFT TILT"]):
                    last_gesture = "HEAD SHAKE"
                    _gesture_history.clear(); _gesture_start = None

            # ── Altitude ─────────────────────────────────────────────────
            try:
                _, alt = read_bmp280()
            except Exception:
                alt = 0.0

            if _alt_baseline is None and alt != 0.0:
                _alt_baseline = alt

            alt_delta = round(alt - _alt_baseline, 2) if _alt_baseline else 0.0

            # ── Write shared state ───────────────────────────────────────
            with posture_lock:
                posture_state["orientation"]      = orientation
                posture_state["pitch"]            = round(pitch, 1)
                posture_state["roll"]             = round(roll, 1)
                posture_state["turn_direction"]   = turn_direction
                posture_state["yaw_rate_dps"]     = round(yaw_dps, 1)
                posture_state["fall_state"]       = fall_state
                posture_state["last_gesture"]     = last_gesture
                posture_state["altitude_m"]       = round(alt, 2)
                posture_state["altitude_delta_m"] = alt_delta
                posture_state["acc_magnitude"]    = round(acc_mag, 0)

        except Exception as e:
            print(f"[PostureEngine] error: {e}")

        time.sleep(0.05)   # 20 Hz


@app.route("/posture", methods=["GET"])
def get_posture():
    with posture_lock:
        return jsonify(dict(posture_state))


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)
    
    # Camera thread is now started dynamically on-demand when requested by a client.
    
    if init_imu():
        Thread(target=calibrate_gyro, daemon=True).start()
        Thread(target=posture_engine, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
