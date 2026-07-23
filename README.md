# 🪖 Smart Helmet — Integrated Safety & Telemetry System

> **An IoT-based wearable system for industrial worker safety monitoring.**  
> Combines a Raspberry Pi helmet unit with a custom Android app for real-time PPE inspection, head-gesture commands, fall detection, and GPS-tracked emergency SOS.

---

## 📑 Table of Contents

- [System Overview](#-system-overview)
- [Hardware Platform](#-hardware-platform)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Helmet Server (Python / Flask)](#-helmet-server-python--flask)
  - [IMU Telemetry & Posture Engine](#imu-telemetry--posture-engine)
  - [Camera & Video Streaming](#camera--video-streaming)
  - [Video Recording Pipeline](#video-recording-pipeline)
  - [Fall Detection Finite State Machine](#fall-detection-finite-state-machine)
  - [Barometric Altitude (BMP280)](#barometric-altitude-bmp280)
- [Android Application (Java)](#-android-application-java)
  - [Two-Stage PPE Detection Pipeline](#two-stage-ppe-detection-pipeline)
  - [YOLOv8 Model Training](#yolov8-model-training)
  - [PaddleOCR Live Location Scanner](#paddleocr-live-location-scanner)
  - [Emergency SOS System](#emergency-sos-system)
- [Posture & Turn Detection](#-posture--turn-detection)
- [Gesture Recognition](#-gesture-recognition)
- [Hands-Free Head Commands](#-hands-free-head-commands)
- [REST API Reference](#-rest-api-reference)
- [Android Permissions](#-android-permissions)

---

## 🌐 System Overview

```
┌──────────────────────────────────┐        Wi-Fi / HTTP :5000
│     RASPBERRY PI ZERO 2 W        │ ─────────────────────────────▶  Android App
│                                  │
│  Sony IMX708 (12MP)  ─ MJPEG ──▶ │  /video     → Live camera stream
│  MPU6050 (IMU)  ─── I2C 0x68 ──▶ │  /posture   → Orientation, fall, gestures
│  BMP280 (Baro)  ─── I2C 0x76 ──▶ │  /imu       → Raw sensor data
│                                  │  /media/*   → Files: MP4, JPEG, CSV
│  GY-91 Module (MPU6050+BMP280)   │  /status    → System health
│  Python 3 + Flask  port 5000     │
│  UDP Discovery      port 5005    │ ─────────────────────────────▶  Auto-discovers IP
└──────────────────────────────────┘
```

The Android application connects to the helmet over local Wi-Fi. It continuously polls `/posture` (~1 Hz), streams `/video` for live viewing, triggers PPE compliance scans and OCR location scans, and detects fall events to initiate emergency SOS alerts.

---

## 🔧 Hardware Platform

| Component | Model / Specification | Interface |
|---|---|---|
| Main Compute | Raspberry Pi Zero 2 W (quad-core ARM Cortex-A53, 512 MB RAM) | — |
| Camera | Sony IMX708, 12 MP, Camera Module 3 | MIPI CSI-2 |
| Multi-Sensor Module | GY-91 Breakout Board | I2C |
| 6-DOF IMU | MPU6050 (3-axis accelerometer + 3-axis gyroscope) | I2C @ 0x68 |
| Barometer | BMP280 (pressure + temperature sensor) | I2C @ 0x76 |
| Android Host | Any Android 8.0+ (API level 26+) smartphone | Wi-Fi |

---

## 📁 Repository Structure

```
.
├── helmet_server/                  # Raspberry Pi server code
│   ├── helmet_server.py            # Main Flask server (IMU, camera, posture engine)
│   ├── index.html                  # Local web dashboard served at /
│   └── test_camera.py              # Standalone camera test script
│
├── android_app/                    # Android Studio project
│   └── app/src/main/
│       ├── AndroidManifest.xml     # Permissions: SMS, GPS, Bluetooth, Internet
│       ├── java/com/smarthelmet/app/
│       │   └── MainActivity.java   # Core App (~5000 lines): UI, AI inference, SOS, OCR
│       ├── assets/
│       │   ├── best_calibrated_model.tflite   # YOLOv8 PPE detection model (6 classes)
│       │   └── paddle/                        # PaddleOCR models + character set
│       └── cpp/                    # PaddleOCR JNI native bridge (C++)
│
├── docs/
│   └── images/                     # System architecture & gesture diagrams
│       ├── posture_states.jpg
│       ├── gesture_recognition.jpg
│       ├── head_commands.jpg
│       └── fall_detection_fsm.jpg
│
├── SmartHelmet.apk                 # Pre-built debug APK (sideload directly)
├── README.md                       # Main GitHub repository documentation
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Deploy the Helmet Server (Raspberry Pi)

```bash
# Install dependencies on Raspberry Pi
pip install flask smbus2

# Copy server script to Pi from host PC
scp helmet_server/helmet_server.py pi@<HELMET_IP>:~/smarthelmet/

# Execute server
python3 helmet_server.py
# Flask starts on 0.0.0.0:5000
# Gyroscope automatically calibrates on startup (~1.5s)
```

**Run as a systemd service (auto-start on boot):**

Create `/etc/systemd/system/smarthelmet.service`:
```ini
[Unit]
Description=Smart Helmet Server
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/smarthelmet/helmet_server.py
WorkingDirectory=/home/pi/smarthelmet
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable smarthelmet
sudo systemctl start smarthelmet
```

### 2. Install Android Application

- **Option A (Sideload prebuilt APK):** Transfer `SmartHelmet.apk` to your smartphone and install.
- **Option B (Build from source):** Open `android_app/` in Android Studio, connect phone via USB debugging, and tap **Run**.

### 3. Connection Setup

1. Connect phone and helmet Pi to the **same Wi-Fi network**.
2. Launch the app and navigate to the **Connect** tab.
3. The app auto-discovers the helmet IP via UDP broadcast on port 5005 (or manually enter `10.x.x.x:5000`).

---

## 🖥 Helmet Server (Python / Flask)

> **Source File:** [`helmet_server/helmet_server.py`](helmet_server/helmet_server.py)

The helmet server is implemented in Python 3 using Flask. It runs background threads for GY-91 sensor reading (MPU6050 IMU + BMP280 barometer at 20 Hz), camera streaming, video recording encoding, and REST endpoints.

### Key Thresholds & Configuration Block

```python
# ─────────────────────────────────────────────────────────────────────────────
# GY-91 SENSOR I2C ADDRESSES
# ─────────────────────────────────────────────────────────────────────────────
IMU_ADDR = 0x68          # MPU6050 Accelerometer + Gyroscope
BMP_ADDR = 0x76          # BMP280 Barometric Pressure + Temperature

# ─────────────────────────────────────────────────────────────────────────────
# FALL DETECTION THRESHOLDS (Raw ADC counts at ±2g, 16,384 LSB/g)
# ─────────────────────────────────────────────────────────────────────────────
_FREE_FALL_THRESHOLD = 4000   # ~0.24g  — near-weightlessness during fall
_IMPACT_THRESHOLD    = 22000  # ~1.34g  — impact force threshold
_STILL_GYRO_THRESH   = 5000   # ~38°/s  — post-impact stillness check
_SETTLING_TIME       = 2.0    # seconds to wait for post-impact bounce settling

# ─────────────────────────────────────────────────────────────────────────────
# GESTURE & TIMING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_GESTURE_TIMEOUT  = 1.5   # seconds: gesture window limit
_LEVEL_RESET_TIME = 2.0   # seconds at LEVEL orientation before clearing history
_YAW_TURN_ANGLE   = 30.0  # degrees: threshold for turn classification
```

---

### IMU Telemetry & Posture Engine

The posture engine calculates head pitch and roll from accelerometer data:

```python
# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION COMPUTATION BLOCK
# ─────────────────────────────────────────────────────────────────────────────
#   pitch = atan2(-ax, sqrt(ay² + az²))   [degrees]
#   roll  = atan2( ay, az)                [degrees]
#
# Note: -ax is used to correct for physical inverted MPU6050 mounting orientation.
#
# State classification priority order:
#   1. az < 0          → UPSIDE-DOWN  (helmet inverted)
#   2. pitch > +30°    → LOOKING UP
#   3. pitch < -30°    → LOOKING DOWN
#   4. roll  > +30°    → LEFT TILT
#   5. roll  < -30°    → RIGHT TILT
#   6. (default)       → LEVEL
# ─────────────────────────────────────────────────────────────────────────────

def _calc_orientation(ax, ay, az):
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
    roll  = math.degrees(math.atan2(ay, az))

    if az < 0:             orientation = "UPSIDE-DOWN"
    elif pitch > 30:       orientation = "LOOKING UP"
    elif pitch < -30:      orientation = "LOOKING DOWN"
    elif roll > 30:        orientation = "LEFT TILT"
    elif roll < -30:       orientation = "RIGHT TILT"
    else:                  orientation = "LEVEL"

    return pitch, roll, orientation
```

Turn direction is calculated by integrating calibrated Z-axis gyroscope data:

```python
# ─────────────────────────────────────────────────────────────────────────────
# TURN DIRECTION DETECTION BLOCK (Gyroscope Z Integration)
# ─────────────────────────────────────────────────────────────────────────────
#   yaw_dps = (gz_raw - gyro_offset_z) / 131.0   [°/s]
#
#   Deadband: if |yaw_dps| <= 4.0 °/s → treat as noise, apply decay factor
#   Active integration: yaw_angle += yaw_dps * dt
#   Decay factor: yaw_angle *= 0.88 per 50ms tick when stationary
#   Clamp: yaw_angle clamped to [-90.0, +90.0] degrees
#
#   yaw_angle > +25.0° → LOOKING LEFT
#   yaw_angle < -25.0° → LOOKING RIGHT
#   otherwise          → FORWARD
# ─────────────────────────────────────────────────────────────────────────────

gz_cal  = gz - gyro_offset_z
yaw_dps = gz_cal / 131.0

if abs(yaw_dps) > 4.0:
    yaw_angle += yaw_dps * dt     # accumulate head turn
else:
    yaw_angle *= 0.88             # decay smoothly when stationary

yaw_angle = max(-90.0, min(90.0, yaw_angle))

if   yaw_angle >  25.0: turn_direction = "LOOKING LEFT"
elif yaw_angle < -25.0: turn_direction = "LOOKING RIGHT"
else:                   turn_direction = "FORWARD"
```

---

### Camera & Video Streaming

The camera engine manages an on-demand background worker (`camera_worker`) that starts when HTTP clients request `/video` and stops when zero clients remain:

```python
# ─────────────────────────────────────────────────────────────────────────────
# CAMERA CAPTURE STRATEGY FALLBACK CHAIN BLOCK
# ─────────────────────────────────────────────────────────────────────────────
# The server attempts 6 capture methods in order until valid frames arrive:
#   1. rpicam-vid (Pi Camera Module 3, MJPEG inline, 640x480 @ 30fps)
#   2. libcamera-vid (alternative Pi stack, 640x480 @ 30fps)
#   3. v4l2-ctl (V4L2 direct MJPEG stream, 640x480 @ 30fps)
#   4. FFmpeg MJPEG copy (640x480 @ 30fps)
#   5. FFmpeg MJPEG copy (320x240 @ 30fps)
#   6. FFmpeg raw V4L2 transcode (320x240 @ 10fps)
#
# Serves MJPEG over HTTP: multipart/x-mixed-replace; boundary=frame
# ─────────────────────────────────────────────────────────────────────────────
```

---

### Video Recording Pipeline

```python
# ─────────────────────────────────────────────────────────────────────────────
# RECORDING & MP4 CONVERSION PIPELINE BLOCK
# ─────────────────────────────────────────────────────────────────────────────
# 1. POST /camera/record/start creates video_YYYYMMDD_HHMMSS.mjpg.
# 2. Camera worker appends incoming JPEG frames directly to the raw file.
# 3. POST /camera/record/stop closes file handle instantly and returns.
# 4. Async background thread executes FFmpeg conversion:
#
#    ffmpeg -y -f mjpeg -framerate 30 -i input.mjpg #           -c:v libx264 -pix_fmt yuv420p -preset ultrafast #           -movflags +faststart output.mp4
#
#    -movflags +faststart relocates metadata atom to file start for web streaming.
# ─────────────────────────────────────────────────────────────────────────────
```

---

### Fall Detection Finite State Machine

<p align="center">
  <img src="./docs/images/fall_detection_fsm.jpg" alt="SMART HELMET 4-STAGE FALL DETECTION FSM" width="90%"/>
  <br/>
  <sub><b>Figure: SMART HELMET 4-Stage Fall Detection Finite State Machine</b></sub>
</p>

```python
# ─────────────────────────────────────────────────────────────────────────────
# FALL DETECTION 4-STAGE FSM BLOCK
# ─────────────────────────────────────────────────────────────────────────────
# State 1: NORMAL
#   - Monitoring acc_mag = sqrt(ax² + ay² + az²)
#   - If acc_mag < 4000 ADC counts (~0.24g) → Transition to FREE FALL
#
# State 2: FREE FALL
#   - Weightlessness detected. Waiting for impact.
#   - If acc_mag > 22000 ADC counts (~1.34g) within 1.0s → Transition to IMPACT
#   - If 1.0s expires without impact → Reset to NORMAL (false alarm)
#
# State 3: IMPACT
#   - Impact force recorded. Waiting 2.0s for bounce settling.
#   - After 2.0s, if gyro_mag < 5000 (~38°/s) → Transition to FALL DETECTED
#   - If movement persists > 5.0s → Reset to NORMAL (worker recovered)
#
# State 4: FALL DETECTED
#   - Fall confirmed. Triggers Android SOS countdown on next /posture poll.
#   - Auto-resets to NORMAL after 20 seconds.
# ─────────────────────────────────────────────────────────────────────────────
```

---

### Barometric Altitude (BMP280)

```python
# ─────────────────────────────────────────────────────────────────────────────
# BAROMETRIC ALTITUDE BLOCK (BMP280)
# ─────────────────────────────────────────────────────────────────────────────
# Address: 0x76 | Chip ID: 0x58
# Pressure formula:
#   altitude = 44330.0 * (1.0 - (pressure / 101325.0) ** (1.0 / 5.255))  [metres]
#
# Baseline elevation (_alt_baseline) captured on first reading.
# Relative height change reported as: altitude_delta_m = current - baseline
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 📱 Android Application (Java)

> **Source File:** [`android_app/app/src/main/java/com/smarthelmet/app/MainActivity.java`](android_app/app/src/main/java/com/smarthelmet/app/MainActivity.java)

---

### Two-Stage PPE Detection Pipeline

The PPE inspection system uses a cascaded two-stage architecture to maximize inference efficiency on mobile hardware:

1. **Stage 1 (Google ML Kit Gating):** Detects human presence using Face and Object detection. If no person is stably present for ≥ 800 ms, Stage 2 is bypassed, saving > 60% CPU compute.
2. **Stage 2 (YOLOv8 TFLite Model):** Runs custom 640×640 object detection on crops generated by Stage 1 to detect 5 required PPE gear items (**Helmet, Vest, Gloves, Mask, Goggles**).

```java
// ─────────────────────────────────────────────────────────────────────────────
// STAGE 1: GOOGLE ML KIT PERSON PRESENCE & ROI DERIVATION
// ─────────────────────────────────────────────────────────────────────────────
// 1. Initialize ML Kit Face Detector (PERFORMANCE_MODE_FAST, minFaceSize=0.08f)
if (mlKitFaceDetector != null) {
    Task<List<Face>> faceTask = mlKitFaceDetector.process(image);
    List<Face> faces = Tasks.await(faceTask);

    if (faces != null && !faces.isEmpty()) {
        for (Face face : faces) {
            Rect rect = face.getBoundingBox();
            float faceW = rect.width();
            float faceH = rect.height();

            // Filter out distant/noisy faces (< 6% width or < 7% height)
            if (faceW < bitmap.getWidth() * 0.06f || faceH < bitmap.getHeight() * 0.07f) {
                continue;
            }
            realPersonDetected = true;

            // Head ROI (expanded ±55% wide, +135% height upward for helmet region)
            float headLeft   = Math.max(0, faceLeft - faceW * 0.55f);
            float headTop    = Math.max(0, faceTop - faceH * 1.35f);
            float headRight  = Math.min(bitmap.getWidth(), faceRight + faceW * 0.55f);
            float headBottom = Math.min(bitmap.getHeight(), faceBottom + faceH * 0.40f);

            // Body ROI (expanded ±250% wide, -40% to +450% height down for vest region)
            float bodyLeft   = Math.max(0, faceCx - faceW * 2.5f);
            float bodyTop    = Math.max(0, faceTop - faceH * 0.40f);
            float bodyRight  = Math.min(bitmap.getWidth(), faceCx + faceW * 2.5f);
            float bodyBottom = Math.min(bitmap.getHeight(), faceBottom + faceH * 4.5f);

            stage1Regions.add(new Stage1Detection(bodyLeft, bodyTop, bodyRight, bodyBottom, 0, 1.0f));
            stage1Regions.add(new Stage1Detection(headLeft, headTop, headRight, headBottom, 1, 1.0f));
        }
    }
}

// Track person stability: must be continuously detected for >= 800ms before scan starts
boolean personCurrentlyInFrame = (lastPersonDetectedTime > 0 && 
    (now - lastPersonDetectedTime < PPE_PERSON_ABSENT_RESET_MS)); // 2200ms reset
boolean personStableForScan = personCurrentlyInFrame &&
    personFirstDetectedTime > 0 &&
    (nowTime - personFirstDetectedTime >= 800);
```

```java
// ─────────────────────────────────────────────────────────────────────────────
// STAGE 2: YOLOV8 TFLITE INFERENCE & ANATOMICAL FILTERING
// ─────────────────────────────────────────────────────────────────────────────
// Preprocess cropped ROI to 640x640 FLOAT32 normalized tensor [0.0, 1.0]
ImageProcessor imageProcessor = new ImageProcessor.Builder()
        .add(new ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
        .add(new NormalizeOp(0.0f, 255.0f))
        .build();

TensorImage stage2Image = new TensorImage(DataType.FLOAT32);
stage2Image.load(croppedBitmap);
stage2Image = imageProcessor.process(stage2Image);

// Model Output Tensor Shape: [1 x 10 x 8400]
float[][][] stage2Out = new float[1][10][8400];
tflite.run(stage2Image.getBuffer(), stage2Out);

// Class Index Remapping & Anatomical Filtering across 8,400 anchors
for (int boxIdx = 0; boxIdx < 8400; boxIdx++) {
    int modelClassId = -1;
    float maxConf = -1.0f;
    for (int c = 0; c < 6; c++) {
        float conf = stage2Out[0][4 + c][boxIdx];
        if (conf > maxConf) { maxConf = conf; modelClassId = c; }
    }

    // Remap HuggingFace class indices to App indices:
    if (modelClassId == 0)      modelClassId = 1; // HF Gloves -> App Gloves
    else if (modelClassId == 1) modelClassId = 5; // HF Vest -> App Vest
    else if (modelClassId == 2) modelClassId = 3; // HF Goggles -> App Goggles
    else if (modelClassId == 3) modelClassId = 2; // HF Helmet -> App Helmet
    else if (modelClassId == 4) modelClassId = 4; // HF Mask -> App Mask
    else if (modelClassId == 5) continue;         // Ignore boots

    // Anatomical Filter 1: Vest (5) cannot exist inside Head crops (classId == 1)
    if (region.classId == 1 && modelClassId == 5) continue;

    // Anatomical Filter 2: Vest bbox center Y must be in torso region (cy >= 250px)
    if (modelClassId == 5 && stage2Out[0][1][boxIdx] < 250f) continue;

    // Per-Class Confidence Threshold Check
    float cutoff = getConfidenceThreshold(modelClassId);
    if (maxConf > cutoff) {
        // Translate bounding box coordinates back to full frame space and save detection
        localDetections.add(new Detection(fullLeft, fullTop, fullRight, fullBottom, modelClassId, maxConf));
    }
}
```

```java
// ─────────────────────────────────────────────────────────────────────────────
// 15-SECOND COMPLIANCE ACCUMULATION & TTS REMINDER CASCADE
// ─────────────────────────────────────────────────────────────────────────────
if (isScanningActive && (currentTime - scanStartTime >= 15000)) {
    isScanningActive = false;
    scanResultsAnnounced = true;

    List<String> violations = new ArrayList<>();
    if (!scanDetectedHelmet)  violations.add("wear safety helmet");
    if (!scanDetectedVest)    violations.add("wear safety vest");
    if (!scanDetectedGloves)  violations.add("wear protective gloves");
    if (!scanDetectedMask)    violations.add("wear face mask");
    if (!scanDetectedGoggles) violations.add("wear safety goggles");

    String speechText = violations.isEmpty() ?
            "Safety compliance passed. Access granted." :
            "Warning. Please " + String.join(", ", violations) + ".";

    // Save scan report to internal storage and trigger TTS speech output
    savePpeScanReport(violations, speechText);
    ttsEngine.speak(speechText, TextToSpeech.QUEUE_FLUSH, null, "PpeSpeechAlert");
}
```

### YOLOv8 Model Training

Model trained on **Kaggle** with NVIDIA Tesla T4 GPU for 5 target PPE gear classes (**Helmet, Vest, Gloves, Mask, Goggles**):

| Parameter | Specification |
|---|---|
| Dataset Size | 42,000 annotated images |
| Train / Val / Test Split | 33,600 (80%) / 4,200 (10%) / 4,200 (10%) |
| Target Classes | 5 Classes: Helmet, Vest, Gloves, Mask, Goggles |
| Resolution | 640 × 640 pixels |
| Epochs | 100 (AdamW, lr=0.001, momentum=0.937, batch=64) |
| Export Format | TFLite FP16 Quantized (`best_calibrated_model.tflite`) |
| Performance (Test Set) | **94.0% Precision**, **92.1% Recall**, **93.7% mAP@0.50** |

### PaddleOCR Live Location Scanner

The Live Location Subsystem enables hands-free zone recognition by continuously reading zone identification signage using Baidu PaddleOCR (PaddleLite C++ engine with JNI bridge) and OpenCV.

#### Architecture Breakdown:
1. **Stage A — Text Detection (DB Algorithm):** MobileNetV3 backbone generates probability and threshold maps. Differentiable Binarization extracts text bounding boxes under challenging industrial lighting.
2. **Stage B — Text Recognition (CRNN + BiLSTM):** CNN extracts visual features → 2-layer Bidirectional LSTM models character sequence dependencies → CTC decoder outputs final recognized text string.

```java
// ─────────────────────────────────────────────────────────────────────────────
// PADDLEOCR ENGINE RUNTIME & ASSET VALIDATION
// ─────────────────────────────────────────────────────────────────────────────
private boolean isPaddleOcrReady() {
    return arePaddleOcrAssetsPresent() && isPaddleRuntimeReady();
}

private boolean arePaddleOcrAssetsPresent() {
    try {
        // Verify key dictionary asset in assets/paddle/
        getAssets().open("paddle/ppocr_keys_v1.txt").close();
        return true;
    } catch (Exception e) {
        return false;
    }
}

private List<String> runPaddleOcr(Bitmap bitmap) {
    if (!isPaddleOcrReady() || bitmap == null) return new ArrayList<>();
    // Execute Native C++ PaddleLite Inference Engine via JNI
    return runNativePaddleOcrInference(bitmap);
}
```

```java
// ─────────────────────────────────────────────────────────────────────────────
// MULTI-FRAME TEMPORAL CONSENSUS & 60-SECOND COOLDOWN LOGIC
// ─────────────────────────────────────────────────────────────────────────────
private void handleLiveCodeCandidate(final JSONObject mapping, String recognizedText) {
    long now = System.currentTimeMillis();
    String normalized = recognizedText.toUpperCase().replaceAll("\\s+", "");

    // 1. Cooldown Check: Ignore code if SMS was dispatched for it within last 60 seconds
    if (normalized.equals(lastSentLiveCode) && now - lastSentLiveCodeMs < 60000) {
        setLiveScanStatus("Matched recently - cooldown active");
        return;
    }

    // 2. Multi-Frame Consensus: Require 3 matching reads within a 3,000ms window
    if (!normalized.equals(pendingLiveCode) || now - pendingLiveCodeFirstMs > 3000) {
        pendingLiveCode = normalized;
        pendingLiveCodeHits = 1;
        pendingLiveCodeFirstMs = now;
    } else {
        pendingLiveCodeHits++;
    }

    setLiveScanStatus("Matched " + mapping.optString("code", normalized) + 
                      " (" + pendingLiveCodeHits + "/3 hits)");

    // 3. Validation Threshold Reached: Trigger SMS alert & save location log
    if (pendingLiveCodeHits >= 3) {
        lastSentLiveCode = normalized;
        lastSentLiveCodeMs = now;

        ttsEngine.speak("Text detected and verified.", TextToSpeech.QUEUE_FLUSH, null, "OcrSuccess");
        sendLiveLocationSms(mapping, mapping.optString("code", pendingLiveCode));

        // Reset hit counters after successful dispatch
        pendingLiveCode = "";
        pendingLiveCodeHits = 0;
        pendingLiveCodeFirstMs = 0;
    }
}
```

```java
// ─────────────────────────────────────────────────────────────────────────────
// LOCATION SMS DISPATCH & LOCAL LOGGING
// ─────────────────────────────────────────────────────────────────────────────
private void sendLiveLocationSms(JSONObject mapping, String detectedCode) {
    String phone = mapping.optString("phone", "");
    String location = mapping.optString("location", "");

    String smsMessage = "📍 Smart Helmet Location Alert\n" +
                        "Zone Code: " + detectedCode + "\n" +
                        "Location: " + location + "\n" +
                        "Time: " + new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss", Locale.US).format(new Date());

    if (!phone.isEmpty()) {
        try {
            SmsManager smsManager = SmsManager.getDefault();
            ArrayList<String> parts = smsManager.divideMessage(smsMessage);
            smsManager.sendMultipartTextMessage(phone, null, parts, null, null);
        } catch (Exception e) {
            // Fallback to System Messages App Intent if direct SMS permission denied
            Intent intent = new Intent(Intent.ACTION_SENDTO);
            intent.setData(Uri.parse("smsto:" + phone));
            intent.putExtra("sms_body", smsMessage);
            startActivity(intent);
        }
    }
    // Save timestamped entry in internal live_location_logs/ directory
    saveLiveLocationLog(detectedCode, location, phone, true, "Sent by OCR code match");
}
```

### Emergency SOS System

```java
// ─────────────────────────────────────────────────────────────────────────────
// EMERGENCY SOS DISPATCH BLOCK
// ─────────────────────────────────────────────────────────────────────────────
// Triggered automatically on fall_state == "FALL DETECTED" or via manual SOS button.
// 1. Starts 15-second cancellable CountDownTimer with TTS audio warnings.
// 2. Fetches GPS fix in parallel (GPS_PROVIDER + NETWORK_PROVIDER).
// 3. Dual SMS Dispatch:
//    - Primary: SmsManager.sendMultipartTextMessage() (silent background SMS)
//    - Fallback: Intent.ACTION_SENDTO (smsto: URI pre-populated in Messages app)
// ─────────────────────────────────────────────────────────────────────────────
```

---

## 🧭 Posture & Turn Detection

The helmet classifies head orientation and turn direction into the following states:

<p align="center">
  <img src="./docs/images/posture_states.jpg" alt="SMART HELMET POSTURE AND TURN DETECTION" width="90%"/>
  <br/>
  <sub><b>Figure: Head Posture States & Gyro-Z Turn Direction Detection</b></sub>
</p>

| State | Category | Detection Condition & Threshold |
|---|---|---|
| **LEVEL** | Static Pose | Neutral position (`pitch` ~ 0°, `roll` ~ 0°) |
| **LOOKING UP** | Pitch Pose | Head tilted backward (`pitch` > **+30°**) |
| **LOOKING DOWN** | Pitch Pose | Head tilted forward (`pitch` < **−30°**) |
| **LEFT TILT** | Roll Pose | Head rolling left (`roll` > **+30°**) |
| **RIGHT TILT** | Roll Pose | Head rolling right (`roll` < **−30°**) |
| **UPSIDE-DOWN** | Critical Inversion | Helmet inverted (`az` < 0, gravity reversed) |
| **LOOKING LEFT** | Yaw Turn | Head turned left (`yaw_angle` > **+25°**, Gyro-Z) |
| **LOOKING RIGHT** | Yaw Turn | Head turned right (`yaw_angle` < **−25°**, Gyro-Z) |
| **FORWARD** | Yaw Turn | Head facing forward (`yaw_angle` in [−25°, +25°]) |

---

## 🤚 Gesture Recognition

Gestures are recognized from orientation transitions completed within **1.5 seconds**:

<p align="center">
  <img src="./docs/images/gesture_recognition.jpg" alt="SMART HELMET GESTURE RECOGNITION" width="90%"/>
  <br/>
  <sub><b>Figure: Head Gesture Sequence Detection (NOD, REVERSE NOD, HEAD SHAKE)</b></sub>
</p>

| Gesture | Motion Sequence | Performing Motion |
|---|---|---|
| **NOD** | `LOOKING DOWN` → `LOOKING UP` | Tilt head down then quickly up |
| **REVERSE NOD** | `LOOKING UP` → `LOOKING DOWN` | Tilt head up then quickly down |
| **HEAD SHAKE** | `LEFT TILT` ↔ `RIGHT TILT` | Tilt head left then right (or vice versa) |

---

## 🎮 Hands-Free Head Commands

Rapid-repeat motions (3 occurrences within **3.5 seconds**) trigger app actions without touching the phone:

<p align="center">
  <img src="./docs/images/head_commands.jpg" alt="SMART HELMET HANDS FREE HEAD COMMANDS" width="90%"/>
  <br/>
  <sub><b>Figure: Rapid-Repeat Hands-Free Head Commands (START_PPE, START_LOCATION, STOP_ALL)</b></sub>
</p>

| Command | Motion Pattern | Action Triggered |
|---|---|---|
| **START_PPE** | Tilt head **LEFT** 3× in ≤3.5s | Starts 15-second PPE safety scan |
| **START_LOCATION** | Tilt head **RIGHT** 3× in ≤3.5s | Starts PaddleOCR location code scan |
| **STOP_ALL** | **NOD** (UP or DOWN) 3× in ≤3.5s | Stops all active AI scans immediately |

---

## 🌐 REST API Reference

All endpoints operate on port 5000:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves local web dashboard |
| `/status` | GET | Camera, IMU, and recording state JSON |
| `/imu` | GET | Raw sensor data (accel, gyro, temp) |
| `/posture` | GET | Posture state, orientation, gestures, fall FSM, head commands |
| `/video` | GET | Live MJPEG stream (`multipart/x-mixed-replace`) |
| `/video/ppe` | GET | Duplicate stream endpoint for PPE tab |
| `/camera/photo` | GET / POST | Capture still JPEG |
| `/camera/record/start` | GET / POST | Start video recording |
| `/camera/record/stop` | GET / POST | Stop recording and encode MP4 |
| `/imu/record/start` | GET / POST | Start IMU CSV logging |
| `/imu/record/stop` | GET / POST | Stop IMU CSV logging |
| `/media/list` | GET | List recorded media files |
| `/media/download/<filename>` | GET | Download media file |
| `/media/delete/<filename>` | DELETE / POST | Delete media file |
| `/debug/logs` | GET | Server systemd logs |
| `/debug/i2c` | GET | Bus scan diagnostic |

---

## 🔐 Android Permissions

| Permission | Purpose |
|---|---|
| `INTERNET` | Connection to Flask server |
| `ACCESS_NETWORK_STATE` | Wi-Fi connectivity monitoring |
| `SEND_SMS` | Silent emergency SOS alert dispatch |
| `ACCESS_FINE_LOCATION` | High-accuracy GPS location for SOS payload |
| `ACCESS_COARSE_LOCATION` | Network location fallback |
| `BLUETOOTH_CONNECT` / `BLUETOOTH_SCAN` | Legacy Bluetooth device support |

---

*Smart Helmet Integrated Safety & Telemetry System — Version 2.0*
