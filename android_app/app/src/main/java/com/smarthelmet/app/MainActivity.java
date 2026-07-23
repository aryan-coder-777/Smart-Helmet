package com.smarthelmet.app;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.nio.MappedByteBuffer;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Build;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.telephony.SmsManager;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.GridLayout;
import android.widget.RelativeLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import android.view.ViewGroup;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * =====================================================================================
 * SMART HELMET COMMAND CENTER — MAIN ACTIVITY
 * =====================================================================================
 * Package: com.smarthelmet.app
 * Architecture: Android Single-Activity Model (View-based Tab Navigation)
 * Threading: ExecutorService (4 Background Threads) + Handler (Main UI Thread)
 * 
 * CORE RESPONSIBILITIES & SUBSYSTEMS:
 * 1. Network & UDP Auto-Discovery:
 *    - Connects to Raspberry Pi Flask Server on Port 5000 via local Wi-Fi.
 *    - UDP Broadcast on Port 5005 auto-discovers helmet IP without manual input.
 * 
 * 2. Real-Time Telemetry & Posture Engine:
 *    - Polls GET /posture @ 1 Hz to display Pitch, Roll, Yaw, Altitude, and Fall status.
 *    - 3D Canvas Visualizer (Helmet3DView) renders real-time head orientation.
 *    - Listens for Hands-Free Head Commands (START_PPE, START_LOCATION, STOP_ALL).
 * 
 * 3. Emergency SOS Subsystem:
 *    - Triggers on Fall Detection FSM state "FALL DETECTED" or manual SOS button.
 *    - Displays 15-second cancellable countdown dialog with TTS audio alerts.
 *    - Parallel GPS acquisition (GPS_PROVIDER + NETWORK_PROVIDER).
 *    - Dual SMS Dispatch: Direct silent SMS (primary) or System Messages app (fallback).
 * 
 * 4. Two-Stage PPE Safety Inspection Pipeline:
 *    - Stage 1: Google ML Kit Face & Object Gating (detects worker stability ≥ 800ms).
 *    - Stage 2: YOLOv8 TFLite Model (best_calibrated_model.tflite, 640x640, 4 CPU threads).
 *    - Applies HuggingFace→App class index remapping, per-class confidence thresholds,
 *      and anatomical filtering rules (Head Crop vest filter, Vest torso Y-coord filter).
 *    - 15-Second Accumulation Window: TTS compliance results & report PDF/file log export.
 * 
 * 5. Live Location OCR Scanner (PaddleOCR):
 *    - PaddleLite C++ JNI bridge (DB text detection + CRNN BiLSTM recognition).
 *    - Runs every 700 ms. Multi-frame temporal consensus (3 matching reads in 3s).
 *    - Per-code 60-second cooldown timer prevents duplicate location SMS dispatches.
 * =====================================================================================
 */
public class MainActivity extends Activity {
        // ─────────────────────────────────────────────────────────────────────────────
    // CONSTANTS & CONFIGURATION PARAMETERS
    // ─────────────────────────────────────────────────────────────────────────────
    private static final String PREFS = "smart_helmet_prefs";
    private static final String DEFAULT_HOST = "10.245.158.252:5000";
    private static final UUID SPP_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");
    private static final int STREAM_CONNECT_TIMEOUT_MS = 3500;
    private static final int STREAM_READ_TIMEOUT_MS = 15000;
    private static final int STREAM_RETRY_DELAY_MS = 700;
    private static final int CAMERA_DISPLAY_FRAME_INTERVAL_MS = 50;
    private static final int PPE_DISPLAY_FRAME_INTERVAL_MS = 100;
    private static final int PPE_PERSON_ABSENT_RESET_MS = 2200;
    private static final int LIVE_OCR_FRAME_INTERVAL_MS = 700;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    // Camera tab stream — display only, no AI, full FPS
    private volatile int streamGeneration = 0;
    // PPE tab stream — fully independent thread with 2-stage AI inference
    private volatile int ppeStreamGen = 0;
    private volatile boolean ppeReaderRunning = false;
    private boolean ppeStreamOn = false;
    private HttpURLConnection ppeStreamConn = null;

    // --- TENSORFLOW LITE FOR TWO-STAGE PPE DETECTION ---
        // ─────────────────────────────────────────────────────────────────────────────
    // TENSORFLOW LITE & GOOGLE ML KIT (TWO-STAGE PPE DETECTION SUBSYSTEM)
    // ─────────────────────────────────────────────────────────────────────────────
    private Interpreter tflite;         // Stage 2 model (best_calibrated_model.tflite)
    private com.google.mlkit.vision.objects.ObjectDetector mlKitDetector; // Stage 1 ML Kit Object Detector
    private com.google.mlkit.vision.face.FaceDetector mlKitFaceDetector; // Stage 1 person presence detector
    private boolean isModelLoaded = false;
    private final int MODEL_INPUT_SIZE = 640; // Updated model input size is 640x640
    private final String[] ppeLabels = {
        "Boots", "Gloves", "Helmet", "Goggles", "Mask", "Vest"
    };

    // Alert Status flags for two-stage geometric logic
        // ─────────────────────────────────────────────────────────────────────────────
    // PPE SAFETY WARNING & HUD ALERT STATE FLAGS
    // ─────────────────────────────────────────────────────────────────────────────
    private volatile boolean isPersonInFrame = false;
    private volatile boolean alertNoHelmet = false;
    private volatile boolean alertNoGoggles = false;
    private volatile boolean alertNoMask = false;
    private volatile boolean alertNoVest = false;
    private volatile boolean alertMissingPpe = false;
    private volatile boolean alertChinstrapOpen = false;

    // Removed legacy mapping methods to clean up the codebase.

    // ----------------------------------------
    // --- TEXT TO SPEECH ALERT SYSTEM ---
    private android.speech.tts.TextToSpeech ttsEngine;
    private boolean isTtsReady = false;


    private LinearLayout root;
    private LinearLayout content;
    private LinearLayout navBar;
    private EditText hostInput;
    private TextView topStatus;
    private TextView wifiStatus;
    private TextView bluetoothStatus;
    private TextView cameraState;
    private TextView recordingState;
    private TextView accelX;
    private TextView accelY;
    private TextView accelZ;
    private TextView gyroX;
    private TextView gyroY;
    private TextView gyroZ;
    private TextView altitudeText;

    // Posture tab views
    private TextView postureOrientation;
    private TextView postureFallState;
    private TextView postureAltDelta;
    private TextView postureGesture;
    private TextView posturePitch;
    private TextView postureRoll;
    private TextView postureTurnDir;
    private ImageView cameraImage;
    private TextView previewPlaceholder;
    private TextView previewStatus;
    private Button cameraToggleButton;
    private LinearLayout ppeLogsListContainer;
    private BluetoothSocket bluetoothSocket;
    private HttpURLConnection streamConnection;
    private TextView calibStatusText;
    private TextView calibOffsetsText;
    private Button togglePollBtn;
    private Button toggleRecordBtn;
    private Cube3DView cubeView;
    private boolean imuRecordingActive;
    private LinearLayout logsListContainer;
    private LinearLayout cameraLogsListContainer;
    private LinearLayout liveMappingsContainer;
    private LinearLayout liveLogsContainer;
    private EditText liveCodeInput;
    private EditText liveLocationInput;
    private EditText livePhoneInput;
    private TextView liveScanStatus;
    private ImageView liveCameraImage;
    private TextView livePreviewPlaceholder;
    private Button liveScanButton;
    private boolean liveScanOn = false;
    private volatile boolean liveReaderRunning = false;
        // ─────────────────────────────────────────────────────────────────────────────
    // LIVE LOCATION PADDLEOCR STATE VARIABLES
    // ─────────────────────────────────────────────────────────────────────────────
    private volatile boolean liveOcrProcessing = false;
    private volatile int liveStreamGen = 0;
    private HttpURLConnection liveStreamConn = null;
    private String pendingLiveCode = "";
    private int pendingLiveCodeHits = 0;
    private long pendingLiveCodeFirstMs = 0;
    private String lastSentLiveCode = "";
    private long lastSentLiveCodeMs = 0;
    private boolean paddleRuntimeChecked = false;
    private boolean paddleRuntimeReady = false;
    private com.baidu.paddle.lite.demo.ocr.Predictor paddlePredictorInstance = null;
    private long lastTtsTime = 0;
    private long personFirstDetectedTime = 0;
    private long lastPersonDetectedTime = 0;
    private boolean isScanningActive = false;
    private long scanStartTime = 0;
    private boolean scanResultsAnnounced = false;
    private boolean scanDetectedHelmet = false;
    private boolean scanDetectedVest = false;
    private boolean scanDetectedGloves = false;
    private boolean scanDetectedMask = false;
    private boolean scanDetectedGoggles = false;
    private final float[] scanBestConfidence = new float[6];
    private final int[] scanHitCount = new int[6];
    private final long[] scanFirstHitTime = new long[6];
    private int scanAnnouncementCount = 0;
    private int ppePersonReportCounter = 0;
    private volatile boolean isModelProcessing = false;
    private final java.util.List<Detection> latestDetections = new java.util.ArrayList<>();
    private final java.util.List<CachedDetection> cachedDetections = new java.util.ArrayList<>();

    private boolean polling;
    private boolean streamOn;
    private volatile boolean streamReaderRunning;
    private boolean onboarded;
    private String activeTab = "dashboard";
    private String lastWifiState = "Not connected";
    private String lastBluetoothState = "Not paired";

    // ── SOS / Accident Detection ───────────────────────────────────────
    private String  lastFallState     = "NORMAL";   // track transitions
    private boolean sosPending        = false;       // true while countdown is active
    private CountDownTimer sosTimer   = null;
    private double  sosLat            = 0.0;
    private double  sosLng            = 0.0;
    private boolean sosLocationReady  = false;
    private static final int REQ_SMS_LOC = 1001;

    private final Runnable imuPoller = new Runnable() {
        @Override
        public void run() {
            if (!polling || !"sensors".equals(activeTab)) return;
            requestJson("/imu", new JsonCallback() {
                @Override
                public void onSuccess(JSONObject json) {
                    updateImu(json);
                    handler.postDelayed(imuPoller, 1000);
                }
                @Override
                public void onError(String message) {
                    setTopStatus("Sensor link waiting");
                    handler.postDelayed(imuPoller, 2500);
                }
            });
        }
    };

    private final Runnable posturePoller = new Runnable() {
        @Override
        public void run() {
            if (!polling || !"posture".equals(activeTab)) return;
            requestJson("/posture", new JsonCallback() {
                @Override
                public void onSuccess(JSONObject json) {
                    updatePosture(json);
                    handler.postDelayed(posturePoller, 150);
                }
                @Override
                public void onError(String message) {
                    handler.postDelayed(posturePoller, 2500);
                }
            });
        }
    };

    @Override
        /**
     * Activity Lifecycle — onCreate
     * Initializes UI layout, SharedPreferences, TextToSpeech engine, permissions check,
     * and auto-connects to helmet server via UDP discovery.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 1. Existing onboarding routing setup
        onboarded = getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean("onboarded", false);
        if (onboarded) {
            buildAppShell();
        } else {
            showWelcome();
        }

        // PPE model initialization has been moved to loadPpeModelIfNeeded(), which is called when entering the PPE tab.

        // 3. Initialize Text-To-Speech Vocal Audio Engine
        ttsEngine = new android.speech.tts.TextToSpeech(this, new android.speech.tts.TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == android.speech.tts.TextToSpeech.SUCCESS) {
                    int result = ttsEngine.setLanguage(Locale.US);
                    if (result != android.speech.tts.TextToSpeech.LANG_MISSING_DATA &&
                            result != android.speech.tts.TextToSpeech.LANG_NOT_SUPPORTED) {
                        isTtsReady = true;
                        android.util.Log.d("SmartHelmetAI", "TTS Voice Engine initialized successfully!");
                    }
                } else {
                    android.util.Log.e("SmartHelmetAI", "TTS Voice Engine initialization failed!");
                }
            }
        });
        requestSosPermissions();
    } // 🟢 Closes your onCreate method cleanly!


    @Override
    protected void onDestroy() {
        polling = false;
        streamOn = false;
        stopMjpegReader();
        handler.removeCallbacks(imuPoller);
        handler.removeCallbacks(backgroundGesturePoller);
        closeBluetooth();
        executor.shutdownNow();
        // Stop both independent streams
        stopPreview();
        stopPpeStream();
        // Stream threads are daemon threads — they terminate automatically with the app
        if (ttsEngine != null) {
            ttsEngine.stop();
            ttsEngine.shutdown();
            isTtsReady = false;
            android.util.Log.d("SmartHelmetAI", "TTS Engine safely released.");
        }
        if (tflite != null) {
            tflite.close();
            isModelLoaded = false;
            android.util.Log.d("SmartHelmetAI", "Stage 2 Model interpreter safely closed.");
        }
        if (mlKitDetector != null) {
            mlKitDetector.close();
            android.util.Log.d("SmartHelmetAI", "ML Kit Detector safely closed.");
        }
        if (mlKitFaceDetector != null) {
            mlKitFaceDetector.close();
        }
        if (paddlePredictorInstance != null) {
            try {
                paddlePredictorInstance.releaseModel();
            } catch (Exception ignored) {}
            paddlePredictorInstance = null;
        }
        super.onDestroy();
    }

    private void showWelcome() {
        getWindow().setStatusBarColor(Color.parseColor("#07378E"));

        LinearLayout welcome = new LinearLayout(this);
        welcome.setOrientation(LinearLayout.VERTICAL);
        welcome.setGravity(Gravity.CENTER);
        welcome.setPadding(dp(28), dp(24), dp(28), dp(34));
        welcome.setBackgroundColor(Color.parseColor("#173A61"));
        setContentView(welcome);

        TextView badge = iconCircle("SH", "#3B82F6", "#FFFFFF", 28);
        LinearLayout.LayoutParams badgeParams = new LinearLayout.LayoutParams(dp(96), dp(96));
        welcome.addView(badge, badgeParams);

        TextView title = text("Smart Helmet", 38, "#FFFFFF", Typeface.BOLD);
        title.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams titleParams = matchWrap();
        titleParams.setMargins(0, dp(28), 0, dp(8));
        welcome.addView(title, titleParams);

        TextView subtitle = text("Smart helmet command center", 17, "#B8C8D9", Typeface.BOLD);
        subtitle.setGravity(Gravity.CENTER);
        welcome.addView(subtitle, matchWrap());

        View spacer = new View(this);
        welcome.addView(spacer, new LinearLayout.LayoutParams(1, 0, 1));

        Button start = actionButton("Enter Dashboard", "#3B82F6", 0);
        LinearLayout.LayoutParams startParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(58)
        );
        welcome.addView(start, startParams);

        TextView footer = text("Wi-Fi camera link + Bluetooth pairing", 13, "#B8C8D9", Typeface.NORMAL);
        footer.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams footerParams = matchWrap();
        footerParams.setMargins(0, dp(18), 0, 0);
        welcome.addView(footer, footerParams);

        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("onboarded", true).apply();
                onboarded = true;
                buildAppShell();
            }
        });
    }

    private void buildAppShell() {
        getWindow().setStatusBarColor(Color.parseColor("#07378E"));

        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#F6F8FB"));
        setContentView(root);

        content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        root.addView(content, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1
        ));

        navBar = new LinearLayout(this);
        navBar.setOrientation(LinearLayout.HORIZONTAL);
        navBar.setGravity(Gravity.CENTER);
        navBar.setPadding(dp(8), dp(8), dp(8), dp(8));
        navBar.setBackgroundColor(Color.WHITE);
        android.widget.HorizontalScrollView navScroller = new android.widget.HorizontalScrollView(this);
        navScroller.setHorizontalScrollBarEnabled(false);
        navScroller.setFillViewport(false);
        navScroller.setBackgroundColor(Color.WHITE);
        navScroller.addView(navBar, new android.widget.HorizontalScrollView.LayoutParams(
                android.widget.HorizontalScrollView.LayoutParams.WRAP_CONTENT,
                android.widget.HorizontalScrollView.LayoutParams.MATCH_PARENT
        ));
        root.addView(navScroller, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(72)
        ));

        renderTab(activeTab);
    }

    private void renderTab(String tab) {
        String previousTab = activeTab;
        // Stop ONLY the stream belonging to the tab we are leaving
        if ("camera".equals(previousTab) && streamOn) stopPreview();
        if ("ppe".equals(previousTab) && ppeStreamOn) stopPpeStream();
        if ("live".equals(previousTab) && liveScanOn) stopLiveLocationScan();
        activeTab = tab;

        if (!"ppe".equals(tab)) {
            synchronized (latestDetections) {
                latestDetections.clear();
                cachedDetections.clear();
            }
            isPersonInFrame = false;
            alertNoHelmet = false;
            alertNoGoggles = false;
            alertNoMask = false;
            alertNoVest = false;
            alertMissingPpe = false;
            alertChinstrapOpen = false;
        }

        calibStatusText = null;
        calibOffsetsText = null;
        togglePollBtn = null;
        toggleRecordBtn = null;
        cubeView = null;
        logsListContainer = null;
        cameraLogsListContainer = null;
        ppeLogsListContainer = null;
        liveMappingsContainer = null;
        liveLogsContainer = null;
        liveScanStatus = null;
        liveCameraImage = null;
        livePreviewPlaceholder = null;
        liveScanButton = null;

        content.removeAllViews();
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        content.addView(scrollView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
        ));

        LinearLayout page = new LinearLayout(this);
        page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(18), statusBarHeight() + dp(14), dp(18), dp(18));
        scrollView.addView(page);

        if ("dashboard".equals(tab)) {
            buildDashboard(page);
        } else if ("camera".equals(tab)) {
            buildCamera(page);
        } else if ("ppe".equals(tab)) {
            buildPpe(page);
        } else if ("sensors".equals(tab)) {
            buildSensors(page);
        } else if ("posture".equals(tab)) {
            buildPosture(page);
            polling = true; // Auto-enable polling when entering posture tab
            handler.removeCallbacks(posturePoller);
            handler.post(posturePoller);
        } else if ("live".equals(tab)) {
            buildLiveLocation(page);
        } else {
            buildConnect(page);
        }

        buildNav();
    }

    private void buildDashboard(LinearLayout page) {
        LinearLayout hero = new LinearLayout(this);
        hero.setOrientation(LinearLayout.VERTICAL);
        hero.setPadding(dp(20), dp(22), dp(20), dp(20));
        hero.setBackground(round("#173A61", 8, "#173A61"));
        page.addView(hero, sectionParams());

        TextView appName = text("Smart Helmet", 29, "#FFFFFF", Typeface.BOLD);
        hero.addView(appName);

        TextView tag = text("Smart helmet dashboard", 14, "#B8C8D9", Typeface.NORMAL);
        hero.addView(tag);

        topStatus = pill("Ready", "#2A527B", "#EAF2FF");
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        statusParams.setMargins(0, dp(18), 0, 0);
        hero.addView(topStatus, statusParams);

        GridLayout quickGrid = new GridLayout(this);
        quickGrid.setColumnCount(2);
        page.addView(quickGrid, sectionParams());

        wifiStatus = dashboardTile(quickGrid, "Wi-Fi Link", lastWifiState, "#3B82F6");
        bluetoothStatus = dashboardTile(quickGrid, "Bluetooth", lastBluetoothState, "#0F766E");
        cameraState = dashboardTile(quickGrid, "Camera", streamOn ? "Live preview" : "Off", "#A855F7");
        recordingState = dashboardTile(quickGrid, "Recording", "Idle", "#D04F3B");

        Button connect = wideButton("Connect Helmet", "#3B82F6");
        page.addView(connect, sectionParams());
        connect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                renderTab("connect");
            }
        });

        LinearLayout actions = panel();
        page.addView(actions, sectionParams());
        actions.addView(sectionTitle("Quick Controls"));

        Button camera = wideButton("Open Camera Controls", "#173A61");
        Button sensors = wideButton("Open IMU Readings", "#0F766E");
        LinearLayout.LayoutParams buttonParams = matchWrap();
        buttonParams.setMargins(0, dp(12), 0, dp(8));
        actions.addView(camera, buttonParams);
        actions.addView(sensors, matchWrap());

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                renderTab("camera");
            }
        });
        sensors.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                renderTab("sensors");
            }
        });

        checkStatus(false);
    }

    private void buildCamera(LinearLayout page) {
        page.addView(pageHeader("Camera", "Live view, photo capture, and recording"));

        FrameLayout videoFrame = new FrameLayout(this);
        videoFrame.setBackground(round("#111827", 8, "#111827"));
        videoFrame.setPadding(dp(2), dp(2), dp(2), dp(2));
        LinearLayout.LayoutParams videoParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                getVideoHeight()
        );
        videoParams.setMargins(0, 0, 0, dp(14));
        page.addView(videoFrame, videoParams);

        cameraImage = new ImageView(this);
        cameraImage.setBackgroundColor(Color.BLACK);
        cameraImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        videoFrame.addView(cameraImage, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        previewPlaceholder = text("Camera preview", 17, "#CBD5E1", Typeface.BOLD);
        previewPlaceholder.setGravity(Gravity.CENTER);
        videoFrame.addView(previewPlaceholder, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        if (streamOn) {
            previewPlaceholder.setVisibility(View.GONE);
            startMjpegReader();
        }

        GridLayout controls = new GridLayout(this);
        controls.setColumnCount(2);
        page.addView(controls, sectionParams());

        cameraToggleButton = actionButton(streamOn ? "Camera Off" : "Camera On", "#3B82F6", 0);
        Button photo = actionButton("Take Picture", "#0F766E", R.drawable.ic_camera);
        Button recordOn = actionButton("Recording On", "#D04F3B", R.drawable.ic_record);
        Button recordOff = actionButton("Recording Off", "#475467", R.drawable.ic_stop);

        addGridButton(controls, cameraToggleButton);
        addGridButton(controls, photo);
        addGridButton(controls, recordOn);
        addGridButton(controls, recordOff);

        cameraToggleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (streamOn) {
                    stopPreview();
                } else {
                    startPreview();
                }
            }
        });

        photo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (streamOn) {
                    // Stop preview first to free up the camera
                    stopPreview();
                    Toast.makeText(MainActivity.this, "Pausing preview to capture photo", Toast.LENGTH_SHORT).show();
                    handler.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            // Take photo and restart preview when complete
                            callCommandWithCallback("/camera/photo", "Picture saved", new Runnable() {
                                @Override
                                public void run() {
                                    startPreview();
                                    Toast.makeText(MainActivity.this, "Preview resumed", Toast.LENGTH_SHORT).show();
                                }
                            });
                        }
                    }, 500);
                } else {
                    // Stream not active, just take the photo
                    runCameraCommand("/camera/photo", "Picture saved", false);
                }
            }
        });

        recordOn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!streamOn) {
                    startPreview();
                }
                runCameraCommand("/camera/record/start", "Recording started", false);
                if (recordingState != null) {
                    recordingState.setText("Active");
                }
            }
        });

        recordOff.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                runCameraCommand("/camera/record/stop", "Recording stopped", false);
                if (recordingState != null) {
                    recordingState.setText("Idle");
                }
            }
        });

        LinearLayout statusPanel = panel();
        page.addView(statusPanel, sectionParams());
        statusPanel.addView(sectionTitle("Camera State"));
        previewStatus = detailRow("Preview", streamOn ? "On" : "Off");
        statusPanel.addView(previewStatus);
        statusPanel.addView(detailRow("Saved media", "/root/smarthelmet/media"));
        statusPanel.addView(detailRow("Control channel", "Wi-Fi HTTP"));

        // Saved Photos & Videos Panel
        LinearLayout mediaPanel = panel();
        page.addView(mediaPanel, sectionParams());
        mediaPanel.addView(sectionTitle("Saved Photos & Videos"));

        cameraLogsListContainer = new LinearLayout(this);
        cameraLogsListContainer.setOrientation(LinearLayout.VERTICAL);
        mediaPanel.addView(cameraLogsListContainer, matchWrap());

        loadCameraSavedLogs();
    }

    private void buildSensors(LinearLayout page) {
        page.addView(pageHeader("IMU Sensor", "3D tracking, controls, and CSV logging"));

        // Control Panel
        LinearLayout controlPanel = panel();
        page.addView(controlPanel, sectionParams());
        controlPanel.addView(sectionTitle("Sensor Controls"));

        LinearLayout buttonsLayout = new LinearLayout(this);
        buttonsLayout.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams buttonsParams = matchWrap();
        buttonsParams.setMargins(0, dp(12), 0, 0);
        controlPanel.addView(buttonsLayout, buttonsParams);

        togglePollBtn = actionButton(polling ? "Stop Sensor Link" : "Start Sensor Link", polling ? "#D04F3B" : "#3B82F6", 0);
        toggleRecordBtn = actionButton(imuRecordingActive ? "Stop CSV Record" : "Start CSV Record", imuRecordingActive ? "#D04F3B" : "#0F766E", 0);
        toggleRecordBtn.setEnabled(polling);

        LinearLayout.LayoutParams btnParams = new LinearLayout.LayoutParams(0, dp(50), 1f);
        btnParams.setMargins(0, 0, dp(6), 0);
        buttonsLayout.addView(togglePollBtn, btnParams);

        LinearLayout.LayoutParams btnParams2 = new LinearLayout.LayoutParams(0, dp(50), 1f);
        btnParams2.setMargins(dp(6), 0, 0, 0);
        buttonsLayout.addView(toggleRecordBtn, btnParams2);

        togglePollBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (polling) {
                    polling = false;
                    handler.removeCallbacks(imuPoller);
                    togglePollBtn.setText("Start Sensor Link");
                    togglePollBtn.setBackground(round("#3B82F6", 8, "#3B82F6"));
                    toggleRecordBtn.setEnabled(false);
                    resetSensorMetrics();
                } else {
                    startPolling();
                    togglePollBtn.setText("Stop Sensor Link");
                    togglePollBtn.setBackground(round("#D04F3B", 8, "#D04F3B"));
                    toggleRecordBtn.setEnabled(true);
                }
            }
        });

        toggleRecordBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (imuRecordingActive) {
                    runSensorRecordCommand("/imu/record/stop", "IMU recording stopped");
                } else {
                    runSensorRecordCommand("/imu/record/start", "IMU recording started");
                }
            }
        });

        // 3D Motion Tracking Panel
        LinearLayout visualPanel = panel();
        page.addView(visualPanel, sectionParams());
        visualPanel.addView(sectionTitle("3D Motion Tracking"));

        cubeView = new Cube3DView(this);
        LinearLayout.LayoutParams cubeParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(180)
        );
        cubeParams.setMargins(0, dp(12), 0, 0);
        visualPanel.addView(cubeView, cubeParams);

        // Numeric readings Grid
        GridLayout sensorGrid = new GridLayout(this);
        sensorGrid.setColumnCount(2);
        page.addView(sensorGrid, sectionParams());

        accelX = addMetric(sensorGrid, "Accel X", "-- g", "#3B82F6");
        accelY = addMetric(sensorGrid, "Accel Y", "-- g", "#3B82F6");
        accelZ = addMetric(sensorGrid, "Accel Z", "-- g", "#3B82F6");
        gyroX = addMetric(sensorGrid, "Gyro X", "-- deg/s", "#0F766E");
        gyroY = addMetric(sensorGrid, "Gyro Y", "-- deg/s", "#0F766E");
        gyroZ = addMetric(sensorGrid, "Gyro Z", "-- deg/s", "#0F766E");

        altitudeText = addMetric(sensorGrid, "Altitude", "-- m", "#2563EB");

        // Calibration Panel
        LinearLayout calibPanel = panel();
        page.addView(calibPanel, sectionParams());
        calibPanel.addView(sectionTitle("Gyro Calibration"));
        calibStatusText = detailRow("Status", "Checking...");
        calibOffsetsText = detailRow("Offsets (X/Y/Z)", "-- / -- / --");
        calibPanel.addView(calibStatusText);
        calibPanel.addView(calibOffsetsText);

        // Saved CSV Records Panel
        LinearLayout logsPanel = panel();
        page.addView(logsPanel, sectionParams());
        logsPanel.addView(sectionTitle("Saved CSV Logs"));

        logsListContainer = new LinearLayout(this);
        logsListContainer.setOrientation(LinearLayout.VERTICAL);
        logsPanel.addView(logsListContainer, matchWrap());

        Button refresh = wideButton("Refresh Sensor Link", "#173A61");
        page.addView(refresh, sectionParams());
        refresh.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                checkStatus(true);
            }
        });

        if (polling) {
            startPolling();
        } else {
            resetSensorMetrics();
        }
        checkStatus(false);
        loadSavedLogs();
    }

    private void buildPosture(LinearLayout page) {
        page.addView(pageHeader("Posture & Safety", "Live helmet orientation and fall detection"));

        // ── Orientation card ───────────────────────────────────────────────
        LinearLayout orientPanel = panel();
        page.addView(orientPanel, sectionParams());
        orientPanel.addView(sectionTitle("Helmet Orientation"));

        GridLayout orientGrid = new GridLayout(this);
        orientGrid.setColumnCount(2);
        orientPanel.addView(orientGrid, matchWrap());
        postureOrientation = addMetric(orientGrid, "Direction",   "-- ",      "#3B82F6");
        posturePitch       = addMetric(orientGrid, "Pitch",       "-- °",     "#0F766E");
        postureRoll        = addMetric(orientGrid, "Roll",        "-- °",     "#0F766E");

        // ── Turn Direction card (gyro Z yaw) ──────────────────────────────
        LinearLayout turnPanel = panel();
        page.addView(turnPanel, sectionParams());
        turnPanel.addView(sectionTitle("Turn Direction (Gyroscope)"));
        postureTurnDir = new TextView(this);
        postureTurnDir.setText("STRAIGHT");
        postureTurnDir.setTextSize(22);
        postureTurnDir.setTypeface(Typeface.DEFAULT_BOLD);
        postureTurnDir.setTextColor(Color.parseColor("#16A34A"));
        postureTurnDir.setPadding(dp(4), dp(10), dp(4), dp(10));
        turnPanel.addView(postureTurnDir, matchWrap());

        // ── Fall Detection card ────────────────────────────────────────────
        LinearLayout fallPanel = panel();
        page.addView(fallPanel, sectionParams());
        fallPanel.addView(sectionTitle("Fall Detection"));

        postureFallState = new TextView(this);
        postureFallState.setText("NORMAL");
        postureFallState.setTextSize(22);
        postureFallState.setTypeface(Typeface.DEFAULT_BOLD);
        postureFallState.setTextColor(Color.parseColor("#16A34A"));
        postureFallState.setPadding(dp(4), dp(10), dp(4), dp(10));
        fallPanel.addView(postureFallState, matchWrap());

        // ── Altitude change card ───────────────────────────────────────────
        LinearLayout altPanel = panel();
        page.addView(altPanel, sectionParams());
        altPanel.addView(sectionTitle("Altitude Change (from boot)"));
        postureAltDelta = new TextView(this);
        postureAltDelta.setText("± 0.00 m");
        postureAltDelta.setTextSize(20);
        postureAltDelta.setTypeface(Typeface.DEFAULT_BOLD);
        postureAltDelta.setTextColor(Color.parseColor("#2563EB"));
        postureAltDelta.setPadding(dp(4), dp(10), dp(4), dp(10));
        altPanel.addView(postureAltDelta, matchWrap());
    }

    private void updatePosture(final JSONObject json) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                String orient = json.optString("orientation", "--");
                double pitch  = json.optDouble("pitch",  0.0);
                double roll   = json.optDouble("roll",   0.0);
                String turn   = json.optString("turn_direction", "FORWARD");
                double yawDps = json.optDouble("yaw_rate_dps", 0.0);
                String fall   = json.optString("fall_state", "NORMAL");
                String gest   = json.optString("last_gesture", "None");
                double altDel = json.optDouble("altitude_delta_m", 0.0);

                // ── SOS accident detection trigger ──────────────────────────
                if ("FALL DETECTED".equals(fall) && !"FALL DETECTED".equals(lastFallState) && !sosPending) {
                    showSosCountdown();
                }
                lastFallState = fall;

                if (postureOrientation != null) postureOrientation.setText(orient);
                if (posturePitch       != null) posturePitch.setText(String.format(java.util.Locale.US, "%.1f \u00b0", pitch));
                if (postureRoll        != null) postureRoll.setText(String.format(java.util.Locale.US, "%.1f \u00b0", roll));

                if (postureTurnDir != null) {
                    postureTurnDir.setText(String.format(java.util.Locale.US, "%s  (%.1f \u00b0/s)", turn, yawDps));
                    if ("LOOKING LEFT".equals(turn)) {
                        postureTurnDir.setTextColor(Color.parseColor("#2563EB"));
                    } else if ("LOOKING RIGHT".equals(turn)) {
                        postureTurnDir.setTextColor(Color.parseColor("#D97706"));
                    } else {
                        postureTurnDir.setTextColor(Color.parseColor("#16A34A"));
                    }
                }

                if (postureFallState != null) {
                    postureFallState.setText(fall);
                    if ("FALL DETECTED".equals(fall)) {
                        postureFallState.setTextColor(Color.parseColor("#DC2626"));
                    } else if ("FREE FALL".equals(fall) || "IMPACT".equals(fall)) {
                        postureFallState.setTextColor(Color.parseColor("#D97706"));
                    } else {
                        postureFallState.setTextColor(Color.parseColor("#16A34A"));
                    }
                }

                if (postureAltDelta != null) {
                    String sign = altDel >= 0 ? "+" : "";
                    postureAltDelta.setText(String.format(java.util.Locale.US, "%s%.2f m", sign, altDel));
                    postureAltDelta.setTextColor(Color.parseColor(altDel > 1.0 ? "#DC2626" : altDel < -1.0 ? "#2563EB" : "#16A34A"));
                }

                if (postureGesture != null && !"None".equals(gest)) {
                    postureGesture.setText(gest);
                }
            }
        });
    }

    // ──────────────────────────────────────────────────────────
    // SOS / Accident Detection Methods
    // ──────────────────────────────────────────────────────────

    private void requestSosPermissions() {
        java.util.List<String> needed = new java.util.ArrayList<>();
        if (checkSelfPermission(Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED)
            needed.add(Manifest.permission.SEND_SMS);
        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED)
            needed.add(Manifest.permission.ACCESS_FINE_LOCATION);
        if (!needed.isEmpty())
            requestPermissions(needed.toArray(new String[0]), REQ_SMS_LOC);
    }

    private void showSosCountdown() {
        sosPending = true;
        sosLocationReady = false;

        // Start fetching location immediately in parallel
        startLocationFetch();

        // TTS voice alert
        if (isTtsReady) {
            ttsEngine.speak("Accident detected! Sending SOS in 15 seconds. Press cancel to abort.",
                    android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, null);
        }

        // Build countdown dialog UI
        final TextView countdown = new TextView(this);
        countdown.setGravity(Gravity.CENTER);
        countdown.setTextSize(48);
        countdown.setTypeface(Typeface.DEFAULT_BOLD);
        countdown.setTextColor(Color.parseColor("#DC2626"));
        countdown.setText("15");

        TextView msg = new TextView(this);
        msg.setText("\ud83d\udea8 ACCIDENT DETECTED\n\nSending SOS SMS with live location in:");
        msg.setGravity(Gravity.CENTER);
        msg.setTextSize(15);
        msg.setTypeface(Typeface.DEFAULT_BOLD);
        msg.setTextColor(Color.parseColor("#17212B"));
        msg.setPadding(0, 0, 0, dp(8));

        LinearLayout dialogBody = new LinearLayout(this);
        dialogBody.setOrientation(LinearLayout.VERTICAL);
        dialogBody.setPadding(dp(24), dp(20), dp(24), dp(8));
        dialogBody.addView(msg);
        dialogBody.addView(countdown);

        final AlertDialog dialog = new AlertDialog.Builder(this)
                .setView(dialogBody)
                .setCancelable(false)
                .setNegativeButton("\u274c CANCEL SOS", null)
                .create();

        dialog.setOnShowListener(new DialogInterface.OnShowListener() {
            @Override
            public void onShow(DialogInterface di) {
                dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setTextColor(Color.parseColor("#DC2626"));
            }
        });

        sosTimer = new CountDownTimer(15000, 1000) {
            @Override
            public void onTick(long ms) {
                countdown.setText(String.valueOf(ms / 1000 + 1));
            }
            @Override
            public void onFinish() {
                dialog.dismiss();
                sosPending = false;
                fetchLocationAndSendSms();
            }
        }.start();

        dialog.show();

        // Wire cancel button AFTER show()
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sosTimer.cancel();
                sosPending = false;
                dialog.dismiss();
                Toast.makeText(MainActivity.this, "SOS cancelled", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void startLocationFetch() {
        try {
            LocationManager lm = (LocationManager) getSystemService(LOCATION_SERVICE);
            if (lm == null) return;

            // Try last known location first (instant)
            if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                Location last = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER);
                if (last == null) last = lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                if (last != null) {
                    sosLat = last.getLatitude();
                    sosLng = last.getLongitude();
                    sosLocationReady = true;
                }
                // Also request fresh fix
                lm.requestSingleUpdate(LocationManager.GPS_PROVIDER, new LocationListener() {
                    @Override public void onLocationChanged(Location loc) {
                        sosLat = loc.getLatitude();
                        sosLng = loc.getLongitude();
                        sosLocationReady = true;
                    }
                    @Override public void onStatusChanged(String p, int s, Bundle e) {}
                    @Override public void onProviderEnabled(String p) {}
                    @Override public void onProviderDisabled(String p) {}
                }, Looper.getMainLooper());
            }
        } catch (Exception e) {
            android.util.Log.e("SOS", "Location fetch failed: " + e.getMessage());
        }
    }

    private void fetchLocationAndSendSms() {
        if (sosLocationReady) {
            sendSosSms(sosLat, sosLng);
        } else {
            // Try one more time with last known before giving up
            try {
                if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                    LocationManager lm = (LocationManager) getSystemService(LOCATION_SERVICE);
                    Location last = lm != null ? lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER) : null;
                    if (last != null) {
                        sendSosSms(last.getLatitude(), last.getLongitude());
                        return;
                    }
                }
            } catch (Exception ignored) {}
            sendSosSms(0, 0);  // send without coords
        }
    }

    private void sendSosSms(double lat, double lng) {
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String emergencyNumber = prefs.getString("sos_number", "");

        if (emergencyNumber.isEmpty()) {
            runOnUiThread(new Runnable() { @Override public void run() {
                Toast.makeText(MainActivity.this,
                        "No emergency number set! Go to Connect tab > SOS Settings",
                        Toast.LENGTH_LONG).show();
            }});
            return;
        }

        String time = new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss", Locale.getDefault()).format(new Date());
        String locLine;
        if (lat != 0 || lng != 0) {
            locLine = "Live Location: https://maps.google.com/?q=" + lat + "," + lng;
        } else {
            locLine = "Location unavailable (GPS off or indoors)";
        }

        String message =
                "\ud83d\udea8 SOS ALERT - Smart Helmet\n" +
                "Accident detected on the rider's helmet!\n" +
                "The wearer may need immediate help.\n" +
                locLine + "\n" +
                "Time: " + time;

        try {
            if (checkSelfPermission(Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED) {
                SmsManager sms = SmsManager.getDefault();
                java.util.ArrayList<String> parts = sms.divideMessage(message);
                sms.sendMultipartTextMessage(emergencyNumber, null, parts, null, null);
                runOnUiThread(new Runnable() { @Override public void run() {
                    Toast.makeText(MainActivity.this, "🚨 SOS sent to " + emergencyNumber, Toast.LENGTH_LONG).show();
                }});
            } else {
                // Fallback: Open system Messages app pre-filled with SOS text
                final String finalMsg = message;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            android.content.Intent intent = new android.content.Intent(android.content.Intent.ACTION_SENDTO);
                            intent.setData(android.net.Uri.parse("smsto:" + android.net.Uri.encode(emergencyNumber)));
                            intent.putExtra("sms_body", finalMsg);
                            startActivity(intent);
                            Toast.makeText(MainActivity.this, "Opening SMS app to send SOS...", Toast.LENGTH_LONG).show();
                        } catch (Exception e) {
                            Toast.makeText(MainActivity.this, "SMS permission denied. Grant it in Phone Settings > Apps > Smart Helmet", Toast.LENGTH_LONG).show();
                        }
                    }
                });
            }
        } catch (Exception e) {
            android.util.Log.e("SOS", "SMS send failed: " + e.getMessage());
            // Fallback to Intent on SmsManager exception
            final String finalMsg = message;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        android.content.Intent intent = new android.content.Intent(android.content.Intent.ACTION_SENDTO);
                        intent.setData(android.net.Uri.parse("smsto:" + android.net.Uri.encode(emergencyNumber)));
                        intent.putExtra("sms_body", finalMsg);
                        startActivity(intent);
                    } catch (Exception ignored) {}
                }
            });
        }
    }

    private void buildConnect(LinearLayout page) {
        page.addView(pageHeader("Connect Helmet", "Choose Wi-Fi control or Bluetooth pairing"));

        LinearLayout wifiPanel = panel();
        page.addView(wifiPanel, sectionParams());
        wifiPanel.addView(sectionTitle("Wi-Fi Control"));

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        hostInput = new EditText(this);
        hostInput.setSingleLine(true);
        hostInput.setText(prefs.getString("host", DEFAULT_HOST));
        hostInput.setTextColor(Color.parseColor("#17212B"));
        hostInput.setTextSize(15);
        hostInput.setHint("Pi IP:5000");
        hostInput.setBackground(inputBackground());
        hostInput.setPadding(dp(12), 0, dp(12), 0);
        LinearLayout.LayoutParams inputParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(52)
        );
        inputParams.setMargins(0, dp(12), 0, dp(10));
        wifiPanel.addView(hostInput, inputParams);

        Button autoDiscover = wideButton("Auto-Discover Helmet IP", "#10B981");
        LinearLayout.LayoutParams autoParams = matchWrap();
        autoParams.setMargins(0, dp(6), 0, dp(10));
        wifiPanel.addView(autoDiscover, autoParams);

        autoDiscover.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                performUdpDiscovery(true);
            }
        });

        Button wifiConnect = wideButton("Connect Wi-Fi", "#3B82F6");
        wifiPanel.addView(wifiConnect, matchWrap());
        wifiPanel.addView(detailRow("Server", "Auto-discovered or manual IP"));

        wifiConnect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                saveHost();
                checkStatus(true);
            }
        });

        LinearLayout bluetoothPanel = panel();
        page.addView(bluetoothPanel, sectionParams());
        bluetoothPanel.addView(sectionTitle("Bluetooth Pairing"));

        TextView btNote = text("Paired Raspberry Pi devices appear here.", 13, "#667085", Typeface.NORMAL);
        LinearLayout.LayoutParams noteParams = matchWrap();
        noteParams.setMargins(0, dp(8), 0, dp(12));
        bluetoothPanel.addView(btNote, noteParams);

        Button scan = wideButton("Show Paired Devices", "#0F766E");
        bluetoothPanel.addView(scan, matchWrap());

        LinearLayout deviceList = new LinearLayout(this);
        deviceList.setOrientation(LinearLayout.VERTICAL);
        bluetoothPanel.addView(deviceList, matchWrap());

        scan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                listPairedDevices(deviceList);
            }
        });

        // ── SOS Settings Panel ────────────────────────────────────────────
        LinearLayout sosPanel = panel();
        page.addView(sosPanel, sectionParams());
        sosPanel.addView(sectionTitle("🚨 SOS Emergency Settings"));

        TextView sosNote = text("Set a number to call/SMS when an accident is detected by the helmet.", 13, "#667085", Typeface.NORMAL);
        LinearLayout.LayoutParams sosNoteParams = matchWrap();
        sosNoteParams.setMargins(0, dp(6), 0, dp(10));
        sosPanel.addView(sosNote, sosNoteParams);

        final SharedPreferences sosprefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        final EditText sosInput = new EditText(this);
        sosInput.setSingleLine(true);
        sosInput.setInputType(android.text.InputType.TYPE_CLASS_PHONE);
        sosInput.setText(sosprefs.getString("sos_number", ""));
        sosInput.setTextColor(Color.parseColor("#17212B"));
        sosInput.setTextSize(15);
        sosInput.setHint("+91XXXXXXXXXX");
        sosInput.setBackground(inputBackground());
        sosInput.setPadding(dp(12), 0, dp(12), 0);
        LinearLayout.LayoutParams sosInputParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(52));
        sosInputParams.setMargins(0, 0, 0, dp(10));
        sosPanel.addView(sosInput, sosInputParams);

        Button saveNum = wideButton("Save Emergency Number", "#DC2626");
        sosPanel.addView(saveNum, matchWrap());
        saveNum.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String num = sosInput.getText().toString().trim();
                sosprefs.edit().putString("sos_number", num).apply();
                Toast.makeText(MainActivity.this, "Emergency number saved: " + num, Toast.LENGTH_SHORT).show();
                requestSosPermissions();
            }
        });

        LinearLayout.LayoutParams testBtnParams = matchWrap();
        testBtnParams.setMargins(0, dp(8), 0, 0);
        Button testSos = wideButton("Send Test SOS Now", "#7C3AED");
        sosPanel.addView(testSos, testBtnParams);
        testSos.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sosprefs.edit().putString("sos_number", sosInput.getText().toString().trim()).apply();
                fetchLocationAndSendSms();
            }
        });
    }

    private void buildPpe(LinearLayout page) {
        page.addView(pageHeader("PPE Compliance", "Real-time AI safety equipment check"));

        FrameLayout videoFrame = new FrameLayout(this);
        videoFrame.setBackground(round("#111827", 8, "#111827"));
        videoFrame.setPadding(dp(2), dp(2), dp(2), dp(2));
        LinearLayout.LayoutParams videoParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                getVideoHeight()
        );
        videoParams.setMargins(0, 0, 0, dp(14));
        page.addView(videoFrame, videoParams);

        cameraImage = new ImageView(this);
        cameraImage.setBackgroundColor(Color.BLACK);
        cameraImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        videoFrame.addView(cameraImage, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        previewPlaceholder = text("Camera preview", 17, "#CBD5E1", Typeface.BOLD);
        previewPlaceholder.setGravity(Gravity.CENTER);
        videoFrame.addView(previewPlaceholder, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        // Stream is always stopped when entering any tab.
        // User must press "PPE Check On" to start.

        // Trigger loading the model asynchronously if it isn't already loaded
        if (!isModelLoaded) {
            previewPlaceholder.setText("Loading PPE AI model...");
            executor.execute(new Runnable() {
                @Override
                public void run() {
                    loadPpeModelIfNeeded();
                }
            });
        }

        cameraToggleButton = wideButton(ppeStreamOn ? "PPE Check Off" : "PPE Check On", "#3B82F6");
        LinearLayout.LayoutParams toggleParams = matchWrap();
        toggleParams.setMargins(0, 0, 0, dp(14));
        page.addView(cameraToggleButton, toggleParams);

        cameraToggleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (ppeStreamOn) {
                    stopPpeStream();
                } else {
                    startPpeStream();
                }
            }
        });

        // Let's create a beautiful PPE Status panel displaying real-time compliance results
        LinearLayout statusPanel = panel();
        page.addView(statusPanel, sectionParams());
        statusPanel.addView(sectionTitle("Real-time Compliance"));
        
        previewStatus = detailRow("Inference status", isModelLoaded ? "AI Model Ready" : "Loading model...");
        statusPanel.addView(previewStatus);

        LinearLayout logsPanel = panel();
        page.addView(logsPanel, sectionParams());
        logsPanel.addView(sectionTitle("Saved PPE Logs"));

        ppeLogsListContainer = new LinearLayout(this);
        ppeLogsListContainer.setOrientation(LinearLayout.VERTICAL);
        logsPanel.addView(ppeLogsListContainer, matchWrap());
        loadSavedPpeLogs();
    }

    private void loadPpeModelIfNeeded() {
        if (isModelLoaded && tflite != null) {
            return;
        }
        try {
            // Load TFLite interpreter
            MappedByteBuffer modelFile = FileUtil.loadMappedFile(this, "best_calibrated_model.tflite");
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            tflite = new Interpreter(modelFile, options);
            
            // Initialize ML Kit detector if null
            if (mlKitDetector == null) {
                com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions detectorOptions =
                        new com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions.Builder()
                                .setDetectorMode(com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                                .enableMultipleObjects()
                                .enableClassification() // Enable classification to distinguish people from furniture
                                .build();
                mlKitDetector = com.google.mlkit.vision.objects.ObjectDetection.getClient(detectorOptions);
            }

            if (mlKitFaceDetector == null) {
                com.google.mlkit.vision.face.FaceDetectorOptions faceOptions =
                        new com.google.mlkit.vision.face.FaceDetectorOptions.Builder()
                                .setPerformanceMode(com.google.mlkit.vision.face.FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                                .setMinFaceSize(0.08f)
                                .build();
                mlKitFaceDetector = com.google.mlkit.vision.face.FaceDetection.getClient(faceOptions);
            }
            
            isModelLoaded = true;
            android.util.Log.d("SmartHelmetAI", "Stage 2 PPE Model + ML Kit initialized!");
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, "PPE AI Model loaded successfully!", Toast.LENGTH_SHORT).show();
                    if (previewPlaceholder != null && !ppeStreamOn) {
                        previewPlaceholder.setText("Camera preview");
                    }
                    if (previewStatus != null) {
                        previewStatus.setText(ppeStreamOn ? "Waiting for person" : "AI Model Ready");
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
            isModelLoaded = false;
            android.util.Log.e("SmartHelmetAI", "Failed to load PPE AI components: " + e.getMessage());
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, "Failed to load PPE AI components", Toast.LENGTH_LONG).show();
                    if (previewPlaceholder != null) {
                        previewPlaceholder.setText("Failed to load AI model");
                    }
                }
            });
        }
    }

    private void buildLiveLocation(LinearLayout page) {
        page.addView(pageHeader("Live Location", "OCR code detection mapped to saved helmet locations"));

        FrameLayout videoFrame = new FrameLayout(this);
        videoFrame.setBackground(round("#111827", 8, "#111827"));
        videoFrame.setPadding(dp(2), dp(2), dp(2), dp(2));
        LinearLayout.LayoutParams videoParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                getVideoHeight()
        );
        videoParams.setMargins(0, 0, 0, dp(14));
        page.addView(videoFrame, videoParams);

        liveCameraImage = new ImageView(this);
        liveCameraImage.setBackgroundColor(Color.BLACK);
        liveCameraImage.setScaleType(ImageView.ScaleType.FIT_CENTER);
        videoFrame.addView(liveCameraImage, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        livePreviewPlaceholder = text("Live location OCR preview", 17, "#CBD5E1", Typeface.BOLD);
        livePreviewPlaceholder.setGravity(Gravity.CENTER);
        videoFrame.addView(livePreviewPlaceholder, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        liveScanButton = wideButton(liveScanOn ? "Stop Code Scan" : "Start Code Scan", "#3B82F6");
        LinearLayout.LayoutParams scanBtnParams = matchWrap();
        scanBtnParams.setMargins(0, 0, 0, dp(14));
        page.addView(liveScanButton, scanBtnParams);
        liveScanButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (liveScanOn) {
                    stopLiveLocationScan();
                } else {
                    startLiveLocationScan();
                }
            }
        });

        LinearLayout statusPanel = panel();
        page.addView(statusPanel, sectionParams());
        statusPanel.addView(sectionTitle("Scan Status"));
        liveScanStatus = detailRow("Engine", isPaddleOcrReady() ? "Paddle OCR Ready" : "Paddle OCR assets missing");
        statusPanel.addView(liveScanStatus);

        LinearLayout mappingPanel = panel();
        page.addView(mappingPanel, sectionParams());
        mappingPanel.addView(sectionTitle("Saved Code Location"));

        liveCodeInput = input("Code e.g. A25-2E&^&HD");
        liveLocationInput = input("Location name e.g. DRI Section 1");
        livePhoneInput = input("Phone number for SMS");
        mappingPanel.addView(liveCodeInput, inputParams());
        mappingPanel.addView(liveLocationInput, inputParams());
        mappingPanel.addView(livePhoneInput, inputParams());

        Button saveMapping = wideButton("Save Code Location", "#0F766E");
        LinearLayout.LayoutParams saveParams = matchWrap();
        saveParams.setMargins(0, dp(8), 0, 0);
        mappingPanel.addView(saveMapping, saveParams);
        saveMapping.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveLiveLocationMapping();
            }
        });

        liveMappingsContainer = new LinearLayout(this);
        liveMappingsContainer.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams mappingsParams = matchWrap();
        mappingsParams.setMargins(0, dp(12), 0, 0);
        mappingPanel.addView(liveMappingsContainer, mappingsParams);
        loadLiveLocationMappings();

        LinearLayout logsPanel = panel();
        page.addView(logsPanel, sectionParams());
        logsPanel.addView(sectionTitle("Detection Logs"));
        liveLogsContainer = new LinearLayout(this);
        liveLogsContainer.setOrientation(LinearLayout.VERTICAL);
        logsPanel.addView(liveLogsContainer, matchWrap());
        loadLiveLocationLogs();
    }

    private EditText input(String hint) {
        EditText editText = new EditText(this);
        editText.setSingleLine(true);
        editText.setTextColor(Color.parseColor("#17212B"));
        editText.setTextSize(15);
        editText.setHint(hint);
        editText.setBackground(inputBackground());
        editText.setPadding(dp(12), 0, dp(12), 0);
        return editText;
    }

    private LinearLayout.LayoutParams inputParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(52)
        );
        params.setMargins(0, dp(10), 0, 0);
        return params;
    }

    private JSONArray getLiveLocationMappings() {
        String raw = getSharedPreferences(PREFS, MODE_PRIVATE).getString("live_location_mappings", "[]");
        try {
            return new JSONArray(raw);
        } catch (Exception e) {
            return new JSONArray();
        }
    }

    private void saveLiveLocationMappings(JSONArray mappings) {
        getSharedPreferences(PREFS, MODE_PRIVATE)
                .edit()
                .putString("live_location_mappings", mappings.toString())
                .apply();
    }

    private void saveLiveLocationMapping() {
        String rawCode = liveCodeInput != null ? liveCodeInput.getText().toString().trim() : "";
        String normalizedCode = normalizeLiveCode(rawCode);
        String location = liveLocationInput != null ? liveLocationInput.getText().toString().trim() : "";
        String phone = livePhoneInput != null ? livePhoneInput.getText().toString().trim() : "";

        if (normalizedCode.isEmpty() || location.isEmpty() || phone.isEmpty()) {
            Toast.makeText(this, "Enter code, location, and phone number", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            JSONArray mappings = getLiveLocationMappings();
            JSONObject item = null;
            for (int i = 0; i < mappings.length(); i++) {
                JSONObject existing = mappings.optJSONObject(i);
                if (existing != null && normalizedCode.equals(existing.optString("normalizedCode"))) {
                    item = existing;
                    break;
                }
            }
            if (item == null) {
                item = new JSONObject();
                mappings.put(item);
            }
            item.put("code", rawCode);
            item.put("normalizedCode", normalizedCode);
            item.put("location", location);
            item.put("phone", phone);
            saveLiveLocationMappings(mappings);

            liveCodeInput.setText("");
            liveLocationInput.setText("");
            livePhoneInput.setText("");
            Toast.makeText(this, "Code location saved", Toast.LENGTH_SHORT).show();
            loadLiveLocationMappings();
        } catch (Exception e) {
            Toast.makeText(this, "Failed to save mapping", Toast.LENGTH_SHORT).show();
        }
    }

    private void loadLiveLocationMappings() {
        if (liveMappingsContainer == null) return;
        liveMappingsContainer.removeAllViews();
        final JSONArray mappings = getLiveLocationMappings();
        if (mappings.length() == 0) {
            TextView empty = text("No code locations saved", 14, "#94A3B8", Typeface.NORMAL);
            liveMappingsContainer.addView(empty, matchWrap());
            return;
        }

        for (int i = 0; i < mappings.length(); i++) {
            final int index = i;
            JSONObject item = mappings.optJSONObject(i);
            if (item == null) continue;
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.VERTICAL);
            LinearLayout.LayoutParams rowParams = matchWrap();
            rowParams.setMargins(0, dp(8), 0, dp(8));
            liveMappingsContainer.addView(row, rowParams);

            TextView title = text(item.optString("code") + "  ->  " + item.optString("location"), 14, "#17212B", Typeface.BOLD);
            row.addView(title, matchWrap());
            TextView phone = text("SMS: " + item.optString("phone"), 12, "#64748B", Typeface.NORMAL);
            row.addView(phone, matchWrap());

            Button delete = wideButton("Delete Mapping", "#DC2626");
            LinearLayout.LayoutParams deleteParams = matchWrap();
            deleteParams.setMargins(0, dp(6), 0, 0);
            row.addView(delete, deleteParams);
            delete.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    JSONArray current = getLiveLocationMappings();
                    JSONArray next = new JSONArray();
                    for (int j = 0; j < current.length(); j++) {
                        if (j != index) next.put(current.optJSONObject(j));
                    }
                    saveLiveLocationMappings(next);
                    loadLiveLocationMappings();
                }
            });

            View divider = new View(this);
            divider.setBackgroundColor(Color.parseColor("#E2E8F0"));
            liveMappingsContainer.addView(divider, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)));
        }
    }

    private JSONObject findLiveLocationMapping(String recognizedText) {
        String normalized = normalizeLiveCode(recognizedText);
        if (normalized.isEmpty()) return null;
        JSONArray mappings = getLiveLocationMappings();
        for (int i = 0; i < mappings.length(); i++) {
            JSONObject item = mappings.optJSONObject(i);
            if (item != null && normalized.equals(item.optString("normalizedCode"))) {
                return item;
            }
        }
        return null;
    }

    private String normalizeLiveCode(String value) {
        if (value == null) return "";
        String upper = value.toUpperCase(Locale.US);
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < upper.length(); i++) {
            char c = upper.charAt(i);
            if (!Character.isWhitespace(c)) {
                out.append(c);
            }
        }
        return out.toString();
    }

    private java.io.File getLiveLocationLogsDir() {
        java.io.File dir = new java.io.File(getFilesDir(), "live_location_logs");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    private void saveLiveLocationLog(String code, String location, String phone, boolean smsSent, String note) {
        try {
            Date now = new Date();
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(now);
            String humanTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(now);
            java.io.File log = new java.io.File(getLiveLocationLogsDir(), "live_location_" + stamp + ".txt");
            StringBuilder body = new StringBuilder();
            body.append("Smart Helmet Live Location OCR Log\n");
            body.append("Time: ").append(humanTime).append("\n");
            body.append("Detected Code: ").append(code).append("\n");
            body.append("Mapped Location: ").append(location).append("\n");
            body.append("Phone: ").append(phone).append("\n");
            body.append("SMS Sent: ").append(smsSent ? "Yes" : "No").append("\n");
            body.append("Note: ").append(note).append("\n");

            java.io.FileOutputStream output = new java.io.FileOutputStream(log);
            output.write(body.toString().getBytes("UTF-8"));
            output.close();

            handler.post(new Runnable() {
                @Override
                public void run() {
                    loadLiveLocationLogs();
                }
            });
        } catch (Exception ignored) {
        }
    }

    private void loadLiveLocationLogs() {
        if (liveLogsContainer == null) return;
        liveLogsContainer.removeAllViews();
        java.io.File[] files = getLiveLocationLogsDir().listFiles(new java.io.FilenameFilter() {
            @Override
            public boolean accept(java.io.File dir, String name) {
                return name != null && name.endsWith(".txt");
            }
        });
        if (files == null || files.length == 0) {
            liveLogsContainer.addView(text("No live location logs found", 14, "#94A3B8", Typeface.NORMAL), matchWrap());
            return;
        }
        java.util.Arrays.sort(files, new java.util.Comparator<java.io.File>() {
            @Override
            public int compare(java.io.File a, java.io.File b) {
                return Long.compare(b.lastModified(), a.lastModified());
            }
        });
        for (final java.io.File file : files) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.VERTICAL);
            LinearLayout.LayoutParams rowParams = matchWrap();
            rowParams.setMargins(0, dp(8), 0, dp(8));
            liveLogsContainer.addView(row, rowParams);
            row.addView(text(file.getName(), 14, "#17212B", Typeface.BOLD), matchWrap());

            LinearLayout actions = new LinearLayout(this);
            actions.setOrientation(LinearLayout.HORIZONTAL);
            LinearLayout.LayoutParams actionsParams = matchWrap();
            actionsParams.setMargins(0, dp(6), 0, 0);
            row.addView(actions, actionsParams);

            Button view = wideButton("View", "#0F766E");
            Button download = wideButton("Download", "#3B82F6");
            Button delete = wideButton("Delete", "#DC2626");
            actions.addView(view, new LinearLayout.LayoutParams(0, dp(42), 1f));
            actions.addView(download, new LinearLayout.LayoutParams(0, dp(42), 1f));
            actions.addView(delete, new LinearLayout.LayoutParams(0, dp(42), 1f));
            view.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) { showPpeLog(file); }
            });
            download.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) { exportLocalPpeLogToDownloads(file); }
            });
            delete.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) {
                    if (file.delete()) loadLiveLocationLogs();
                }
            });
        }
    }

    private void sendLiveLocationSms(JSONObject mapping, String detectedCode) {
        final String phone = mapping.optString("phone", "");
        final String location = mapping.optString("location", "");
        final String code = mapping.optString("code", detectedCode);
        if (phone.isEmpty() || location.isEmpty()) return;
        if (checkSelfPermission(Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.SEND_SMS}, REQ_SMS_LOC);
            saveLiveLocationLog(code, location, phone, false, "SMS permission missing");
            return;
        }
        String message = "Smart Helmet Live Location\nCode Detected: " + code + "\nLocation: " + location;
        try {
            SmsManager sms = SmsManager.getDefault();
            java.util.ArrayList<String> parts = sms.divideMessage(message);
            sms.sendMultipartTextMessage(phone, null, parts, null, null);
            saveLiveLocationLog(code, location, phone, true, "Sent by OCR code match");
            Toast.makeText(this, "Live location SMS sent: " + location, Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            saveLiveLocationLog(code, location, phone, false, e.getMessage() != null ? e.getMessage() : "SMS failed");
        }
    }

    private java.io.File getPpeReportsDir() {
        java.io.File dir = new java.io.File(getFilesDir(), "ppe_reports");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    private void loadSavedPpeLogs() {
        if (ppeLogsListContainer == null) return;
        ppeLogsListContainer.removeAllViews();

        java.io.File[] files = getPpeReportsDir().listFiles(new java.io.FilenameFilter() {
            @Override
            public boolean accept(java.io.File dir, String name) {
                return name != null && name.endsWith(".txt");
            }
        });

        if (files == null || files.length == 0) {
            TextView empty = new TextView(MainActivity.this);
            empty.setText("No saved PPE logs found");
            empty.setTextColor(Color.parseColor("#94A3B8"));
            empty.setTextSize(14);
            ppeLogsListContainer.addView(empty, matchWrap());
            return;
        }

        java.util.Arrays.sort(files, new java.util.Comparator<java.io.File>() {
            @Override
            public int compare(java.io.File a, java.io.File b) {
                return Long.compare(b.lastModified(), a.lastModified());
            }
        });

        for (final java.io.File reportFile : files) {
            double kb = reportFile.length() / 1024.0;
            String sizeStr = String.format(Locale.US, "%.1f KB", kb);

            LinearLayout row = new LinearLayout(MainActivity.this);
            row.setOrientation(LinearLayout.VERTICAL);
            LinearLayout.LayoutParams rowParams = matchWrap();
            rowParams.setMargins(0, dp(8), 0, dp(8));
            row.setLayoutParams(rowParams);

            LinearLayout metaRow = new LinearLayout(MainActivity.this);
            metaRow.setOrientation(LinearLayout.HORIZONTAL);

            TextView nameText = new TextView(MainActivity.this);
            nameText.setText(reportFile.getName());
            nameText.setTextColor(Color.parseColor("#1E293B"));
            nameText.setTextSize(14);
            nameText.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
            LinearLayout.LayoutParams nameParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
            metaRow.addView(nameText, nameParams);

            TextView sizeText = new TextView(MainActivity.this);
            sizeText.setText(sizeStr);
            sizeText.setTextColor(Color.parseColor("#64748B"));
            sizeText.setTextSize(12);
            metaRow.addView(sizeText, wrapWrap());
            row.addView(metaRow, matchWrap());

            LinearLayout actionRow = new LinearLayout(MainActivity.this);
            actionRow.setOrientation(LinearLayout.HORIZONTAL);
            LinearLayout.LayoutParams actionParams = matchWrap();
            actionParams.setMargins(0, dp(8), 0, 0);
            row.addView(actionRow, actionParams);

            Button viewBtn = new Button(MainActivity.this);
            viewBtn.setText("View");
            viewBtn.setTextColor(Color.WHITE);
            viewBtn.setTextSize(11);
            viewBtn.setAllCaps(false);
            viewBtn.setPadding(dp(8), 0, dp(8), 0);
            viewBtn.setBackground(round("#0F766E", 6, "#0F766E"));
            viewBtn.setMinHeight(0);
            viewBtn.setMinimumHeight(0);
            LinearLayout.LayoutParams viewParams = wrapWrap();
            viewParams.setMargins(dp(10), 0, 0, 0);
            viewBtn.setLayoutParams(viewParams);
            viewBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showPpeLog(reportFile);
                }
            });
            actionRow.addView(viewBtn);

            Button dlBtn = new Button(MainActivity.this);
            dlBtn.setText("Download");
            dlBtn.setTextColor(Color.WHITE);
            dlBtn.setTextSize(11);
            dlBtn.setAllCaps(false);
            dlBtn.setPadding(dp(8), 0, dp(8), 0);
            dlBtn.setBackground(round("#3B82F6", 6, "#3B82F6"));
            dlBtn.setMinHeight(0);
            dlBtn.setMinimumHeight(0);
            LinearLayout.LayoutParams btnParams = wrapWrap();
            btnParams.setMargins(dp(10), 0, 0, 0);
            dlBtn.setLayoutParams(btnParams);
            dlBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    exportLocalPpeLogToDownloads(reportFile);
                }
            });
            actionRow.addView(dlBtn);

            Button deleteBtn = new Button(MainActivity.this);
            deleteBtn.setText("Delete");
            deleteBtn.setTextColor(Color.WHITE);
            deleteBtn.setTextSize(11);
            deleteBtn.setAllCaps(false);
            deleteBtn.setPadding(dp(8), 0, dp(8), 0);
            deleteBtn.setBackground(round("#DC2626", 6, "#DC2626"));
            deleteBtn.setMinHeight(0);
            deleteBtn.setMinimumHeight(0);
            LinearLayout.LayoutParams deleteParams = wrapWrap();
            deleteParams.setMargins(dp(10), 0, 0, 0);
            deleteBtn.setLayoutParams(deleteParams);
            deleteBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    deletePpeLog(reportFile);
                }
            });
            actionRow.addView(deleteBtn);

            ppeLogsListContainer.addView(row);

            View divider = new View(MainActivity.this);
            divider.setBackgroundColor(Color.parseColor("#E2E8F0"));
            LinearLayout.LayoutParams divParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1));
            ppeLogsListContainer.addView(divider, divParams);
        }
    }

    private void savePpeScanReport(final java.util.List<String> violations, final String spokenResult) {
        try {
            SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
            int nextPerson = prefs.getInt("ppe_person_report_counter", 0) + 1;
            prefs.edit().putInt("ppe_person_report_counter", nextPerson).apply();
            ppePersonReportCounter = nextPerson;

            Date now = new Date();
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(now);
            String humanTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(now);
            String filename = String.format(Locale.US, "ppe_person_%03d_%s.txt", nextPerson, stamp);
            java.io.File reportFile = new java.io.File(getPpeReportsDir(), filename);

            StringBuilder report = new StringBuilder();
            report.append("Person ").append(nextPerson).append(" PPE Report\n");
            report.append("Time: ").append(humanTime).append("\n");
            report.append("Result: ").append(violations.isEmpty() ? "PASS" : "PPE MISSING").append("\n\n");
            report.append("Detected PPE\n");
            report.append("Helmet: ").append(scanDetectedHelmet ? "Yes" : "No")
                    .append(" (best ").append(formatPercent(scanBestConfidence[2])).append(")\n");
            report.append("Vest: ").append(scanDetectedVest ? "Yes" : "No")
                    .append(" (best ").append(formatPercent(scanBestConfidence[5])).append(")\n");
            report.append("Gloves: ").append(scanDetectedGloves ? "Yes" : "No")
                    .append(" (best ").append(formatPercent(scanBestConfidence[1])).append(")\n");
            report.append("Mask: ").append(scanDetectedMask ? "Yes" : "No")
                    .append(" (best ").append(formatPercent(scanBestConfidence[4])).append(")\n");
            report.append("Goggles: ").append(scanDetectedGoggles ? "Yes" : "No")
                    .append(" (best ").append(formatPercent(scanBestConfidence[3])).append(")\n\n");
            report.append("Missing PPE: ");
            if (violations.isEmpty()) {
                report.append("None");
            } else {
                for (int i = 0; i < violations.size(); i++) {
                    report.append(violations.get(i));
                    if (i < violations.size() - 1) report.append(", ");
                }
            }
            report.append("\nVoice Result: ").append(spokenResult).append("\n");

            java.io.FileOutputStream output = new java.io.FileOutputStream(reportFile);
            output.write(report.toString().getBytes("UTF-8"));
            output.close();

            handler.post(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, "PPE report saved", Toast.LENGTH_SHORT).show();
                    loadSavedPpeLogs();
                }
            });
        } catch (final Exception e) {
            handler.post(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, "Failed to save PPE report", Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    private void showPpeLog(java.io.File reportFile) {
        try {
            StringBuilder body = new StringBuilder();
            BufferedReader reader = new BufferedReader(new java.io.InputStreamReader(new java.io.FileInputStream(reportFile), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                body.append(line).append("\n");
            }
            reader.close();

            TextView logText = new TextView(this);
            logText.setText(body.toString());
            logText.setTextColor(Color.parseColor("#17212B"));
            logText.setTextSize(14);
            logText.setPadding(dp(18), dp(14), dp(18), dp(14));

            ScrollView scroll = new ScrollView(this);
            scroll.addView(logText);

            new AlertDialog.Builder(this)
                    .setTitle(reportFile.getName())
                    .setView(scroll)
                    .setPositiveButton("Close", null)
                    .show();
        } catch (Exception e) {
            Toast.makeText(this, "Failed to open PPE report", Toast.LENGTH_SHORT).show();
        }
    }

    private void deletePpeLog(final java.io.File reportFile) {
        new AlertDialog.Builder(this)
                .setTitle("Delete PPE report?")
                .setMessage(reportFile.getName())
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Delete", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        if (reportFile.delete()) {
                            Toast.makeText(MainActivity.this, "PPE report deleted", Toast.LENGTH_SHORT).show();
                            loadSavedPpeLogs();
                        } else {
                            Toast.makeText(MainActivity.this, "Failed to delete PPE report", Toast.LENGTH_SHORT).show();
                        }
                    }
                })
                .show();
    }

    private String formatPercent(float value) {
        return String.format(Locale.US, "%.0f%%", value * 100f);
    }

    private void exportLocalPpeLogToDownloads(final java.io.File sourceFile) {
        if (sourceFile == null || !sourceFile.exists()) {
            Toast.makeText(this, "PPE report file not found", Toast.LENGTH_SHORT).show();
            return;
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
                checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1002);
            Toast.makeText(this, "Allow storage permission, then tap Download again", Toast.LENGTH_LONG).show();
            return;
        }
        Toast.makeText(this, "Downloading " + sourceFile.getName() + "...", Toast.LENGTH_SHORT).show();

        executor.execute(new Runnable() {
            @Override
            public void run() {
                java.io.InputStream input = null;
                java.io.OutputStream output = null;
                try {
                    input = new java.io.FileInputStream(sourceFile);

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        android.content.ContentValues values = new android.content.ContentValues();
                        values.put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, sourceFile.getName());
                        values.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "text/plain");
                        values.put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS);

                        android.content.ContentResolver resolver = getContentResolver();
                        android.net.Uri uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
                        if (uri != null) {
                            output = resolver.openOutputStream(uri);
                        }
                    }

                    if (output == null) {
                        java.io.File downloadDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS);
                        if (!downloadDir.exists()) {
                            downloadDir.mkdirs();
                        }
                        java.io.File targetFile = new java.io.File(downloadDir, sourceFile.getName());
                        output = new java.io.FileOutputStream(targetFile);
                    }

                    byte[] data = new byte[4096];
                    int count;
                    while ((count = input.read(data)) != -1) {
                        output.write(data, 0, count);
                    }
                    output.flush();

                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, sourceFile.getName() + " saved to Downloads", Toast.LENGTH_LONG).show();
                        }
                    });
                } catch (final Exception e) {
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Download failed: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        }
                    });
                } finally {
                    try {
                        if (output != null) output.close();
                        if (input != null) input.close();
                    } catch (Exception ignored) {
                    }
                }
            }
        });
    }

    private LinearLayout pageHeader(String title, String subtitle) {
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams headerParams = sectionParams();
        header.setLayoutParams(headerParams);

        TextView titleView = text(title, 28, "#17212B", Typeface.BOLD);
        header.addView(titleView);
        TextView subtitleView = text(subtitle, 14, "#667085", Typeface.NORMAL);
        LinearLayout.LayoutParams subParams = matchWrap();
        subParams.setMargins(0, dp(4), 0, 0);
        header.addView(subtitleView, subParams);
        return header;
    }

    private void buildNav() {
        navBar.removeAllViews();
        addNavButton("Dashboard", "dashboard");
        addNavButton("Camera", "camera");
        addNavButton("PPE", "ppe");
        addNavButton("Live Location", "live");
        addNavButton("IMU", "sensors");
        addNavButton("Posture", "posture");
        addNavButton("Connect", "connect");
    }

    private void addNavButton(String label, final String tab) {
        Button button = new Button(this);
        boolean active = activeTab.equals(tab);
        button.setText(label);
        button.setAllCaps(false);
        button.setTextSize(11);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setTextColor(Color.parseColor(active ? "#2563EB" : "#98A2B3"));
        button.setBackground(round(active ? "#EAF2FF" : "#FFFFFF", 8, active ? "#EAF2FF" : "#FFFFFF"));
        button.setPadding(dp(2), 0, dp(2), 0);
        button.setMinHeight(0);
        button.setMinimumHeight(0);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(label.length() > 9 ? 118 : 86), dp(54));
        params.setMargins(dp(3), 0, dp(3), 0);
        navBar.addView(button, params);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                renderTab(tab);
            }
        });
    }

    private void checkStatus(final boolean showToast) {
        setTopStatus("Connecting...");
        requestJson("/status", new JsonCallback() {
            @Override
            public void onSuccess(JSONObject json) {
                lastWifiState = "Connected";
                boolean recording = json.optBoolean("recording", false);
                setTopStatus("Helmet connected");
                if (wifiStatus != null) {
                    wifiStatus.setText(lastWifiState);
                }
                if (recordingState != null) {
                    recordingState.setText(recording ? "Active" : "Idle");
                }

                JSONObject calib = json.optJSONObject("gyro_calibration");
                if (calib != null) {
                    String status = calib.optString("status", "unknown");
                    JSONObject offsets = calib.optJSONObject("offsets");
                    String offsetStr = "-- / -- / --";
                    if (offsets != null) {
                        offsetStr = String.format(Locale.US, "X: %.1f, Y: %.1f, Z: %.1f",
                                offsets.optDouble("x"), offsets.optDouble("y"), offsets.optDouble("z"));
                    }
                    if (calibStatusText != null) {
                        calibStatusText.setText("Status  " + status);
                    }
                    if (calibOffsetsText != null) {
                        calibOffsetsText.setText("Offsets (X/Y/Z)  " + offsetStr);
                    }
                }

                imuRecordingActive = json.optBoolean("imu_recording", false);
                if (toggleRecordBtn != null) {
                    toggleRecordBtn.setText(imuRecordingActive ? "Stop CSV Record" : "Start CSV Record");
                    toggleRecordBtn.setBackground(round(imuRecordingActive ? "#D04F3B" : "#0F766E", 8, imuRecordingActive ? "#D04F3B" : "#0F766E"));
                }

                if (showToast) {
                    Toast.makeText(MainActivity.this, "Wi-Fi connected", Toast.LENGTH_SHORT).show();
                }
                startPolling();
            }

            @Override
            public void onError(String message) {
                lastWifiState = "Waiting";
                setTopStatus("Helmet not reachable");
                if (wifiStatus != null) {
                    wifiStatus.setText(lastWifiState);
                }
                if (calibStatusText != null) {
                    calibStatusText.setText("Status  offline");
                }
                if (calibOffsetsText != null) {
                    calibOffsetsText.setText("Offsets (X/Y/Z)  -- / -- / --");
                }
                if (showToast) {
                    Toast.makeText(MainActivity.this, "Check Pi IP and hotspot", Toast.LENGTH_SHORT).show();
                }
                // Silently attempt UDP auto-discovery on local Wi-Fi if current IP is unreachable
                performUdpDiscovery(false);
            }
        });
    }

    private void performUdpDiscovery(final boolean showToast) {
        if (showToast) {
            Toast.makeText(this, "Scanning local network for Helmet...", Toast.LENGTH_SHORT).show();
        }
        executor.execute(new Runnable() {
            @Override
            public void run() {
                java.net.DatagramSocket socket = null;
                try {
                    socket = new java.net.DatagramSocket();
                    socket.setBroadcast(true);
                    socket.setSoTimeout(1800);

                    byte[] sendData = "SMART_HELMET_DISCOVER".getBytes(java.nio.charset.StandardCharsets.UTF_8);

                    // 1. Try global broadcast 255.255.255.255
                    try {
                        java.net.DatagramPacket packet = new java.net.DatagramPacket(
                                sendData, sendData.length,
                                java.net.InetAddress.getByName("255.255.255.255"), 5005
                        );
                        socket.send(packet);
                    } catch (Exception ignored) {}

                    // 2. Scan Mobile Hotspot ARP table (/proc/net/arp) to ping tethered client devices directly
                    try {
                        java.io.BufferedReader arpReader = new java.io.BufferedReader(new java.io.FileReader("/proc/net/arp"));
                        String line;
                        while ((line = arpReader.readLine()) != null) {
                            String[] tokens = line.trim().split("\\s+");
                            if (tokens.length >= 4 && !tokens[0].equals("IP") && !tokens[3].equals("00:00:00:00:00:00")) {
                                try {
                                    java.net.InetAddress candidate = java.net.InetAddress.getByName(tokens[0]);
                                    java.net.DatagramPacket p = new java.net.DatagramPacket(sendData, sendData.length, candidate, 5005);
                                    socket.send(p);
                                } catch (Exception ignored) {}
                            }
                        }
                        arpReader.close();
                    } catch (Exception ignored) {}

                    // 3. Try subnet broadcast addresses on all active network interfaces
                    try {
                        java.util.Enumeration<java.net.NetworkInterface> interfaces = java.net.NetworkInterface.getNetworkInterfaces();
                        while (interfaces != null && interfaces.hasMoreElements()) {
                            java.net.NetworkInterface ni = interfaces.nextElement();
                            if (ni.isLoopback() || !ni.isUp()) continue;
                            for (java.net.InterfaceAddress ia : ni.getInterfaceAddresses()) {
                                java.net.InetAddress broadcast = ia.getBroadcast();
                                if (broadcast != null) {
                                    java.net.DatagramPacket p = new java.net.DatagramPacket(sendData, sendData.length, broadcast, 5005);
                                    socket.send(p);
                                }
                            }
                        }
                    } catch (Exception ignored) {}

                    byte[] recvBuf = new byte[1024];
                    java.net.DatagramPacket recvPacket = new java.net.DatagramPacket(recvBuf, recvBuf.length);
                    socket.receive(recvPacket);

                    String resp = new String(recvPacket.getData(), 0, recvPacket.getLength(), java.nio.charset.StandardCharsets.UTF_8);
                    if (resp.startsWith("SMART_HELMET_RESPONSE")) {
                        final String discoveredHost = recvPacket.getAddress().getHostAddress() + ":5000";
                        handler.post(new Runnable() {
                            @Override
                            public void run() {
                                if (hostInput != null) {
                                    hostInput.setText(discoveredHost);
                                }
                                getSharedPreferences(PREFS, MODE_PRIVATE)
                                        .edit()
                                        .putString("host", discoveredHost)
                                        .apply();
                                Toast.makeText(MainActivity.this, "Auto-discovered Helmet at " + discoveredHost, Toast.LENGTH_LONG).show();
                                checkStatus(true);
                            }
                        });
                        return;
                    }
                } catch (java.net.SocketTimeoutException e) {
                    if (showToast) {
                        handler.post(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(MainActivity.this, "No Helmet discovered on local Wi-Fi", Toast.LENGTH_SHORT).show();
                            }
                        });
                    }
                } catch (Exception e) {
                    if (showToast) {
                        handler.post(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(MainActivity.this, "Discovery error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                            }
                        });
                    }
                } finally {
                    if (socket != null && !socket.isClosed()) {
                        socket.close();
                    }
                }
            }
        });
    }

    private long lastHandledHeadCmdId = 0;

    private final Runnable backgroundGesturePoller = new Runnable() {
        @Override
        public void run() {
            if (!polling) return;
            requestJson("/posture", new JsonCallback() {
                @Override
                public void onSuccess(JSONObject json) {
                    long cmdId = json.optLong("head_cmd_id", 0);
                    String headCmd = json.optString("active_head_command", "NONE");
                    if (cmdId > 0 && cmdId > lastHandledHeadCmdId) {
                        lastHandledHeadCmdId = cmdId;
                        handleHeadGestureCommand(headCmd);
                    }
                    handler.postDelayed(backgroundGesturePoller, 350);
                }
                @Override
                public void onError(String message) {
                    handler.postDelayed(backgroundGesturePoller, 2000);
                }
            });
        }
    };

    private void handleHeadGestureCommand(String cmd) {
        if ("START_PPE".equals(cmd)) {
            if (isTtsReady && ttsEngine != null) {
                ttsEngine.speak("Head gesture detected. Starting P P E safety compliance audit.", 
                                android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "GesturePpeSpeech");
            }
            Toast.makeText(this, "Gesture: Starting PPE Scan", Toast.LENGTH_SHORT).show();
            loadPpeModelIfNeeded();
            startPpeStream();
        } else if ("START_LOCATION".equals(cmd)) {
            if (isTtsReady && ttsEngine != null) {
                ttsEngine.speak("Head gesture detected. Starting Live Location code scan.", 
                                android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "GestureLocSpeech");
            }
            Toast.makeText(this, "Gesture: Starting Live Location", Toast.LENGTH_SHORT).show();
            startLiveLocationScan();
        } else if ("STOP_ALL".equals(cmd)) {
            Toast.makeText(this, "Gesture: Model turned off", Toast.LENGTH_SHORT).show();
            stopPpeStream();
            stopLiveLocationScan();
            resetPpeScanState(true);
            if (isTtsReady && ttsEngine != null) {
                ttsEngine.speak("Model turned off.", 
                                android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "GestureStopSpeech");
            }
        }
    }

    private void startPolling() {
        polling = true;
        handler.removeCallbacks(imuPoller);
        handler.post(imuPoller);
        handler.removeCallbacks(backgroundGesturePoller);
        handler.post(backgroundGesturePoller);
    }

    private void updateImu(JSONObject json) {
        JSONObject accel = json.optJSONObject("accelerometer");
        JSONObject gyro = json.optJSONObject("gyroscope");
        JSONObject mag = json.optJSONObject("magnetometer");

        if (accel != null && accelX != null) {
            double ax = accel.optDouble("x_g", 0.0);
            double ay = accel.optDouble("y_g", 0.0);
            double az = accel.optDouble("z_g", 0.0);

            accelX.setText(format(ax, " g"));
            accelY.setText(format(ay, " g"));
            accelZ.setText(format(az, " g"));

            // Calculate tilt pitch and roll for 3D View
            float pitch = (float) Math.toDegrees(Math.atan2(-ax, Math.sqrt(ay * ay + az * az)));
            float roll = (float) Math.toDegrees(Math.atan2(ay, az));
            if (cubeView != null) {
                cubeView.updateOrientation(pitch, roll, 0);
            }
        }

        if (gyro != null && gyroX != null) {
            gyroX.setText(format(gyro.optDouble("x_dps"), " deg/s"));
            gyroY.setText(format(gyro.optDouble("y_dps"), " deg/s"));
            gyroZ.setText(format(gyro.optDouble("z_dps"), " deg/s"));
        }



        if (altitudeText != null && json.has("altitude_m")) {
            altitudeText.setText(format(json.optDouble("altitude_m"), " m"));
        }


    }

    private void resetSensorMetrics() {
        if (accelX != null) accelX.setText("-- g");
        if (accelY != null) accelY.setText("-- g");
        if (accelZ != null) accelZ.setText("-- g");
        if (gyroX != null) gyroX.setText("-- deg/s");
        if (gyroY != null) gyroY.setText("-- deg/s");
        if (gyroZ != null) gyroZ.setText("-- deg/s");

        if (altitudeText != null) altitudeText.setText("-- m");

        if (cubeView != null) cubeView.updateOrientation(0, 0, 0);
    }

    private void runSensorRecordCommand(final String path, final String successMessage) {
        requestJson(path, new JsonCallback() {
            @Override
            public void onSuccess(JSONObject json) {
                Toast.makeText(MainActivity.this, successMessage, Toast.LENGTH_SHORT).show();
                imuRecordingActive = path.contains("start");
                if (toggleRecordBtn != null) {
                    toggleRecordBtn.setText(imuRecordingActive ? "Stop CSV Record" : "Start CSV Record");
                    toggleRecordBtn.setBackground(round(imuRecordingActive ? "#D04F3B" : "#0F766E", 8, imuRecordingActive ? "#D04F3B" : "#0F766E"));
                }
                String file = json.optString("file", "");
                if (!file.isEmpty()) {
                    setTopStatus(successMessage + " on Pi");
                }
                loadSavedLogs();
            }

            @Override
            public void onError(String message) {
                Toast.makeText(MainActivity.this, "Action failed", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void loadSavedLogs() {
        if (logsListContainer == null) return;
        logsListContainer.removeAllViews();

        TextView loading = new TextView(this);
        loading.setText("Loading files...");
        loading.setTextColor(Color.parseColor("#94A3B8"));
        loading.setTextSize(14);
        logsListContainer.addView(loading, matchWrap());

        requestJson("/media/list", new JsonCallback() {
            @Override
            public void onSuccess(final JSONObject json) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (logsListContainer == null) return;
                        logsListContainer.removeAllViews();

                        JSONArray files = json.optJSONArray("files");
                        if (files == null || files.length() == 0) {
                            TextView empty = new TextView(MainActivity.this);
                            empty.setText("No saved logs found");
                            empty.setTextColor(Color.parseColor("#94A3B8"));
                            empty.setTextSize(14);
                            logsListContainer.addView(empty, matchWrap());
                            return;
                        }

                        boolean foundCsv = false;
                        for (int i = 0; i < files.length(); i++) {
                            JSONObject file = files.optJSONObject(i);
                            if (file != null) {
                                final String name = file.optString("name", "");
                                if (name.endsWith(".csv")) {
                                    foundCsv = true;
                                    double kb = file.optDouble("size", 0) / 1024.0;
                                    String sizeStr = String.format(Locale.US, "%.1f KB", kb);

                                    LinearLayout row = new LinearLayout(MainActivity.this);
                                    row.setOrientation(LinearLayout.HORIZONTAL);
                                    LinearLayout.LayoutParams rowParams = matchWrap();
                                    rowParams.setMargins(0, dp(8), 0, dp(8));
                                    row.setLayoutParams(rowParams);

                                    TextView nameText = new TextView(MainActivity.this);
                                    nameText.setText(name);
                                    nameText.setTextColor(Color.parseColor("#1E293B"));
                                    nameText.setTextSize(14);
                                    nameText.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
                                    LinearLayout.LayoutParams nameParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
                                    row.addView(nameText, nameParams);

                                    TextView sizeText = new TextView(MainActivity.this);
                                    sizeText.setText(sizeStr);
                                    sizeText.setTextColor(Color.parseColor("#64748B"));
                                    sizeText.setTextSize(12);
                                    row.addView(sizeText, wrapWrap());

                                    Button dlBtn = new Button(MainActivity.this);
                                    dlBtn.setText("Download");
                                    dlBtn.setTextColor(Color.WHITE);
                                    dlBtn.setTextSize(11);
                                    dlBtn.setAllCaps(false);
                                    dlBtn.setPadding(dp(8), 0, dp(8), 0);
                                    dlBtn.setBackground(round("#3B82F6", 6, "#3B82F6"));
                                    dlBtn.setMinHeight(0);
                                    dlBtn.setMinimumHeight(0);
                                    LinearLayout.LayoutParams btnParams = wrapWrap();
                                    btnParams.setMargins(dp(10), 0, 0, 0);
                                    dlBtn.setLayoutParams(btnParams);
                                    dlBtn.setOnClickListener(new View.OnClickListener() {
                                        @Override
                                        public void onClick(View v) {
                                            downloadFile(name);
                                        }
                                    });
                                    row.addView(dlBtn);

                                    logsListContainer.addView(row);

                                    View divider = new View(MainActivity.this);
                                    divider.setBackgroundColor(Color.parseColor("#E2E8F0"));
                                    LinearLayout.LayoutParams divParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1));
                                    logsListContainer.addView(divider, divParams);
                                }
                            }
                        }

                        if (!foundCsv) {
                            TextView empty = new TextView(MainActivity.this);
                            empty.setText("No saved logs found");
                            empty.setTextColor(Color.parseColor("#94A3B8"));
                            empty.setTextSize(14);
                            logsListContainer.addView(empty, matchWrap());
                        }
                    }
                });
            }

            @Override
            public void onError(String message) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (logsListContainer == null) return;
                        logsListContainer.removeAllViews();
                        TextView error = new TextView(MainActivity.this);
                        error.setText("Failed to load logs");
                        error.setTextColor(Color.parseColor("#EF4444"));
                        error.setTextSize(14);
                        logsListContainer.addView(error, matchWrap());
                    }
                });
            }
        });
    }

    private void startPreview() {
        if (ppeStreamOn) {
            stopPpeStream();
        }
        streamOn = true;
        setTopStatus("Camera preview active");
        if (cameraState != null) {
            cameraState.setText("Live preview");
        }
        if (cameraToggleButton != null) {
            cameraToggleButton.setText("Camera Off");
        }
        if (previewStatus != null) {
            previewStatus.setText("Preview  On");
        }
        if (previewPlaceholder != null) {
            previewPlaceholder.setVisibility(View.GONE);
        }
        startMjpegReader();
    }

    private void stopPreview() {
        streamOn = false;
        stopMjpegReader();
        setTopStatus("Camera preview off");
        if (cameraState != null) {
            cameraState.setText("Off");
        }
        if (cameraToggleButton != null) {
            cameraToggleButton.setText("Camera On");
        }
        if (previewStatus != null) {
            previewStatus.setText("Preview  Off");
        }
        if (cameraImage != null) {
            cameraImage.setImageBitmap(null);
        }
        if (previewPlaceholder != null) {
            previewPlaceholder.setVisibility(View.VISIBLE);
        }
    }


    private void startMjpegReader() {
        stopMjpegReader();
        streamReaderRunning = true;
        final int myGeneration = ++streamGeneration;
        // Snapshot cameraImage NOW so tab-switch can't corrupt this thread's reference
        final ImageView myImage = cameraImage;
        final TextView myPlaceholder = previewPlaceholder;

        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                readMjpegStream(myGeneration, myImage, myPlaceholder);
            }
        });
        t.setDaemon(true);
        t.setName("mjpeg-stream-" + myGeneration);
        t.start();
    }

    private void stopMjpegReader() {
        streamReaderRunning = false;
        streamGeneration++; // invalidate any running stream thread
        if (streamConnection != null) {
            try { streamConnection.disconnect(); } catch (Exception ignored) {}
            streamConnection = null;
        }
    }

    private void readMjpegStream(final int myGeneration, final ImageView myImage, final TextView myPlaceholder) {
        HttpURLConnection connection = null;
        InputStream stream = null;
        try {
            URL url = new URL(baseUrl() + "/video");
            connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(STREAM_CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(STREAM_READ_TIMEOUT_MS);
            connection.setRequestMethod("GET");
            streamConnection = connection;
            if (connection.getResponseCode() != 200) throw new Exception("HTTP " + connection.getResponseCode());

            stream = connection.getInputStream();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream(65536);
            byte[] chunk = new byte[8192];
            boolean inJpeg = false;
            int previous = -1, read;
            long lastDisplayFrameMs = 0;

            while (myGeneration == streamGeneration && streamReaderRunning && (read = stream.read(chunk)) != -1) {
                for (int i = 0; i < read; i++) {
                    if (myGeneration != streamGeneration) break;
                    int current = chunk[i] & 0xFF;
                    if (!inJpeg) {
                        if (previous == 0xFF && current == 0xD8) {
                            buffer.reset(); buffer.write(0xFF); buffer.write(0xD8); inJpeg = true;
                        }
                    } else {
                        buffer.write(current);
                        if (previous == 0xFF && current == 0xD9) {
                            // CAMERA TAB: display only — zero AI, zero overhead
                            long now = System.currentTimeMillis();
                            if (now - lastDisplayFrameMs >= CAMERA_DISPLAY_FRAME_INTERVAL_MS) {
                                lastDisplayFrameMs = now;
                                final byte[] frameBytes = buffer.toByteArray();
                                final Bitmap bmp = BitmapFactory.decodeByteArray(frameBytes, 0, frameBytes.length);
                                if (bmp != null && myGeneration == streamGeneration) {
                                    handler.post(new Runnable() {
                                        @Override public void run() {
                                            if (myGeneration == streamGeneration && streamOn && myImage != null) {
                                                myImage.setImageBitmap(bmp);
                                                if (myPlaceholder != null) myPlaceholder.setVisibility(View.GONE);
                                            }
                                        }
                                    });
                                }
                            }
                            buffer.reset(); inJpeg = false;
                        }
                    }
                    previous = current;
                }
            }
        } catch (final Exception e) {
            if (streamReaderRunning && myGeneration == streamGeneration) {
                handler.post(new Runnable() {
                    @Override public void run() {
                        setTopStatus("Camera stream waiting");
                        if (myPlaceholder != null) { myPlaceholder.setVisibility(View.VISIBLE); myPlaceholder.setText("Camera stream reconnecting..."); }
                    }
                });
                scheduleCameraStreamRetry(myGeneration);
            }
        } finally {
            try { if (stream != null) stream.close(); } catch (Exception ignored) {}
            if (connection != null) connection.disconnect();
            if (streamConnection == connection) streamConnection = null;
        }
    }

    // ── PPE INDEPENDENT STREAM ─────────────────────────────────────────────────
    private void scheduleCameraStreamRetry(final int failedGeneration) {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (streamOn && streamReaderRunning && failedGeneration == streamGeneration && "camera".equals(activeTab)) {
                    startMjpegReader();
                }
            }
        }, STREAM_RETRY_DELAY_MS);
    }

    private void startPpeStream() {
        if (streamOn) {
            stopPreview();
        }
        stopPpeStream();
        resetPpeScanState(true);
        ppeStreamOn = true;
        ppeReaderRunning = true;
        setTopStatus("PPE stream active");
        if (previewStatus != null) {
            previewStatus.setText(isModelLoaded ? "Waiting for person" : "Loading model...");
        }
        final int gen = ++ppeStreamGen;
        final ImageView img = cameraImage;
        final TextView ph = previewPlaceholder;
        if (cameraToggleButton != null) cameraToggleButton.setText("PPE Check Off");
        if (ph != null) ph.setVisibility(View.GONE);
        Thread t = new Thread(new Runnable() {
            @Override public void run() { readPpeStream(gen, img, ph); }
        });
        t.setDaemon(true);
        t.setName("ppe-stream-" + gen);
        t.start();
    }

    private void stopPpeStream() {
        ppeStreamOn = false;
        ppeReaderRunning = false;
        ppeStreamGen++;
        if (ppeStreamConn != null) { try { ppeStreamConn.disconnect(); } catch (Exception ignored) {} ppeStreamConn = null; }
        if (cameraToggleButton != null) cameraToggleButton.setText("PPE Check On");
        if (cameraImage != null) cameraImage.setImageBitmap(null);
        if (previewPlaceholder != null) previewPlaceholder.setVisibility(View.VISIBLE);
        // Reset scanner states immediately
        resetPpeScanState(true);
        if (previewStatus != null) {
            previewStatus.setText(isModelLoaded ? "AI Model Ready" : "Loading model...");
        }
    }

    private void resetPpeScanState(boolean clearPersonState) {
        isScanningActive = false;
        scanStartTime = 0;
        scanResultsAnnounced = false;
        scanAnnouncementCount = 0;
        scanDetectedHelmet = false;
        scanDetectedVest = false;
        scanDetectedGloves = false;
        scanDetectedMask = false;
        scanDetectedGoggles = false;
        java.util.Arrays.fill(scanBestConfidence, 0f);
        java.util.Arrays.fill(scanHitCount, 0);
        java.util.Arrays.fill(scanFirstHitTime, 0L);
        if (clearPersonState) {
            personFirstDetectedTime = 0;
            lastPersonDetectedTime = 0;
            isPersonInFrame = false;
        }
        alertNoHelmet = false;
        alertNoGoggles = false;
        alertNoMask = false;
        alertNoVest = false;
        alertMissingPpe = false;
        alertChinstrapOpen = false;
        synchronized (latestDetections) {
            cachedDetections.clear();
            latestDetections.clear();
        }
    }

    private void setPpeInferenceStatus(final String status) {
        handler.post(new Runnable() {
            @Override
            public void run() {
                if (previewStatus != null) {
                    previewStatus.setText(status);
                }
            }
        });
    }

    private void rememberScanDetection(int classId, float confidence) {
        if (classId < 0 || classId >= scanBestConfidence.length) return;
        
        // Accept valid detections starting at confidence threshold for class
        if (confidence < getConfidenceThreshold(classId)) return;
        
        if (confidence > scanBestConfidence[classId]) {
            scanBestConfidence[classId] = confidence;
        }
        
        long now = System.currentTimeMillis();
        scanHitCount[classId]++;
        if (scanFirstHitTime[classId] == 0) {
            scanFirstHitTime[classId] = now;
        }
        
        // Multi-Frame Verification: Mark present if detected at least twice or sustained for >= 800ms
        if (scanHitCount[classId] >= 2 || (now - scanFirstHitTime[classId] >= 800)) {
            if (classId == 2) scanDetectedHelmet = true;
            if (classId == 5) scanDetectedVest = true;
            if (classId == 1) scanDetectedGloves = true;
            if (classId == 4) scanDetectedMask = true;
            if (classId == 3) scanDetectedGoggles = true;
        }
    }

    private void startLiveLocationScan() {
        if (streamOn) stopPreview();
        if (ppeStreamOn) stopPpeStream();
        stopLiveLocationScan();
        liveScanOn = true;
        liveReaderRunning = true;
        pendingLiveCode = "";
        pendingLiveCodeHits = 0;
        pendingLiveCodeFirstMs = 0;
        final int gen = ++liveStreamGen;
        final ImageView img = liveCameraImage;
        final TextView ph = livePreviewPlaceholder;
        if (liveScanButton != null) liveScanButton.setText("Stop Code Scan");
        if (ph != null) ph.setVisibility(View.GONE);
        setLiveScanStatus(isPaddleOcrReady() ? "Scanning for saved codes" : "Paddle OCR assets missing");
        Thread t = new Thread(new Runnable() {
            @Override public void run() { readLiveLocationStream(gen, img, ph); }
        });
        t.setDaemon(true);
        t.setName("live-location-stream-" + gen);
        t.start();
    }

    private void stopLiveLocationScan() {
        liveScanOn = false;
        liveReaderRunning = false;
        liveStreamGen++;
        if (liveStreamConn != null) {
            try { liveStreamConn.disconnect(); } catch (Exception ignored) {}
            liveStreamConn = null;
        }
        if (liveScanButton != null) liveScanButton.setText("Start Code Scan");
        if (liveCameraImage != null) liveCameraImage.setImageBitmap(null);
        if (livePreviewPlaceholder != null) livePreviewPlaceholder.setVisibility(View.VISIBLE);
        setLiveScanStatus("Stopped");
    }

    private void setLiveScanStatus(final String status) {
        handler.post(new Runnable() {
            @Override public void run() {
                if (liveScanStatus != null) {
                    liveScanStatus.setText(status);
                }
            }
        });
    }

    private void readLiveLocationStream(final int gen, final ImageView img, final TextView ph) {
        HttpURLConnection conn = null;
        InputStream stream = null;
        try {
            URL url = new URL(baseUrl() + "/video");
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(STREAM_CONNECT_TIMEOUT_MS);
            conn.setReadTimeout(STREAM_READ_TIMEOUT_MS);
            liveStreamConn = conn;
            if (conn.getResponseCode() != 200) throw new Exception("HTTP " + conn.getResponseCode());

            stream = conn.getInputStream();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream(65536);
            byte[] chunk = new byte[8192];
            boolean inJpeg = false;
            int previous = -1, read;
            long lastOcrFrameMs = 0;

            while (gen == liveStreamGen && liveReaderRunning && (read = stream.read(chunk)) != -1) {
                for (int i = 0; i < read; i++) {
                    if (gen != liveStreamGen) break;
                    int current = chunk[i] & 0xFF;
                    if (!inJpeg) {
                        if (previous == 0xFF && current == 0xD8) {
                            buffer.reset(); buffer.write(0xFF); buffer.write(0xD8); inJpeg = true;
                        }
                    } else {
                        buffer.write(current);
                        if (previous == 0xFF && current == 0xD9) {
                            final byte[] frameBytes = buffer.toByteArray();
                            final Bitmap bmp = BitmapFactory.decodeByteArray(frameBytes, 0, frameBytes.length);
                            if (bmp != null && gen == liveStreamGen) {
                                handler.post(new Runnable() {
                                    @Override public void run() {
                                        if (gen == liveStreamGen && liveScanOn && img != null) {
                                            img.setImageBitmap(bmp);
                                            if (ph != null) ph.setVisibility(View.GONE);
                                        }
                                    }
                                });
                                long now = System.currentTimeMillis();
                                if (now - lastOcrFrameMs >= LIVE_OCR_FRAME_INTERVAL_MS && !liveOcrProcessing) {
                                    lastOcrFrameMs = now;
                                    liveOcrProcessing = true;
                                    final Bitmap ocrBitmap = bmp.copy(Bitmap.Config.ARGB_8888, true);
                                    executor.execute(new Runnable() {
                                        @Override public void run() {
                                            try { runLiveLocationOcr(ocrBitmap); }
                                            finally { liveOcrProcessing = false; }
                                        }
                                    });
                                }
                            }
                            buffer.reset(); inJpeg = false;
                        }
                    }
                    previous = current;
                }
            }
        } catch (final Exception e) {
            if (liveReaderRunning && gen == liveStreamGen) {
                setLiveScanStatus("Live location stream reconnecting");
                handler.postDelayed(new Runnable() {
                    @Override public void run() {
                        if (liveScanOn && liveReaderRunning && gen == liveStreamGen && "live".equals(activeTab)) {
                            startLiveLocationScan();
                        }
                    }
                }, STREAM_RETRY_DELAY_MS);
            }
        } finally {
            try { if (stream != null) stream.close(); } catch (Exception ignored) {}
            if (conn != null) conn.disconnect();
            if (liveStreamConn == conn) liveStreamConn = null;
        }
    }

        /**
     * OCR ENGINE — PaddleOCR Zone Code Location Recognition
     * Runs PaddleLite C++ runtime (DB text detection + CRNN BiLSTM recognition) on camera bitmap.
     * Evaluates multi-frame temporal consensus (3 matching reads in 3s) and applies 60s per-code cooldown.
     */
    private void runLiveLocationOcr(Bitmap bitmap) {
        java.util.List<String> recognizedTexts = runPaddleOcr(bitmap);
        if (recognizedTexts.isEmpty()) {
            if (!isPaddleOcrReady()) {
                setLiveScanStatus("Paddle OCR assets missing: add native runtime + models");
            }
            return;
        }
        for (String textValue : recognizedTexts) {
            JSONObject mapping = findLiveLocationMapping(textValue);
            if (mapping != null) {
                handleLiveCodeCandidate(mapping, textValue);
                return;
            }
        }
        String seen = recognizedTexts.get(0);
        if (seen.length() > 20) {
            seen = seen.substring(0, 17) + "...";
        }
        setLiveScanStatus("Seen: \"" + seen + "\" (No saved match)");
    }

    private void handleLiveCodeCandidate(final JSONObject mapping, String recognizedText) {
        String normalized = normalizeLiveCode(recognizedText);
        long now = System.currentTimeMillis();
        if (!normalized.equals(pendingLiveCode) || now - pendingLiveCodeFirstMs > 3000) {
            pendingLiveCode = normalized;
            pendingLiveCodeHits = 1;
            pendingLiveCodeFirstMs = now;
        } else {
            pendingLiveCodeHits++;
        }
        setLiveScanStatus("Matched " + mapping.optString("code", normalized) + " (" + pendingLiveCodeHits + "/3)");
        if (pendingLiveCodeHits >= 3) {
            if (normalized.equals(lastSentLiveCode) && now - lastSentLiveCodeMs < 60000) {
                setLiveScanStatus("Matched recently - cooldown active");
                return;
            }
            lastSentLiveCode = normalized;
            lastSentLiveCodeMs = now;
            if (isTtsReady && ttsEngine != null) {
                ttsEngine.speak("Text detected and verified.", android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "OcrTextVerified");
            }
            handler.post(new Runnable() {
                @Override public void run() {
                    sendLiveLocationSms(mapping, mapping.optString("code", pendingLiveCode));
                }
            });
            pendingLiveCode = "";
            pendingLiveCodeHits = 0;
            pendingLiveCodeFirstMs = 0;
        }
    }

    private boolean isPaddleOcrReady() {
        return arePaddleOcrAssetsPresent() && isPaddleRuntimeReady();
    }

    private boolean arePaddleOcrAssetsPresent() {
        try {
            getAssets().open("paddle/det_db.nb").close();
            getAssets().open("paddle/rec_crnn.nb").close();
            getAssets().open("paddle/cls.nb").close();
            getAssets().open("paddle/ppocr_keys_v1.txt").close();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private boolean isPaddleRuntimeReady() {
        if (paddleRuntimeChecked) return paddleRuntimeReady;
        paddleRuntimeChecked = true;
        try {
            System.loadLibrary("Native");
            paddleRuntimeReady = true;
        } catch (Throwable t) {
            paddleRuntimeReady = false;
        }
        return paddleRuntimeReady;
    }

    private java.util.List<String> runPaddleOcr(Bitmap bitmap) {
        java.util.List<String> results = new java.util.ArrayList<>();
        if (!isPaddleOcrReady() || bitmap == null) {
            return results;
        }
        try {
            if (paddlePredictorInstance == null) {
                paddlePredictorInstance = new com.baidu.paddle.lite.demo.ocr.Predictor();
                boolean initSuccess = paddlePredictorInstance.init(
                    MainActivity.this,
                    "paddle",
                    "paddle/ppocr_keys_v1.txt",
                    0, // useOpencl: 0
                    4, // cpuThreadNum: 4
                    "LITE_POWER_HIGH",
                    960, // detLongSize
                    0.1f // scoreThreshold
                );
                if (!initSuccess) {
                    android.util.Log.e("SmartHelmetOCR", "Failed to initialize PaddleOCR Predictor");
                    paddlePredictorInstance = null;
                    return results;
                }
            }
            java.util.ArrayList<com.baidu.paddle.lite.demo.ocr.OcrResultModel> ocrResults =
                paddlePredictorInstance.runImage(bitmap, 1, 0, 1);
            if (ocrResults != null) {
                for (com.baidu.paddle.lite.demo.ocr.OcrResultModel res : ocrResults) {
                    if (res != null && res.getLabel() != null) {
                        String txt = res.getLabel().trim();
                        if (!txt.isEmpty()) {
                            results.add(txt);
                        }
                    }
                }
            }
        } catch (Exception e) {
            android.util.Log.e("SmartHelmetOCR", "Error running PaddleOCR: " + e.getMessage(), e);
        }
        return results;
    }

    private void readPpeStream(final int gen, final ImageView img, final TextView ph) {
        HttpURLConnection conn = null;
        InputStream stream = null;
        try {
            URL url = new URL(baseUrl() + "/video/ppe");
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(STREAM_CONNECT_TIMEOUT_MS);
            conn.setReadTimeout(STREAM_READ_TIMEOUT_MS);
            ppeStreamConn = conn;
            if (conn.getResponseCode() != 200) throw new Exception("HTTP " + conn.getResponseCode());

            stream = conn.getInputStream();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream(65536);
            byte[] chunk = new byte[8192];
            boolean inJpeg = false;
            int previous = -1, read;
            long lastDisplayFrameMs = 0;

            while (gen == ppeStreamGen && ppeReaderRunning && (read = stream.read(chunk)) != -1) {
                for (int i = 0; i < read; i++) {
                    if (gen != ppeStreamGen) break;
                    int current = chunk[i] & 0xFF;
                    if (!inJpeg) {
                        if (previous == 0xFF && current == 0xD8) {
                            buffer.reset(); buffer.write(0xFF); buffer.write(0xD8); inJpeg = true;
                        }
                    } else {
                        buffer.write(current);
                        if (previous == 0xFF && current == 0xD9) {
                            // PPE TAB: decode frame, overlay detections, then run 2-stage AI
                            long now = System.currentTimeMillis();
                            if (now - lastDisplayFrameMs >= PPE_DISPLAY_FRAME_INTERVAL_MS) {
                                lastDisplayFrameMs = now;
                                final byte[] frameBytes = buffer.toByteArray();
                                final Bitmap bmp = BitmapFactory.decodeByteArray(frameBytes, 0, frameBytes.length);
                                if (bmp != null && gen == ppeStreamGen) {
                                    final Bitmap display = drawDetectionsOnBitmap(bmp, latestDetections);
                                    handler.post(new Runnable() {
                                        @Override public void run() {
                                            if (gen == ppeStreamGen && ppeStreamOn && img != null) {
                                                img.setImageBitmap(display);
                                                if (ph != null) ph.setVisibility(View.GONE);
                                            }
                                        }
                                    });
                                    // Run 2-stage AI inference without blocking the stream reader
                                    if (isModelLoaded && !isModelProcessing) {
                                        isModelProcessing = true;
                                        final Bitmap pb = bmp.copy(Bitmap.Config.ARGB_8888, true);
                                        executor.execute(new Runnable() {
                                            @Override public void run() {
                                                try { runInferenceOnBitmap(pb); }
                                                finally { isModelProcessing = false; }
                                            }
                                        });
                                    }
                                }
                            }
                            buffer.reset(); inJpeg = false;
                        }
                    }
                    previous = current;
                }
            }
        } catch (final Exception e) {
            if (ppeReaderRunning && gen == ppeStreamGen) {
                handler.post(new Runnable() {
                    @Override public void run() {
                        setTopStatus("PPE stream waiting");
                        if (ph != null) { ph.setVisibility(View.VISIBLE); ph.setText("PPE stream reconnecting..."); }
                    }
                });
                schedulePpeStreamRetry(gen);
            }
        } finally {
            try { if (stream != null) stream.close(); } catch (Exception ignored) {}
            if (conn != null) conn.disconnect();
            if (ppeStreamConn == conn) ppeStreamConn = null;
        }
    }

    private void schedulePpeStreamRetry(final int failedGeneration) {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                if (ppeStreamOn && ppeReaderRunning && failedGeneration == ppeStreamGen && "ppe".equals(activeTab)) {
                    startPpeStream();
                }
            }
        }, STREAM_RETRY_DELAY_MS);
    }

    private void runCameraCommand(final String path, final String successMessage, boolean pausePreviewFirst) {
        if (pausePreviewFirst && streamOn) {
            stopPreview();
            Toast.makeText(this, "Preview paused for camera action", Toast.LENGTH_SHORT).show();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
                    callCommand(path, successMessage);
                }
            }, 900);
            return;
        }

        callCommand(path, successMessage);
    }

    private void callCommand(String path, final String successMessage) {
        requestJson(path, new JsonCallback() {
            @Override
            public void onSuccess(JSONObject json) {
                Toast.makeText(MainActivity.this, successMessage, Toast.LENGTH_SHORT).show();
                String file = json.optString("file", "");
                if (file.isEmpty()) {
                    setTopStatus(successMessage);
                } else {
                    setTopStatus(successMessage + " on Pi");
                }
                loadCameraSavedLogs();
            }

            @Override
            public void onError(String message) {
                Toast.makeText(MainActivity.this, "Command failed", Toast.LENGTH_SHORT).show();
                setTopStatus("Command failed");
            }
        });
    }

    private void callCommandWithCallback(String path, final String successMessage, final Runnable onComplete) {
        requestJson(path, new JsonCallback() {
            @Override
            public void onSuccess(JSONObject json) {
                Toast.makeText(MainActivity.this, successMessage, Toast.LENGTH_SHORT).show();
                String file = json.optString("file", "");
                if (file.isEmpty()) {
                    setTopStatus(successMessage);
                } else {
                    setTopStatus(successMessage + " on Pi");
                }
                loadCameraSavedLogs();
                // Execute the callback after command succeeds
                if (onComplete != null) {
                    handler.postDelayed(onComplete, 500);
                }
            }

            @Override
            public void onError(String message) {
                Toast.makeText(MainActivity.this, "Command failed", Toast.LENGTH_SHORT).show();
                setTopStatus("Command failed");
                // Still try to resume preview even if photo failed
                if (onComplete != null) {
                    onComplete.run();
                }
            }
        });
    }

    private void loadCameraSavedLogs() {
        if (cameraLogsListContainer == null) return;
        cameraLogsListContainer.removeAllViews();

        TextView loading = new TextView(this);
        loading.setText("Loading files...");
        loading.setTextColor(Color.parseColor("#94A3B8"));
        loading.setTextSize(14);
        cameraLogsListContainer.addView(loading, matchWrap());

        requestJson("/media/list", new JsonCallback() {
            @Override
            public void onSuccess(final JSONObject json) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (cameraLogsListContainer == null) return;
                        cameraLogsListContainer.removeAllViews();

                        JSONArray files = json.optJSONArray("files");
                        if (files == null || files.length() == 0) {
                            TextView empty = new TextView(MainActivity.this);
                            empty.setText("No saved media found");
                            empty.setTextColor(Color.parseColor("#94A3B8"));
                            empty.setTextSize(14);
                            cameraLogsListContainer.addView(empty, matchWrap());
                            return;
                        }

                        boolean foundMedia = false;
                        for (int i = 0; i < files.length(); i++) {
                            JSONObject file = files.optJSONObject(i);
                            if (file != null) {
                                final String name = file.optString("name", "");
                                String nameLower = name.toLowerCase();
                                if (nameLower.endsWith(".jpg") || nameLower.endsWith(".jpeg") || nameLower.endsWith(".mjpg") || nameLower.endsWith(".mjpeg") || nameLower.endsWith(".mp4") || nameLower.endsWith(".avi") || nameLower.endsWith(".h264") || nameLower.endsWith(".csv")) {
                                    foundMedia = true;
                                    double kb = file.optDouble("size", 0) / 1024.0;
                                    String sizeStr = String.format(Locale.US, "%.1f KB", kb);
                                    if (kb > 1024.0) {
                                        sizeStr = String.format(Locale.US, "%.1f MB", kb / 1024.0);
                                    }

                                    LinearLayout card = new LinearLayout(MainActivity.this);
                                    card.setOrientation(LinearLayout.VERTICAL);
                                    LinearLayout.LayoutParams cardParams = matchWrap();
                                    cardParams.setMargins(0, dp(8), 0, dp(8));
                                    card.setLayoutParams(cardParams);

                                    // Top Line: Icon + File Name + Size
                                    LinearLayout topRow = new LinearLayout(MainActivity.this);
                                    topRow.setOrientation(LinearLayout.HORIZONTAL);
                                    topRow.setGravity(android.view.Gravity.CENTER_VERTICAL);

                                    String iconPrefix = (nameLower.endsWith(".mp4") || nameLower.endsWith(".mjpg") || nameLower.endsWith(".avi")) ? "🎬 " : (nameLower.endsWith(".jpg") || nameLower.endsWith(".jpeg") ? "📷 " : "📄 ");

                                    TextView nameText = new TextView(MainActivity.this);
                                    nameText.setText(iconPrefix + name);
                                    nameText.setTextColor(Color.parseColor("#1E293B"));
                                    nameText.setTextSize(13);
                                    nameText.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
                                    nameText.setEllipsize(android.text.TextUtils.TruncateAt.MIDDLE);
                                    nameText.setSingleLine(true);
                                    LinearLayout.LayoutParams nameParams = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
                                    topRow.addView(nameText, nameParams);

                                    TextView sizeText = new TextView(MainActivity.this);
                                    sizeText.setText(sizeStr);
                                    sizeText.setTextColor(Color.parseColor("#64748B"));
                                    sizeText.setTextSize(12);
                                    topRow.addView(sizeText, wrapWrap());

                                    card.addView(topRow, matchWrap());

                                    // Bottom Line: Play/View, Download, Delete Buttons
                                    LinearLayout btnRow = new LinearLayout(MainActivity.this);
                                    btnRow.setOrientation(LinearLayout.HORIZONTAL);
                                    btnRow.setGravity(android.view.Gravity.START);
                                    LinearLayout.LayoutParams btnRowParams = matchWrap();
                                    btnRowParams.setMargins(0, dp(6), 0, 0);
                                    btnRow.setLayoutParams(btnRowParams);

                                    Button viewBtn = new Button(MainActivity.this);
                                    viewBtn.setText(nameLower.endsWith(".mp4") || nameLower.endsWith(".mjpg") || nameLower.endsWith(".avi") ? "▶ Play" : "👁 View");
                                    viewBtn.setTextColor(Color.WHITE);
                                    viewBtn.setTextSize(11);
                                    viewBtn.setAllCaps(false);
                                    viewBtn.setPadding(dp(12), 0, dp(12), 0);
                                    viewBtn.setBackground(round("#10B981", 6, "#10B981"));
                                    viewBtn.setMinHeight(dp(32));
                                    viewBtn.setMinimumHeight(dp(32));
                                    btnRow.addView(viewBtn, wrapWrap());

                                    Button dlBtn = new Button(MainActivity.this);
                                    dlBtn.setText("📥 Download");
                                    dlBtn.setTextColor(Color.WHITE);
                                    dlBtn.setTextSize(11);
                                    dlBtn.setAllCaps(false);
                                    dlBtn.setPadding(dp(12), 0, dp(12), 0);
                                    dlBtn.setBackground(round("#3B82F6", 6, "#3B82F6"));
                                    dlBtn.setMinHeight(dp(32));
                                    dlBtn.setMinimumHeight(dp(32));
                                    LinearLayout.LayoutParams btnParams = wrapWrap();
                                    btnParams.setMargins(dp(8), 0, 0, 0);
                                    dlBtn.setLayoutParams(btnParams);
                                    btnRow.addView(dlBtn);

                                    Button delBtn = new Button(MainActivity.this);
                                    delBtn.setText("🗑 Delete");
                                    delBtn.setTextColor(Color.WHITE);
                                    delBtn.setTextSize(11);
                                    delBtn.setAllCaps(false);
                                    delBtn.setPadding(dp(12), 0, dp(12), 0);
                                    delBtn.setBackground(round("#EF4444", 6, "#EF4444"));
                                    delBtn.setMinHeight(dp(32));
                                    delBtn.setMinimumHeight(dp(32));
                                    LinearLayout.LayoutParams delParams = wrapWrap();
                                    delParams.setMargins(dp(8), 0, 0, 0);
                                    delBtn.setLayoutParams(delParams);
                                    btnRow.addView(delBtn);

                                    card.addView(btnRow, matchWrap());

                                    viewBtn.setOnClickListener(new View.OnClickListener() {
                                        @Override
                                        public void onClick(View v) {
                                            viewMediaFile(name);
                                        }
                                    });

                                    dlBtn.setOnClickListener(new View.OnClickListener() {
                                        @Override
                                        public void onClick(View v) {
                                            downloadFile(name);
                                        }
                                    });

                                    delBtn.setOnClickListener(new View.OnClickListener() {
                                        @Override
                                        public void onClick(View v) {
                                            confirmDeleteMedia(name);
                                        }
                                    });

                                    cameraLogsListContainer.addView(card);

                                    View divider = new View(MainActivity.this);
                                    divider.setBackgroundColor(Color.parseColor("#E2E8F0"));
                                    LinearLayout.LayoutParams divParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1));
                                    cameraLogsListContainer.addView(divider, divParams);
                                }
                            }
                        }

                        if (!foundMedia) {
                            TextView empty = new TextView(MainActivity.this);
                            empty.setText("No saved media found");
                            empty.setTextColor(Color.parseColor("#94A3B8"));
                            empty.setTextSize(14);
                            cameraLogsListContainer.addView(empty, matchWrap());
                        }
                    }
                });
            }

            @Override
            public void onError(String message) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (cameraLogsListContainer == null) return;
                        cameraLogsListContainer.removeAllViews();
                        TextView error = new TextView(MainActivity.this);
                        error.setText("Failed to load media");
                        error.setTextColor(Color.parseColor("#EF4444"));
                        error.setTextSize(14);
                        cameraLogsListContainer.addView(error, matchWrap());
                    }
                });
            }
        });
    }

    private void downloadFile(final String filename) {
        final String urlStr = baseUrl() + "/media/download/" + filename;
        Toast.makeText(this, "Downloading " + filename + "...", Toast.LENGTH_SHORT).show();

        executor.execute(new Runnable() {
            @Override
            public void run() {
                HttpURLConnection connection = null;
                InputStream input = null;
                java.io.OutputStream output = null;
                try {
                    URL url = new URL(urlStr);
                    connection = (HttpURLConnection) url.openConnection();
                    connection.connect();

                    if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
                        showDownloadError("Server returned code " + connection.getResponseCode());
                        return;
                    }

                    input = connection.getInputStream();

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        android.content.ContentValues values = new android.content.ContentValues();
                        values.put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, filename);
                        String fLower = filename.toLowerCase();
                        if (fLower.endsWith(".csv")) {
                            values.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "text/csv");
                        } else if (fLower.endsWith(".jpg") || fLower.endsWith(".jpeg") || fLower.endsWith(".png")) {
                            values.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");
                        } else if (fLower.endsWith(".mp4") || fLower.endsWith(".mjpg") || fLower.endsWith(".mjpeg") || fLower.endsWith(".avi")) {
                            values.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "video/x-motion-jpeg");
                        } else {
                            values.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "application/octet-stream");
                        }
                        values.put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS);

                        android.content.ContentResolver resolver = getContentResolver();
                        android.net.Uri uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
                        if (uri != null) {
                            output = resolver.openOutputStream(uri);
                        }
                    }

                    if (output == null) {
                        java.io.File downloadDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS);
                        if (!downloadDir.exists()) {
                            downloadDir.mkdirs();
                        }
                        java.io.File targetFile = new java.io.File(downloadDir, filename);
                        output = new java.io.FileOutputStream(targetFile);
                    }

                    byte[] data = new byte[4096];
                    int count;
                    while ((count = input.read(data)) != -1) {
                        output.write(data, 0, count);
                    }
                    output.flush();

                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, filename + " successfully downloaded to Downloads folder!", Toast.LENGTH_LONG).show();
                        }
                    });

                } catch (final Exception e) {
                    showDownloadError(e.getMessage() != null ? e.getMessage() : "Unknown error");
                } finally {
                    try {
                        if (output != null) output.close();
                        if (input != null) input.close();
                    } catch (Exception ignored) {
                    }
                    if (connection != null) connection.disconnect();
                }
            }
        });
    }

    private void viewMediaFile(final String filename) {
        final String nameLower = filename.toLowerCase();
        final String mediaUrl = baseUrl() + "/media/" + filename;

        try {
            final android.app.Dialog dialog = new android.app.Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen);
            dialog.getWindow().setBackgroundDrawable(new android.graphics.drawable.ColorDrawable(Color.BLACK));

            RelativeLayout root = new RelativeLayout(this);
            root.setBackgroundColor(Color.BLACK);
            root.setLayoutParams(new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

            // Close Button
            Button closeBtn = new Button(this);
            closeBtn.setText("✖ Close");
            closeBtn.setTextColor(Color.WHITE);
            closeBtn.setTextSize(13);
            closeBtn.setAllCaps(false);
            closeBtn.setBackground(round("#EF4444", 6, "#EF4444"));
            RelativeLayout.LayoutParams closeParams = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            closeParams.addRule(RelativeLayout.ALIGN_PARENT_TOP);
            closeParams.addRule(RelativeLayout.ALIGN_PARENT_END);
            closeParams.setMargins(dp(16), dp(16), dp(16), dp(16));
            closeBtn.setLayoutParams(closeParams);
            closeBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    dialog.dismiss();
                }
            });

            if (nameLower.endsWith(".mp4") || nameLower.endsWith(".mjpg") || nameLower.endsWith(".mjpeg") || nameLower.endsWith(".avi")) {
                // Video Playback: Download to local storage and launch native system Video Player
                Toast.makeText(MainActivity.this, "Opening video in player...", Toast.LENGTH_SHORT).show();
                executor.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            java.io.File downloadDir = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS);
                            final java.io.File file = new java.io.File(downloadDir, filename);

                            if (!file.exists() || file.length() < 100) {
                                URL url = new URL(mediaUrl);
                                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                                conn.setConnectTimeout(10000);
                                conn.setReadTimeout(30000);
                                InputStream in = conn.getInputStream();
                                java.io.FileOutputStream out = new java.io.FileOutputStream(file);
                                byte[] buf = new byte[16384];
                                int r;
                                while ((r = in.read(buf)) != -1) {
                                    out.write(buf, 0, r);
                                }
                                out.flush();
                                out.close();
                                in.close();
                                conn.disconnect();
                            }

                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        android.net.Uri uri;
                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                                            uri = androidx.core.content.FileProvider.getUriForFile(MainActivity.this, getPackageName() + ".fileprovider", file);
                                        } else {
                                            uri = android.net.Uri.fromFile(file);
                                        }
                                        android.content.Intent intent = new android.content.Intent(android.content.Intent.ACTION_VIEW);
                                        intent.setDataAndType(uri, "video/*");
                                        intent.addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION);
                                        startActivity(intent);
                                    } catch (Exception e) {
                                        try {
                                            android.content.Intent intent = new android.content.Intent(android.content.Intent.ACTION_VIEW);
                                            intent.setDataAndType(android.net.Uri.parse(mediaUrl), "video/*");
                                            startActivity(intent);
                                        } catch (Exception ex) {
                                            Toast.makeText(MainActivity.this, "No video player app found on phone", Toast.LENGTH_SHORT).show();
                                        }
                                    }
                                }
                            });
                        } catch (final Exception e) {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(MainActivity.this, "Failed to load video: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                                }
                            });
                        }
                    }
                });
            } else if (nameLower.endsWith(".jpg") || nameLower.endsWith(".jpeg") || nameLower.endsWith(".png")) {
                // Image Viewer with Loading Spinner
                final android.widget.ProgressBar loadingBar = new android.widget.ProgressBar(this);
                RelativeLayout.LayoutParams loadParams = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
                loadParams.addRule(RelativeLayout.CENTER_IN_PARENT);
                loadingBar.setLayoutParams(loadParams);

                final ImageView imageView = new ImageView(this);
                imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
                RelativeLayout.LayoutParams imgParams = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
                imageView.setLayoutParams(imgParams);

                root.addView(imageView);
                root.addView(loadingBar);
                root.addView(closeBtn);
                dialog.setContentView(root);
                dialog.show();

                // Load Image Bytes Asynchronously
                executor.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            URL url = new URL(mediaUrl);
                            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                            conn.setConnectTimeout(8000);
                            conn.setReadTimeout(8000);
                            InputStream is = conn.getInputStream();
                            final Bitmap bmp = BitmapFactory.decodeStream(is);
                            is.close();
                            conn.disconnect();

                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    loadingBar.setVisibility(View.GONE);
                                    if (bmp != null) {
                                        imageView.setImageBitmap(bmp);
                                    } else {
                                        Toast.makeText(MainActivity.this, "Could not decode image", Toast.LENGTH_SHORT).show();
                                    }
                                }
                            });
                        } catch (final Exception e) {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    loadingBar.setVisibility(View.GONE);
                                    Toast.makeText(MainActivity.this, "Failed to load photo: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                                }
                            });
                        }
                    }
                });
            } else {
                Toast.makeText(this, "Downloading file: " + filename, Toast.LENGTH_SHORT).show();
                downloadFile(filename);
            }
        } catch (Exception e) {
            Toast.makeText(this, "Error viewing media: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void confirmDeleteMedia(final String filename) {
        new android.app.AlertDialog.Builder(this)
            .setTitle("Delete File")
            .setMessage("Are you sure you want to delete '" + filename + "' from the helmet?")
            .setPositiveButton("Delete", new android.content.DialogInterface.OnClickListener() {
                @Override
                public void onClick(android.content.DialogInterface dialog, int which) {
                    deleteMediaFile(filename);
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void deleteMediaFile(final String filename) {
        requestJson("/media/delete/" + filename, new JsonCallback() {
            @Override
            public void onSuccess(JSONObject json) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "Deleted " + filename, Toast.LENGTH_SHORT).show();
                        loadCameraSavedLogs();
                    }
                });
            }
            @Override
            public void onError(final String message) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "Failed to delete: " + message, Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
    }

    private void showDownloadError(final String msg) {
        handler.post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(MainActivity.this, "Download failed: " + msg, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void listPairedDevices(final LinearLayout list) {
        list.removeAllViews();

        if (!hasBluetoothPermission()) {
            requestBluetoothPermission();
            return;
        }

        BluetoothAdapter adapter = BluetoothAdapter.getDefaultAdapter();
        if (adapter == null) {
            list.addView(detailRow("Bluetooth", "Not supported"));
            return;
        }

        if (!adapter.isEnabled()) {
            list.addView(detailRow("Bluetooth", "Turn on from phone settings"));
            return;
        }

        Set<BluetoothDevice> devices = adapter.getBondedDevices();
        if (devices.isEmpty()) {
            list.addView(detailRow("Paired devices", "None found"));
            return;
        }

        for (final BluetoothDevice device : devices) {
            Button row = wideButton(safeDeviceName(device), "#173A61");
            LinearLayout.LayoutParams params = matchWrap();
            params.setMargins(0, dp(10), 0, 0);
            list.addView(row, params);
            row.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    connectBluetooth(device);
                }
            });
        }
    }

    private void connectBluetooth(final BluetoothDevice device) {
        if (!hasBluetoothPermission()) {
            requestBluetoothPermission();
            return;
        }

        lastBluetoothState = "Connecting";
        if (bluetoothStatus != null) {
            bluetoothStatus.setText(lastBluetoothState);
        }

        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    closeBluetooth();
                    BluetoothSocket socket = device.createRfcommSocketToServiceRecord(SPP_UUID);
                    socket.connect();
                    bluetoothSocket = socket;
                    lastBluetoothState = "Connected";
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            if (bluetoothStatus != null) {
                                bluetoothStatus.setText(lastBluetoothState);
                            }
                            Toast.makeText(MainActivity.this, "Bluetooth connected", Toast.LENGTH_SHORT).show();
                        }
                    });
                } catch (Exception e) {
                    lastBluetoothState = "Pairing only";
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            if (bluetoothStatus != null) {
                                bluetoothStatus.setText(lastBluetoothState);
                            }
                            Toast.makeText(MainActivity.this, "Pi Bluetooth service not ready", Toast.LENGTH_SHORT).show();
                        }
                    });
                }
            }
        });
    }

    private String safeDeviceName(BluetoothDevice device) {
        if (!hasBluetoothPermission()) {
            return "Bluetooth device";
        }
        String name = device.getName();
        return name == null || name.trim().isEmpty() ? device.getAddress() : name;
    }

    private boolean hasBluetoothPermission() {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.S
                || checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestBluetoothPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            requestPermissions(new String[]{
                    Manifest.permission.BLUETOOTH_CONNECT,
                    Manifest.permission.BLUETOOTH_SCAN
            }, 45);
        }
    }

    private void closeBluetooth() {
        if (bluetoothSocket != null) {
            try {
                bluetoothSocket.close();
            } catch (Exception ignored) {
            }
            bluetoothSocket = null;
        }
    }

    private void requestJson(final String path, final JsonCallback callback) {
        saveHost();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                HttpURLConnection connection = null;
                try {
                    URL url = new URL(baseUrl() + path);
                    connection = (HttpURLConnection) url.openConnection();
                    connection.setConnectTimeout(3500);
                    connection.setReadTimeout(5000);
                    connection.setRequestMethod("GET");

                    int code = connection.getResponseCode();
                    InputStream stream = code >= 200 && code < 300
                            ? connection.getInputStream()
                            : connection.getErrorStream();
                    String body = readAll(stream);

                    if (code < 200 || code >= 300) {
                        throw new Exception(body.isEmpty() ? "HTTP " + code : body);
                    }

                    final JSONObject json = new JSONObject(body);
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            callback.onSuccess(json);
                        }
                    });
                } catch (final Exception e) {
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            callback.onError(e.getMessage() == null ? "Unknown error" : e.getMessage());
                        }
                    });
                } finally {
                    if (connection != null) {
                        connection.disconnect();
                    }
                }
            }
        });
    }

    private String readAll(InputStream stream) throws Exception {
        if (stream == null) {
            return "";
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        StringBuilder builder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            builder.append(line);
        }
        return builder.toString();
    }

    private String baseUrl() {
        String host = hostInput != null
                ? hostInput.getText().toString().trim()
                : getSharedPreferences(PREFS, MODE_PRIVATE).getString("host", DEFAULT_HOST);
        if (host.startsWith("http://") || host.startsWith("https://")) {
            return host.replaceAll("/+$", "");
        }
        return "http://" + host.replaceAll("/+$", "");
    }

    private void saveHost() {
        if (hostInput == null) {
            return;
        }
        getSharedPreferences(PREFS, MODE_PRIVATE)
                .edit()
                .putString("host", hostInput.getText().toString().trim())
                .apply();
    }

    private void setTopStatus(String value) {
        if (topStatus != null) {
            topStatus.setText(value);
        }
    }

    private TextView dashboardTile(GridLayout grid, String label, String value, String accent) {
        LinearLayout tile = new LinearLayout(this);
        tile.setOrientation(LinearLayout.VERTICAL);
        tile.setPadding(dp(14), dp(14), dp(14), dp(12));
        tile.setBackground(round("#FFFFFF", 8, "#E5E7EB"));

        TextView dot = iconCircle("", accent, "#FFFFFF", 1);
        LinearLayout.LayoutParams dotParams = new LinearLayout.LayoutParams(dp(10), dp(10));
        tile.addView(dot, dotParams);

        TextView labelView = text(label, 13, "#667085", Typeface.BOLD);
        LinearLayout.LayoutParams labelParams = matchWrap();
        labelParams.setMargins(0, dp(10), 0, dp(4));
        tile.addView(labelView, labelParams);

        TextView valueView = text(value, 17, "#17212B", Typeface.BOLD);
        tile.addView(valueView);

        GridLayout.LayoutParams params = new GridLayout.LayoutParams();
        params.width = 0;
        params.height = dp(116);
        params.columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1f);
        params.setMargins(dp(4), dp(4), dp(4), dp(4));
        grid.addView(tile, params);
        return valueView;
    }

    private TextView addMetric(GridLayout grid, String label, String initialValue, String accent) {
        LinearLayout tile = new LinearLayout(this);
        tile.setOrientation(LinearLayout.VERTICAL);
        tile.setPadding(dp(13), dp(13), dp(13), dp(12));
        tile.setBackground(round("#FFFFFF", 8, "#E5E7EB"));

        TextView labelView = text(label, 12, "#667085", Typeface.BOLD);
        tile.addView(labelView);

        TextView valueView = text(initialValue, 19, accent, Typeface.BOLD);
        LinearLayout.LayoutParams valueParams = matchWrap();
        valueParams.setMargins(0, dp(8), 0, 0);
        tile.addView(valueView, valueParams);

        GridLayout.LayoutParams params = new GridLayout.LayoutParams();
        params.width = 0;
        params.height = dp(92);
        params.columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1f);
        params.setMargins(dp(4), dp(4), dp(4), dp(4));
        grid.addView(tile, params);
        return valueView;
    }

    private TextView detailRow(String label, String value) {
        TextView row = text(label + "  " + value, 13, "#667085", Typeface.NORMAL);
        LinearLayout.LayoutParams params = matchWrap();
        params.setMargins(0, dp(10), 0, 0);
        row.setLayoutParams(params);
        return row;
    }

    private TextView sectionTitle(String title) {
        return text(title, 16, "#17212B", Typeface.BOLD);
    }

    private void addGridButton(GridLayout grid, Button button) {
        GridLayout.LayoutParams params = new GridLayout.LayoutParams();
        params.width = 0;
        params.height = dp(54);
        params.columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1f);
        params.setMargins(dp(4), dp(4), dp(4), dp(4));
        grid.addView(button, params);
    }

    private Button wideButton(String label, String color) {
        return actionButton(label, color, 0);
    }

    private Button actionButton(String label, String color, int drawableRes) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextColor(Color.WHITE);
        button.setTextSize(14);
        button.setAllCaps(false);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setGravity(Gravity.CENTER);
        button.setMinHeight(0);
        button.setMinimumHeight(0);
        button.setPadding(dp(10), 0, dp(10), 0);
        button.setBackground(round(color, 8, color));
        if (drawableRes != 0) {
            button.setCompoundDrawablesWithIntrinsicBounds(drawableRes, 0, 0, 0);
            button.setCompoundDrawablePadding(dp(7));
        }
        return button;
    }

    private LinearLayout panel() {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(16), dp(16), dp(16), dp(16));
        panel.setBackground(round("#FFFFFF", 8, "#E5E7EB"));
        return panel;
    }

    private TextView pill(String value, String background, String foreground) {
        TextView textView = text(value, 13, foreground, Typeface.BOLD);
        textView.setGravity(Gravity.CENTER);
        textView.setPadding(dp(12), dp(7), dp(12), dp(7));
        textView.setBackground(round(background, 99, background));
        return textView;
    }

    private TextView iconCircle(String value, String background, String foreground, int size) {
        TextView textView = text(value, size, foreground, Typeface.BOLD);
        textView.setGravity(Gravity.CENTER);
        textView.setBackground(round(background, 99, background));
        return textView;
    }

    private TextView text(String value, int sp, String color, int style) {
        TextView textView = new TextView(this);
        textView.setText(value);
        textView.setTextSize(sp);
        textView.setTextColor(Color.parseColor(color));
        textView.setTypeface(Typeface.DEFAULT, style);
        textView.setIncludeFontPadding(true);
        return textView;
    }

    private GradientDrawable inputBackground() {
        GradientDrawable drawable = round("#FFFFFF", 8, "#D0D5DD");
        drawable.setStroke(dp(1), Color.parseColor("#D0D5DD"));
        return drawable;
    }

    private GradientDrawable round(String fill, int radiusDp, String stroke) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.parseColor(fill));
        drawable.setCornerRadius(dp(radiusDp));
        drawable.setStroke(dp(1), Color.parseColor(stroke));
        return drawable;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
    }

    private LinearLayout.LayoutParams wrapWrap() {
        return new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
    }

    private LinearLayout.LayoutParams sectionParams() {
        LinearLayout.LayoutParams params = matchWrap();
        params.setMargins(0, 0, 0, dp(14));
        return params;
    }

    private int getVideoHeight() {
        int width = getResources().getDisplayMetrics().widthPixels - dp(40);
        return Math.max(dp(300), (int) (width * 3f / 4f));
    }

    private String format(double value, String unit) {
        return String.format(Locale.US, "%.3f%s", value, unit);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private int statusBarHeight() {
        int resourceId = getResources().getIdentifier("status_bar_height", "dimen", "android");
        if (resourceId > 0) {
            return getResources().getDimensionPixelSize(resourceId);
        }
        return dp(24);
    }

    private interface JsonCallback {
        void onSuccess(JSONObject json);

        void onError(String message);
    }

    private static class Cube3DView extends View {
        private float targetPitch = 0;
        private float targetRoll = 0;
        private float targetYaw = 0;

        private float currentPitch = 0;
        private float currentRoll = 0;
        private float currentYaw = 0;

        private final android.graphics.Paint paint = new android.graphics.Paint();
        private final android.os.Handler animHandler = new android.os.Handler(android.os.Looper.getMainLooper());
        private final Runnable animator = new Runnable() {
            @Override
            public void run() {
                // Smooth interpolation (easing filter)
                currentPitch += (targetPitch - currentPitch) * 0.15f;
                currentRoll += (targetRoll - currentRoll) * 0.15f;
                currentYaw += (targetYaw - currentYaw) * 0.15f;
                invalidate();
                animHandler.postDelayed(this, 16); // ~60 FPS
            }
        };

        public Cube3DView(android.content.Context context) {
            super(context);
            animHandler.post(animator);
        }

        public void updateOrientation(float pitch, float roll, float yaw) {
            this.targetPitch = pitch;
            this.targetRoll = roll;
            this.targetYaw = yaw;
        }

        @Override
        protected void onDetachedFromWindow() {
            super.onDetachedFromWindow();
            animHandler.removeCallbacks(animator);
        }

        @Override
        protected void onDraw(android.graphics.Canvas canvas) {
            super.onDraw(canvas);
            int cx = getWidth() / 2;
            int cy = getHeight() / 2;
            int size = Math.min(getWidth(), getHeight()) / 5;
            if (size <= 0) return;

            // 1. Draw a high-tech HUD overlay in the background
            paint.setAntiAlias(true);
            paint.setStyle(android.graphics.Paint.Style.STROKE);
            paint.setColor(0x3338BDF8); // faint cyan
            paint.setStrokeWidth(dp(3));
            canvas.drawCircle(cx, cy, size * 2.2f, paint);

            paint.setColor(0x1F38BDF8);
            paint.setStrokeWidth(dp(1));
            canvas.drawCircle(cx, cy, size * 2.0f, paint);

            // Draw HUD ticks on the outer circle
            paint.setColor(0x6638BDF8);
            paint.setStrokeWidth(dp(2));
            for (int a = 0; a < 360; a += 30) {
                double rad = Math.toRadians(a);
                float startRadius = size * 2.1f;
                float endRadius = size * 2.2f;
                float xStart = cx + (float) (Math.cos(rad) * startRadius);
                float yStart = cy + (float) (Math.sin(rad) * startRadius);
                float xEnd = cx + (float) (Math.cos(rad) * endRadius);
                float yEnd = cy + (float) (Math.sin(rad) * endRadius);
                canvas.drawLine(xStart, yStart, xEnd, yEnd, paint);
            }

            // Real-time HUD stats (positioned on edges to prevent overlaps)
            float textY1 = dp(16);
            float textY2 = textY1 + dp(14);

            paint.setStyle(android.graphics.Paint.Style.FILL);
            paint.setColor(0xFF0EA5E9); // Bright sky blue
            paint.setTextSize(dp(11));
            paint.setTypeface(android.graphics.Typeface.MONOSPACE);
            paint.setTextAlign(android.graphics.Paint.Align.LEFT);
            canvas.drawText(String.format(Locale.US, "PITCH: %+05.1f°", currentPitch), dp(16), textY1, paint);
            canvas.drawText(String.format(Locale.US, "ROLL:  %+05.1f°", currentRoll), dp(16), textY2, paint);

            paint.setTextAlign(android.graphics.Paint.Align.RIGHT);
            canvas.drawText("SYS: ACTIVE", getWidth() - dp(16), textY1, paint);
            canvas.drawText("TELEM: OK", getWidth() - dp(16), textY2, paint);

            // 2. Compute 3D projection using smooth interpolated angles
            float[][] vertices = {
                    {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                    {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}
            };

            float[] rotatedZ = new float[8];
            float[][] projected = new float[8][2];
            double radPitch = Math.toRadians(currentPitch);
            double radRoll = Math.toRadians(currentRoll);
            double radYaw = Math.toRadians(currentYaw);

            for (int i = 0; i < 8; i++) {
                float x = vertices[i][0] * size;
                float y = vertices[i][1] * size;
                float z = vertices[i][2] * size;

                // Pitch (rotate around Y)
                float x1 = (float) (x * Math.cos(radPitch) + z * Math.sin(radPitch));
                float y1 = y;
                float z1 = (float) (-x * Math.sin(radPitch) + z * Math.cos(radPitch));

                // Roll (rotate around X)
                float x2 = x1;
                float y2 = (float) (y1 * Math.cos(radRoll) - z1 * Math.sin(radRoll));
                float z2 = (float) (y1 * Math.sin(radRoll) + z1 * Math.cos(radRoll));

                // Yaw (rotate around Z)
                float x3 = (float) (x2 * Math.cos(radYaw) - y2 * Math.sin(radYaw));
                float y3 = (float) (x2 * Math.sin(radYaw) + y2 * Math.cos(radYaw));
                float z3 = z2;

                rotatedZ[i] = z3;

                float distance = size * 3;
                float sz = distance / (distance + z3);
                projected[i][0] = cx + x3 * sz;
                projected[i][1] = cy + y3 * sz;
            }

            // Draw a flat XZ horizon ring tilted under the cube
            android.graphics.Path circlePath = new android.graphics.Path();
            float radius = size * 1.4f;
            float circleY = size * 0.9f;
            boolean first = true;
            for (int angle = 0; angle <= 360; angle += 15) {
                double rad = Math.toRadians(angle);
                float x = (float) (Math.cos(rad) * radius);
                float z = (float) (Math.sin(rad) * radius);
                float y = circleY;

                float x1 = (float) (x * Math.cos(radPitch) + z * Math.sin(radPitch));
                float y1 = y;
                float z1 = (float) (-x * Math.sin(radPitch) + z * Math.cos(radPitch));

                float x2 = x1;
                float y2 = (float) (y1 * Math.cos(radRoll) - z1 * Math.sin(radRoll));
                float z2 = (float) (y1 * Math.sin(radRoll) + z1 * Math.cos(radRoll));

                float x3 = (float) (x2 * Math.cos(radYaw) - y2 * Math.sin(radYaw));
                float y3 = (float) (x2 * Math.sin(radYaw) + y2 * Math.cos(radYaw));

                float distance = size * 3;
                float sz = distance / (distance + z2);
                float px = cx + x3 * sz;
                float py = cy + y3 * sz;

                if (first) {
                    circlePath.moveTo(px, py);
                    first = false;
                } else {
                    circlePath.lineTo(px, py);
                }
            }
            paint.setStyle(android.graphics.Paint.Style.STROKE);
            paint.setColor(0x800EA5E9); // Translucent bright cyan
            paint.setStrokeWidth(dp(2));
            canvas.drawPath(circlePath, paint);

            // 3. Draw 3D Solid Cube with depth sorting (Painter's Algorithm)
            int[][] faces = {
                    {0, 1, 2, 3}, // Back
                    {4, 5, 6, 7}, // Front
                    {0, 3, 7, 4}, // Left
                    {1, 2, 6, 5}, // Right
                    {0, 1, 5, 4}, // Top
                    {3, 2, 6, 7}  // Bottom
            };

            int[] faceColors = {
                    0xAA1E3A8A, // Back (Deep Navy)
                    0xAA3B82F6, // Front (Neon Blue)
                    0xAA10B981, // Left (Emerald Green)
                    0xAA047857, // Right (Teal Green)
                    0xAAF59E0B, // Top (Orange Gold)
                    0xAAB45309  // Bottom (Amber Bronze)
            };

            String[] faceLabels = {
                    "BACK", "FRONT", "LEFT", "RIGHT", "TOP", "BOTTOM"
            };

            Integer[] faceOrder = {0, 1, 2, 3, 4, 5};
            final float[] faceZ = new float[6];
            for (int f = 0; f < 6; f++) {
                float sumZ = 0;
                for (int vi : faces[f]) {
                    sumZ += rotatedZ[vi];
                }
                faceZ[f] = sumZ / 4f;
            }

            java.util.Arrays.sort(faceOrder, new java.util.Comparator<Integer>() {
                @Override
                public int compare(Integer a, Integer b) {
                    return Float.compare(faceZ[b], faceZ[a]); // Furthest face first
                }
            });

            android.graphics.Path path = new android.graphics.Path();
            for (int f : faceOrder) {
                path.reset();
                int[] face = faces[f];
                path.moveTo(projected[face[0]][0], projected[face[0]][1]);
                for (int i = 1; i < 4; i++) {
                    path.lineTo(projected[face[i]][0], projected[face[i]][1]);
                }
                path.close();

                // Draw filled face
                paint.setStyle(android.graphics.Paint.Style.FILL);
                paint.setColor(faceColors[f]);
                canvas.drawPath(path, paint);

                // Draw borders
                paint.setStyle(android.graphics.Paint.Style.STROKE);
                paint.setColor(Color.WHITE);
                paint.setStrokeWidth(dp(2));
                canvas.drawPath(path, paint);

                // Draw face label
                paint.setStyle(android.graphics.Paint.Style.FILL);
                paint.setColor(Color.WHITE);
                paint.setTextSize(dp(11));
                paint.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
                paint.setTextAlign(android.graphics.Paint.Align.CENTER);

                float sumX = 0, sumY = 0;
                for (int vi : face) {
                    sumX += projected[vi][0];
                    sumY += projected[vi][1];
                }
                canvas.drawText(faceLabels[f], sumX / 4f, (sumY / 4f) + dp(4), paint);
            }

            // Draw HUD target crosshairs in center
            paint.setColor(0x8038BDF8);
            paint.setStrokeWidth(dp(1.5f));
            canvas.drawLine(cx - dp(15), cy, cx + dp(15), cy, paint);
            canvas.drawLine(cx, cy - dp(15), cx, cy + dp(15), paint);
        }

        private float dp(float value) {
            return value * getResources().getDisplayMetrics().density;
        }

    }

    private Bitmap drawDetectionsOnBitmap(Bitmap src, java.util.List<Detection> detections) {
        if (src == null) return null;
        Bitmap mutable = src.copy(Bitmap.Config.ARGB_8888, true);
        android.graphics.Canvas canvas = new android.graphics.Canvas(mutable);
        
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setStyle(android.graphics.Paint.Style.STROKE);
        paint.setStrokeWidth(2.5f);

        android.graphics.Paint textPaint = new android.graphics.Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(11f);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);
        textPaint.setStyle(android.graphics.Paint.Style.FILL);

        android.graphics.Paint textBgPaint = new android.graphics.Paint();
        textBgPaint.setStyle(android.graphics.Paint.Style.FILL);

        // Draw real-time HUD status banner if PPE stream is active
        if (ppeStreamOn) {
            java.util.List<String> hudViolations = new java.util.ArrayList<>();
            if (alertNoHelmet) hudViolations.add("NO HELMET");
            else if (alertChinstrapOpen) hudViolations.add("CHINSTRAP UNFASTENED");
            
            if (alertNoGoggles) hudViolations.add("NO GOGGLES");
            if (alertNoMask) hudViolations.add("NO MASK");
            
            if (alertNoVest) hudViolations.add("NO VEST");
            if (alertMissingPpe) hudViolations.add("MISSING GLOVES");

            android.graphics.Paint bannerPaint = new android.graphics.Paint();
            bannerPaint.setStyle(android.graphics.Paint.Style.FILL);
            
            android.graphics.Paint bannerTextPaint = new android.graphics.Paint();
            bannerTextPaint.setColor(Color.WHITE);
            bannerTextPaint.setTextSize(13f);
            bannerTextPaint.setTypeface(Typeface.DEFAULT_BOLD);
            bannerTextPaint.setTextAlign(android.graphics.Paint.Align.CENTER);

            String warningText;
            if (isScanningActive) {
                bannerPaint.setColor(Color.parseColor("#E6D97604")); // Deep translucent orange/amber
                long elapsed = System.currentTimeMillis() - scanStartTime;
                int secondsLeft = Math.max(0, 15 - (int)(elapsed / 1000));
                warningText = "⌛ SCANNING: PLEASE STAND STILL (" + secondsLeft + "s LEFT)";
            } else if (scanResultsAnnounced) {
                if (hudViolations.isEmpty()) {
                    bannerPaint.setColor(Color.parseColor("#D910B981")); // Translucent green
                    warningText = "✅ SAFE: ACCESS GRANTED";
                } else {
                    bannerPaint.setColor(Color.parseColor("#D9EF4444")); // Translucent red
                    StringBuilder sb = new StringBuilder("⚠️ WARNING: ");
                    for (int i = 0; i < hudViolations.size(); i++) {
                        sb.append(hudViolations.get(i));
                        if (i < hudViolations.size() - 1) {
                            sb.append(" | ");
                        }
                    }
                    warningText = sb.toString();
                }
            } else {
                bannerPaint.setColor(Color.parseColor("#E64B5563")); // Translucent cool grey
                warningText = "⌛ WAITING FOR WORKER...";
            }

            float bannerHeight = 28f;
            float textY = 19f;
            canvas.drawRect(0, 0, mutable.getWidth(), bannerHeight, bannerPaint);
            canvas.drawText(warningText, mutable.getWidth() / 2f, textY, bannerTextPaint);
        }

        // Draw bounding boxes for active detections
        int[] colors = {
            Color.parseColor("#F59E0B"), // 0: Boots - Orange
            Color.parseColor("#3B82F6"), // 1: Gloves - Blue
            Color.parseColor("#10B981"), // 2: Helmet - Green
            Color.parseColor("#EF4444"), // 3: INCORRECT-MASK - Red
            Color.parseColor("#10B981"), // 4: Mask - Green
            Color.parseColor("#10B981")  // 5: Vest - Green
        };

        synchronized (detections) {
            for (Detection det : detections) {
                if (det.classId < 0 || det.classId >= colors.length) continue;
                int color = colors[det.classId];
                String textLabel = (det.customLabel != null) ? det.customLabel : ppeLabels[det.classId];

                paint.setColor(color);
                textBgPaint.setColor(color);

                // Draw box
                canvas.drawRect(det.left, det.top, det.right, det.bottom, paint);

                // Label text
                String fullLabel = textLabel + String.format(Locale.US, " %.0f%%", det.confidence * 100);
                float textWidth = textPaint.measureText(fullLabel);
                float textHeight = textPaint.getTextSize();
                canvas.drawRect(det.left, det.top - textHeight - 3, det.left + textWidth + 6, det.top, textBgPaint);
                canvas.drawText(fullLabel, det.left + 3, det.top - 3, textPaint);
            }
        }

        return mutable;
    }

        /**
     * CORE AI ENGINE — Two-Stage PPE Compliance Inference
     * Stage 1: Google ML Kit Face + Object Gating (derives Head & Body ROIs, filters non-human frames).
     * Stage 2: YOLOv8 TFLite 640x640 model run on each ROI crop (4 CPU threads).
     * Post-processing: Remaps HuggingFace class order → App class order, applies per-class
     * confidence thresholds, and enforces anatomical rules (Head Crop vest filter, Vest torso Y-filter).
     * Accumulation: 15-second compliance window tracks detections → triggers TTS & report export.
     */
    private void runInferenceOnBitmap(Bitmap bitmap) {
        if (!isModelLoaded || bitmap == null) return;
        try {
            // Preprocessor for Stage 2 (320x320)
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new org.tensorflow.lite.support.common.ops.NormalizeOp(0.0f, 255.0f))
                    .build();

            // 1. Stage 1: Object Locator using ML Kit (Worker & Head)
            java.util.List<Stage1Detection> stage1Regions = new java.util.ArrayList<>();
            boolean realPersonDetected = false;
            com.google.mlkit.vision.common.InputImage image =
                    com.google.mlkit.vision.common.InputImage.fromBitmap(bitmap, 0);

            if (mlKitFaceDetector != null) {
                try {
                    com.google.android.gms.tasks.Task<java.util.List<com.google.mlkit.vision.face.Face>> faceTask =
                            mlKitFaceDetector.process(image);
                    java.util.List<com.google.mlkit.vision.face.Face> faces =
                            com.google.android.gms.tasks.Tasks.await(faceTask);

                    if (faces != null && !faces.isEmpty()) {
                        for (com.google.mlkit.vision.face.Face face : faces) {
                            android.graphics.Rect rect = face.getBoundingBox();
                            float faceLeft = Math.max(0, rect.left);
                            float faceTop = Math.max(0, rect.top);
                            float faceRight = Math.min(bitmap.getWidth(), rect.right);
                            float faceBottom = Math.min(bitmap.getHeight(), rect.bottom);
                            float faceW = faceRight - faceLeft;
                            float faceH = faceBottom - faceTop;

                            if (faceW < bitmap.getWidth() * 0.06f || faceH < bitmap.getHeight() * 0.07f) {
                                continue;
                            }

                            realPersonDetected = true;
                            float faceCx = (faceLeft + faceRight) / 2f;

                            float headLeft = Math.max(0, faceLeft - faceW * 0.55f);
                            float headTop = Math.max(0, faceTop - faceH * 1.35f);
                            float headRight = Math.min(bitmap.getWidth(), faceRight + faceW * 0.55f);
                            float headBottom = Math.min(bitmap.getHeight(), faceBottom + faceH * 0.40f);

                            float bodyLeft = Math.max(0, faceCx - faceW * 2.5f);
                            float bodyTop = Math.max(0, faceTop - faceH * 0.40f);
                            float bodyRight = Math.min(bitmap.getWidth(), faceCx + faceW * 2.5f);
                            float bodyBottom = Math.min(bitmap.getHeight(), faceBottom + faceH * 4.5f);

                            stage1Regions.add(new Stage1Detection(bodyLeft, bodyTop, bodyRight, bodyBottom, 0, 1.0f));
                            stage1Regions.add(new Stage1Detection(headLeft, headTop, headRight, headBottom, 1, 1.0f));
                        }
                    }
                } catch (Exception e) {
                    android.util.Log.e("SmartHelmetAI", "Face detector error: " + e.getMessage());
                }
            }

            if (mlKitDetector != null) {
                try {
                    com.google.android.gms.tasks.Task<java.util.List<com.google.mlkit.vision.objects.DetectedObject>> task =
                            mlKitDetector.process(image);
                    java.util.List<com.google.mlkit.vision.objects.DetectedObject> objects =
                            com.google.android.gms.tasks.Tasks.await(task);

                    if (objects != null && !objects.isEmpty()) {
                        for (com.google.mlkit.vision.objects.DetectedObject obj : objects) {
                            boolean isNonPerson = false;
                            for (com.google.mlkit.vision.objects.DetectedObject.Label label : obj.getLabels()) {
                                String text = label.getText();
                                if (text != null && (text.equalsIgnoreCase("Place") || 
                                                     text.equalsIgnoreCase("Plant") || 
                                                     text.equalsIgnoreCase("Food"))) {
                                    isNonPerson = true;
                                    break;
                                }
                            }
                            if (isNonPerson) {
                                continue;
                            }

                            android.graphics.Rect rect = obj.getBoundingBox();
                            float left = Math.max(0, rect.left);
                            float top = Math.max(0, rect.top);
                            float right = Math.min(bitmap.getWidth(), rect.right);
                            float bottom = Math.min(bitmap.getHeight(), rect.bottom);

                            if (right > left && bottom > top) {
                                float width = right - left;
                                float height = bottom - top;
                                float aspect = height / Math.max(1f, width);
                                float areaRatio = (width * height) / Math.max(1f, bitmap.getWidth() * bitmap.getHeight());

                                boolean likelyPersonRegion =
                                        width >= bitmap.getWidth() * 0.08f &&
                                        height >= bitmap.getHeight() * 0.18f &&
                                        areaRatio >= 0.025f;
                                if (!likelyPersonRegion) {
                                    continue;
                                }
                                realPersonDetected = true;

                                float expandedLeft = Math.max(0, left - width * 0.15f);
                                float expandedRight = Math.min(bitmap.getWidth(), right + width * 0.15f);
                                float expandedBottom = Math.min(bitmap.getHeight(), bottom + height * 0.35f);

                                stage1Regions.add(new Stage1Detection(expandedLeft, top, expandedRight, expandedBottom, 0, 1.0f));

                                float headHeight = height * 0.50f;
                                float headTopEst = Math.max(0, top - height * 0.20f);
                                float headBottomEst = top + headHeight;
                                stage1Regions.add(new Stage1Detection(left, headTopEst, right, headBottomEst, 1, 1.0f));
                            }
                        }
                    }
                } catch (Exception e) {
                    android.util.Log.e("SmartHelmetAI", "ML Kit inference error: " + e.getMessage());
                }
            }

            if (!stage1Regions.isEmpty()) {
                stage1Regions = runStage1Nms(stage1Regions);
            }

            // Always add full frame pass when a person is detected to ensure overall context is analyzed
            if (realPersonDetected) {
                stage1Regions.add(new Stage1Detection(0, 0, bitmap.getWidth(), bitmap.getHeight(), 0, 1.0f));
            }

            long now = System.currentTimeMillis();
            if (realPersonDetected) {
                lastPersonDetectedTime = now;
            }
            boolean personCurrentlyInFrame = (lastPersonDetectedTime > 0 && (now - lastPersonDetectedTime < PPE_PERSON_ABSENT_RESET_MS));
            if (personCurrentlyInFrame) {
                if (personFirstDetectedTime == 0) {
                    personFirstDetectedTime = now;
                }
            } else {
                personFirstDetectedTime = 0;
            }

            // 2. Run Stage 2 on each region detected by Stage 1
            java.util.List<Detection> newDetections = new java.util.ArrayList<>();
            boolean overallHelmetAlert = false;
            boolean overallNoGogglesAlert = false;
            boolean overallNoMaskAlert = false;
            boolean overallNoVestAlert = false;
            boolean overallMissingPpeAlert = false;
            boolean overallChinstrapAlert = false;
            boolean personDetected = false;

            for (Stage1Detection region : stage1Regions) {
                int cropX = (int) region.left;
                int cropY = (int) region.top;
                int cropW = (int) (region.right - region.left);
                int cropH = (int) (region.bottom - region.top);

                if (cropW <= 0 || cropH <= 0) continue;
                personDetected = true;

                // Crop region
                Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, cropX, cropY, cropW, cropH);

                TensorImage stage2Image = new TensorImage(DataType.FLOAT32);
                stage2Image.load(croppedBitmap);
                stage2Image = imageProcessor.process(stage2Image);

                // Stage 2 model shape: [1 x 10 x 8400] for 6-class 640x640 model
                float[][][] stage2Out = new float[1][10][8400];
                tflite.run(stage2Image.getBuffer(), stage2Out);

                boolean hasHelmet = false;
                boolean hasVest = false;
                boolean hasGloves = false;
                boolean hasMask = false;
                boolean hasGoggles = false;

                java.util.List<Detection> localDetections = new java.util.ArrayList<>();

                for (int boxIdx = 0; boxIdx < 8400; boxIdx++) {
                    int modelClassId = -1;
                    float maxConf = -1.0f;
                    for (int c = 0; c < 6; c++) {
                        float conf = stage2Out[0][4 + c][boxIdx];
                        if (conf > maxConf) {
                            maxConf = conf;
                            modelClassId = c;
                        }
                    }

                    if (modelClassId != -1) {
                        // Remap HuggingFace model output index to match app's 6-class index order:
                        // HF classes: 0=Gloves, 1=Vest, 2=Goggles, 3=Helmet, 4=Mask, 5=Safety_shoe
                        // App classes: 0=Boots, 1=Gloves, 2=Helmet, 3=Goggles, 4=Mask, 5=Vest
                        if (modelClassId == 0) modelClassId = 1;       // HF Gloves -> App Gloves
                        else if (modelClassId == 1) modelClassId = 5;  // HF Vest -> App Vest
                        else if (modelClassId == 2) modelClassId = 3;  // HF Goggles -> App Goggles
                        else if (modelClassId == 3) modelClassId = 2;  // HF Helmet -> App Helmet
                        else if (modelClassId == 4) modelClassId = 4;  // HF Mask -> App Mask
                        else if (modelClassId == 5) modelClassId = 0;  // HF Safety_shoe -> App Boots

                        // 1. Head crop filter: Inside Head crops (classId == 1), Vest (5) and Boots (0) cannot exist!
                        if (region.classId == 1 && (modelClassId == 5 || modelClassId == 0)) {
                            continue;
                        }

                        // 2. Anatomical filter for Vest: Vest center must be in torso region (cy >= 250 in 640x640 space)
                        if (modelClassId == 5) {
                            float cy = stage2Out[0][1][boxIdx];
                            if (cy < 250f) {
                                continue;
                            }
                        }

                        float cutoff = getConfidenceThreshold(modelClassId);
                        if (maxConf > cutoff) {
                            float cx = stage2Out[0][0][boxIdx];
                            float cy = stage2Out[0][1][boxIdx];
                            float w = stage2Out[0][2][boxIdx];
                            float h = stage2Out[0][3][boxIdx];

                            // New model outputs PIXEL coords in [0-640] space
                            // Correct: divide by 640 then scale to cropW/cropH
                            float left = ((cx - w / 2f) / 640f) * cropW;
                            float top  = ((cy - h / 2f) / 640f) * cropH;
                            float right  = ((cx + w / 2f) / 640f) * cropW;
                            float bottom = ((cy + h / 2f) / 640f) * cropH;

                            // Translate back to global coordinates
                            float fullLeft = cropX + left;
                            float fullTop = cropY + top;
                            float fullRight = cropX + right;
                            float fullBottom = cropY + bottom;

                            // Clip bounds
                            fullLeft = Math.max(0, Math.min(fullLeft, bitmap.getWidth()));
                            fullTop = Math.max(0, Math.min(fullTop, bitmap.getHeight()));
                            fullRight = Math.max(0, Math.min(fullRight, bitmap.getWidth()));
                            fullBottom = Math.max(0, Math.min(fullBottom, bitmap.getHeight()));

                            // Directly use modelClassId as ui classId and label mapping
                            localDetections.add(new Detection(fullLeft, fullTop, fullRight, fullBottom, modelClassId, maxConf));

                            // Update local booleans for geometric compliance logic
                            if (modelClassId == 1) hasGloves = true;
                            else if (modelClassId == 2) hasHelmet = true;
                            else if (modelClassId == 3) hasGoggles = true;
                            else if (modelClassId == 4) hasMask = true;
                            else if (modelClassId == 5) hasVest = true;
                        }
                    }
                }

                java.util.List<Detection> filteredLocal = runLocalNms(localDetections);
                newDetections.addAll(filteredLocal);

            }

            // Simple NMS across all collected detections
            java.util.List<Detection> filteredDetections = runLocalNms(newDetections);

            // Update global latestDetections using cached detections to prevent flickering
            long nowTime = System.currentTimeMillis();
            synchronized (latestDetections) {
                // 1. Remove expired cached detections (older than 800ms)
                java.util.Iterator<CachedDetection> iterator = cachedDetections.iterator();
                while (iterator.hasNext()) {
                    if (nowTime - iterator.next().timestamp > 800) {
                        iterator.remove();
                    }
                }
                
                // 2. Add or update with new detections
                for (Detection det : filteredDetections) {
                    boolean updated = false;
                    for (CachedDetection cached : cachedDetections) {
                        if (cached.detection.classId == det.classId) {
                            cached.detection = det;
                            cached.timestamp = nowTime;
                            updated = true;
                            break;
                        }
                    }
                    if (!updated) {
                        cachedDetections.add(new CachedDetection(det, nowTime));
                    }
                }
                
                // 3. Rebuild latestDetections from the active cache
                latestDetections.clear();
                for (CachedDetection cached : cachedDetections) {
                    latestDetections.add(cached.detection);
                }

                // 4. Query the active cache and handle safety scan logic
                boolean personStableForScan = personCurrentlyInFrame &&
                        personFirstDetectedTime > 0 &&
                        (nowTime - personFirstDetectedTime >= 800);
                if (personStableForScan && !isScanningActive && !scanResultsAnnounced) {
                    // Start a new 15-second scan!
                    isScanningActive = true;
                    scanStartTime = nowTime;
                    scanAnnouncementCount = 0; // Reset announcement count!
                    scanDetectedHelmet = false;
                    scanDetectedVest = false;
                    scanDetectedGloves = false;
                    scanDetectedMask = false;
                    scanDetectedGoggles = false;
                    java.util.Arrays.fill(scanBestConfidence, 0f);
                    java.util.Arrays.fill(scanHitCount, 0);
                    java.util.Arrays.fill(scanFirstHitTime, 0L);
                    
                    // Notify user that scanning has started
                    if (isTtsReady && ttsEngine != null) {
                        ttsEngine.speak("Starting safety scan. Please stand still for fifteen seconds.", 
                                        android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "ScanStartSpeech");
                    }
                    setPpeInferenceStatus("Person detected - scanning for 15 seconds");
                }
                
                if (isScanningActive) {
                    // Accumulate unique frame detections during active scan window
                    for (Detection det : filteredDetections) {
                        rememberScanDetection(det.classId, det.confidence);
                    }
                } else if (!personCurrentlyInFrame && (scanResultsAnnounced || isScanningActive)) {
                    resetPpeScanState(true);
                    setPpeInferenceStatus("Waiting for person");
                }

                // Determine alerts based on scan results
                if (scanResultsAnnounced) {
                    overallHelmetAlert = !scanDetectedHelmet;
                    overallNoGogglesAlert = !scanDetectedGoggles;
                    overallNoMaskAlert = !scanDetectedMask;
                    overallNoVestAlert = !scanDetectedVest;
                    overallMissingPpeAlert = !scanDetectedGloves;
                } else {
                    // Keep alerts off during active scanning to avoid premature warnings
                    overallHelmetAlert = false;
                    overallNoGogglesAlert = false;
                    overallNoMaskAlert = false;
                    overallNoVestAlert = false;
                    overallMissingPpeAlert = false;
                }
            }

            // Update safety warning status state flags for HUD rendering and Vocal Announcements
            isPersonInFrame = personCurrentlyInFrame;
            alertNoHelmet = overallHelmetAlert;
            alertNoGoggles = overallNoGogglesAlert;
            alertNoMask = overallNoMaskAlert;
            alertNoVest = overallNoVestAlert;
            alertMissingPpe = overallMissingPpeAlert;
            alertChinstrapOpen = overallChinstrapAlert;

            // 3. Voice Guidance Announcement Cascade Logic
            long currentTime = System.currentTimeMillis();
            if (isScanningActive && (currentTime - scanStartTime >= 15000)) {
                // Scan just finished! Mark as announced and speak results
                isScanningActive = false;
                scanResultsAnnounced = true;
                
                java.util.List<String> violations = new java.util.ArrayList<>();
                if (!scanDetectedHelmet) violations.add("wear safety helmet");
                if (!scanDetectedVest) violations.add("wear safety vest");
                if (!scanDetectedGloves) violations.add("wear protective gloves");
                if (!scanDetectedMask) violations.add("wear face mask");
                if (!scanDetectedGoggles) violations.add("wear safety goggles");

                String speechText = "";
                if (violations.isEmpty()) {
                    speechText = "Safety compliance passed. Access granted.";
                } else {
                    StringBuilder sb = new StringBuilder("Warning. Please ");
                    for (int i = 0; i < violations.size(); i++) {
                        sb.append(violations.get(i));
                        if (i < violations.size() - 2) {
                            sb.append(", ");
                        } else if (i == violations.size() - 2) {
                            sb.append(" and ");
                        }
                    }
                    sb.append(".");
                    speechText = sb.toString();
                }

                alertNoHelmet = !scanDetectedHelmet;
                alertNoVest = !scanDetectedVest;
                alertMissingPpe = !scanDetectedGloves;
                alertNoMask = !scanDetectedMask;
                alertNoGoggles = !scanDetectedGoggles;
                savePpeScanReport(violations, speechText);
                setPpeInferenceStatus(violations.isEmpty() ? "PPE passed - report saved" : "PPE missing - report saved");

                if (!speechText.isEmpty() && isTtsReady && ttsEngine != null) {
                    if (scanAnnouncementCount < 2) {
                        ttsEngine.speak(speechText, android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "PpeSpeechAlert");
                        scanAnnouncementCount++;
                    }
                    lastTtsTime = currentTime;
                }
            } else if (personCurrentlyInFrame && scanResultsAnnounced && (currentTime - lastTtsTime > 8000)) {
                // If they are still in the frame, repeat the warning reminder every 8 seconds if there are violations
                java.util.List<String> violations = new java.util.ArrayList<>();
                if (!scanDetectedHelmet) violations.add("wear safety helmet");
                if (!scanDetectedVest) violations.add("wear safety vest");
                if (!scanDetectedGloves) violations.add("wear protective gloves");
                if (!scanDetectedMask) violations.add("wear face mask");
                if (!scanDetectedGoggles) violations.add("wear safety goggles");

                if (!violations.isEmpty()) {
                    StringBuilder sb = new StringBuilder("Reminder. Please ");
                    for (int i = 0; i < violations.size(); i++) {
                        sb.append(violations.get(i));
                        if (i < violations.size() - 2) {
                            sb.append(", ");
                        } else if (i == violations.size() - 2) {
                            sb.append(" and ");
                        }
                    }
                    sb.append(".");
                    String speechText = sb.toString();
                    if (isTtsReady && ttsEngine != null) {
                        if (scanAnnouncementCount < 2) {
                            ttsEngine.speak(speechText, android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "PpeSpeechAlert");
                            scanAnnouncementCount++;
                        }
                        lastTtsTime = currentTime;
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private float getConfidenceThreshold(int classId) {
        switch (classId) {
            case 0: return 0.25f; // Boots
            case 1: return 0.35f; // Gloves
            case 2: return 0.20f; // Helmet (Easier detection for Helmet!)
            case 3: return 0.25f; // Goggles
            case 4: return 0.25f; // Mask
            case 5: return 0.45f; // Vest (Higher threshold for Vest to filter low-score 27% noise on helmets)
            default: return 0.25f;
        }
    }

    private static class Stage1Detection {
        float left, top, right, bottom;
        int classId; // 0 = Worker, 1 = Head
        float confidence;
        Stage1Detection(float left, float top, float right, float bottom, int classId, float confidence) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.classId = classId;
            this.confidence = confidence;
        }
    }

    private java.util.List<Stage1Detection> runStage1Nms(java.util.List<Stage1Detection> detections) {
        java.util.List<Stage1Detection> filtered = new java.util.ArrayList<>();
        for (Stage1Detection det : detections) {
            boolean keep = true;
            for (Stage1Detection existing : filtered) {
                if (det.classId == existing.classId) {
                    float intersectionLeft = Math.max(det.left, existing.left);
                    float intersectionTop = Math.max(det.top, existing.top);
                    float intersectionRight = Math.min(det.right, existing.right);
                    float intersectionBottom = Math.min(det.bottom, existing.bottom);
                    if (intersectionRight > intersectionLeft && intersectionBottom > intersectionTop) {
                        float intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
                        float detArea = (det.right - det.left) * (det.bottom - det.top);
                        float existingArea = (existing.right - existing.left) * (existing.bottom - existing.top);
                        float unionArea = detArea + existingArea - intersectionArea;
                        float iou = unionArea > 0 ? intersectionArea / unionArea : 0f;
                        if (iou > 0.45f) {
                            if (det.confidence <= existing.confidence) {
                                keep = false;
                                break;
                            }
                        }
                    }
                }
            }
            if (keep) {
                filtered.add(det);
            }
        }
        return filtered;
    }

    private java.util.List<Detection> runLocalNms(java.util.List<Detection> detections) {
        if (detections == null || detections.isEmpty()) return new java.util.ArrayList<>();
        
        java.util.List<Detection> sorted = new java.util.ArrayList<>(detections);
        java.util.Collections.sort(sorted, new java.util.Comparator<Detection>() {
            @Override
            public int compare(Detection a, Detection b) {
                return Float.compare(b.confidence, a.confidence);
            }
        });

        java.util.List<Detection> filtered = new java.util.ArrayList<>();
        for (Detection det : sorted) {
            boolean keep = true;
            Detection toRemove = null;
            for (Detection existing : filtered) {
                float intersectionLeft = Math.max(det.left, existing.left);
                float intersectionTop = Math.max(det.top, existing.top);
                float intersectionRight = Math.min(det.right, existing.right);
                float intersectionBottom = Math.min(det.bottom, existing.bottom);
                if (intersectionRight > intersectionLeft && intersectionBottom > intersectionTop) {
                    float intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
                    float detArea = (det.right - det.left) * (det.bottom - det.top);
                    float existingArea = (existing.right - existing.left) * (existing.bottom - existing.top);
                    float unionArea = detArea + existingArea - intersectionArea;
                    float iou = unionArea > 0 ? intersectionArea / unionArea : 0f;

                    // 1. Same-class overlap suppression
                    if (det.classId == existing.classId && iou > 0.35f) {
                        keep = false;
                        break;
                    }

                    // 2. Cross-class collision resolution: Helmet (class 2) vs Vest (class 5) on head/neck area
                    if (iou > 0.30f) {
                        if (existing.classId == 2 && det.classId == 5) {
                            // Helmet already exists in this region -> reject colliding Vest box!
                            keep = false;
                            break;
                        } else if (det.classId == 2 && existing.classId == 5) {
                            // Incoming detection is Helmet -> prioritize Helmet over colliding Vest box!
                            toRemove = existing;
                        }
                    }
                }
            }
            if (toRemove != null) {
                filtered.remove(toRemove);
            }
            if (keep) {
                filtered.add(det);
            }
        }
        return filtered;
    }

    private static class Detection {
        float left, top, right, bottom;
        int classId;
        float confidence;
        String customLabel;
        Detection(float left, float top, float right, float bottom, int classId, float confidence) {
            this(left, top, right, bottom, classId, confidence, null);
        }
        Detection(float left, float top, float right, float bottom, int classId, float confidence, String customLabel) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.classId = classId;
            this.confidence = confidence;
            this.customLabel = customLabel;
        }
    }

    private static class CachedDetection {
        Detection detection;
        long timestamp;
        CachedDetection(Detection detection, long timestamp) {
            this.detection = detection;
            this.timestamp = timestamp;
        }
    }

}
