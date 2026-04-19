# ============================================================
#  main2.py  —  Blind Assistant for Raspberry Pi 4
#
#  All models run always (no mode switching).
#  Modelled on the working previous_main2.py architecture.
#
#  Voice Commands (say clearly):
#    "currency" / "money"   → switch to currency-only scan
#    "navigate" / "person"  → switch to object + face scan
#
#  TTS Rules:
#    Objects  → 4 s cooldown
#    Faces    → 10 s cooldown
#    Currency → 4 s cooldown
# ============================================================

import cv2
import os
import time
import queue
import pickle
import threading
import platform
import numpy as np

from ultralytics import YOLO

# ── Optional: face_recognition (needs dlib) ─────────────────
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[INFO] face_recognition not installed — Haar cascade only.")

# ── Optional: GPIO (Pi only) ─────────────────────────────────
try:
    from gpiozero import DistanceSensor, InputDevice
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    print("[INFO] GPIO not available (PC mode).")

# ── Optional: win32com for SAPI5 TTS (Windows) ───────────────
try:
    import win32com.client
    import pythoncom
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

# ================================================================
#  CONSTANTS
# ================================================================
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "face_encodings.pkl"
ON_PI           = platform.system() == "Linux"

CURRENCY_LABELS = {
    # Model uses 'Rs.' prefix
    "Rs.10_rupee":  "10 Rupees",   "Rs.20_rupee":  "20 Rupees",
    "Rs.50_rupee":  "50 Rupees",   "Rs.100_rupee": "100 Rupees",
    "Rs.200_rupee": "200 Rupees",  "Rs.500_rupee": "500 Rupees",
    "Rs.2000_rupee":"2000 Rupees",
    # Fallback without prefix
    "10_rupee":  "10 Rupees",  "20_rupee":  "20 Rupees",
    "50_rupee":  "50 Rupees",  "100_rupee": "100 Rupees",
    "200_rupee": "200 Rupees", "500_rupee": "500 Rupees",
    "2000_rupee":"2000 Rupees",
}

RELEVANT_OBJECTS = {
    "person", "car", "truck", "bus", "bicycle", "motorcycle",
    "chair", "bottle", "cup", "cell phone", "laptop", "tv",
    "book", "bag", "umbrella", "cat", "dog", "door",
}

# ================================================================
#  CONFIG
# ================================================================
CONFIG = {
    "CAMERA_INDEX":  0,
    "FRAME_WIDTH":   640,
    "FRAME_HEIGHT":  480,

    "YOLO_GENERAL_MODEL":       "yolo26n.pt",
    "YOLO_CURRENCY_MODEL_PT":   "new_curr_best.pt",
    "YOLO_CURRENCY_MODEL_NCNN": "new_best_ncnn_model",

    "YOLO_CONF":         0.50,
    "CURRENCY_CONF":     0.50,   # lower = more sensitive
    "YOLO_IMGSZ":        640,

    "FACE_ENCODINGS_FILE": "face_encodings.pkl",
    "FACE_TOLERANCE":      0.50,

    "ULTRASONIC_TRIGGER_PIN": 4,
    "ULTRASONIC_ECHO_PIN":    17,
    "IR_SENSOR_PIN":          24,
    "OBSTACLE_WARN_CM":       60,

    "SPEECH_RATE":       145,
    "TTS_COOLDOWN_S":    8.0,
    "FACE_COOLDOWN_S":  10.0,

    "CURRENCY_CHECK_EVERY": 1,   # every frame in currency mode
    "FACE_CHECK_EVERY":     8,
}


# ================================================================
#  AUTO FACE REGISTRATION
# ================================================================
def _get_latest_mtime(folder):
    latest = 0.0
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in {'.jpg','.jpeg','.png','.bmp'}:
                t = os.path.getmtime(os.path.join(root, f))
                if t > latest:
                    latest = t
    return latest

def auto_register_faces():
    print("[FaceReg] Checking for new faces...")
    if not FACE_RECOGNITION_AVAILABLE:
        print("[FaceReg] face_recognition not installed — skipping.")
        return
    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        return

    photo_mtime = _get_latest_mtime(KNOWN_FACES_DIR)
    if photo_mtime == 0.0:
        print("[FaceReg] No photos in known_faces/.")
        return
    pkl_mtime = os.path.getmtime(ENCODINGS_FILE) if os.path.exists(ENCODINGS_FILE) else 0.0
    if pkl_mtime >= photo_mtime:
        print("[FaceReg] Encodings up to date — skipping.")
        return

    print("[FaceReg] Encoding new photos...")
    encs, names = [], []
    for person in sorted(d for d in os.listdir(KNOWN_FACES_DIR)
                         if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))):
        for photo in os.listdir(os.path.join(KNOWN_FACES_DIR, person)):
            if os.path.splitext(photo)[1].lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                continue
            try:
                img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, person, photo))
                enc = face_recognition.face_encodings(img)
                if enc:
                    encs.append(enc[0]); names.append(person)
            except Exception as e:
                print(f"[FaceReg] Error {photo}: {e}")
    if encs:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({"encodings": encs, "names": names}, f)
        print(f"[FaceReg] {len(encs)} face(s) saved.")


# ================================================================
#  TTS ENGINE — Windows (SAPI5) or Pi (pyttsx3/espeak)
#  Per-category cooldown prevents speech overlap.
# ================================================================
class TTSSpeaker:
    def __init__(self, rate=145):
        # maxsize=1 ensures we NEVER backlog old speech. It stays strictly real-time.
        self._queue = queue.Queue(maxsize=1)
        self._last  = {}   # category → last_spoken_time
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str, category: str = None, cooldown: float = None):
        """
        category:  dedup key (e.g. object name, 'face', currency label)
        cooldown:  seconds between repeat of this category
        """
        key = category or text
        cd  = cooldown if cooldown is not None else CONFIG["TTS_COOLDOWN_S"]
        now = time.time()
        if now - self._last.get(key, 0) < cd:
            return
        self._last[key] = now
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        if WIN32_AVAILABLE:
            pythoncom.CoInitialize()
            sp = win32com.client.Dispatch("SAPI.SpVoice")
            sp.Rate = -1
            while True:
                text = self._queue.get()
                try:
                    sp.Speak(text)
                except Exception as e:
                    print(f"[TTS] {e}")
        else:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", CONFIG["SPEECH_RATE"])
            while True:
                text = self._queue.get()
                try:
                    eng.say(text); eng.runAndWait()
                except Exception as e:
                    print(f"[TTS] {e}")


# ================================================================
#  CAMERA STREAM (threaded — always fresh, never stale)
# ================================================================
class CameraStream:
    def __init__(self, index=0, width=640, height=480):
        self._source = index
        if ON_PI and isinstance(index, int):
            self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture("hel.mp4")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()
        print(f"Camera {index} started at {width}×{height}")

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self._source, str):  # Loop the video if it's a file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._cap.release()


# ================================================================
#  SENSOR MONITOR (Pi only — gracefully offline on PC)
# ================================================================
class SensorMonitor:
    def __init__(self, tts, cfg):
        self.dist_cm      = 999.0
        self.ir_triggered = False
        self._running     = True
        self._ultrasonic  = None
        self._ir          = None

        if GPIO_AVAILABLE:
            try:
                self._ultrasonic = DistanceSensor(
                    echo=cfg["ULTRASONIC_ECHO_PIN"],
                    trigger=cfg["ULTRASONIC_TRIGGER_PIN"],
                    max_distance=4.0)
                self._ir = InputDevice(cfg["IR_SENSOR_PIN"])
            except Exception as e:
                print(f"[Sensor] Init error: {e}")

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            if self._ultrasonic:
                try:
                    self.dist_cm = self._ultrasonic.distance * 100.0
                except Exception:
                    self.dist_cm = 999.0
            if self._ir:
                try:
                    self.ir_triggered = self._ir.is_active
                except Exception:
                    self.ir_triggered = False
            time.sleep(0.2)

    def stop(self):
        self._running = False


# ================================================================
#  FACE ENGINE
# ================================================================
class FaceEngine:
    def __init__(self, enc_file=ENCODINGS_FILE, tolerance=0.50):
        self.known_enc   = []
        self.known_names = []
        self.tolerance   = tolerance
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if FACE_RECOGNITION_AVAILABLE and os.path.exists(enc_file):
            try:
                data = pickle.load(open(enc_file, "rb"))
                self.known_enc   = data["encodings"]
                self.known_names = data["names"]
                print(f"Loaded {len(self.known_enc)} face encoding(s).")
            except Exception as e:
                print(f"[FaceEngine] Cannot load: {e}")

    def process(self, frame):
        """Return list of (name, x, y, w, h)."""
        results = []
        if FACE_RECOGNITION_AVAILABLE and self.known_enc:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb, model="hog")
            encs  = face_recognition.face_encodings(rgb, locs)
            for enc, (top, right, bottom, left) in zip(encs, locs):
                name = "Unknown"
                if self.known_enc:
                    matches = face_recognition.compare_faces(self.known_enc, enc, self.tolerance)
                    dists   = face_recognition.face_distance(self.known_enc, enc)
                    best    = int(np.argmin(dists))
                    if matches[best]:
                        name = self.known_names[best]
                top*=2; right*=2; bottom*=2; left*=2
                results.append((name, left, top, right-left, bottom-top))
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                results.append(("Person", x, y, w, h))
        return results


# ================================================================
#  VOICE COMMAND LISTENER (background thread)
#  Switches active_mode between "navigate" and "currency"
# ================================================================
class VoiceCommandListener:
    """
    Background microphone thread to toggle between detection modes.
    Say "currency" or "money"  →  currency-only mode
    Say "navigate" or "person" →  navigation mode (objects + faces)
    """
    def __init__(self, tts, initial_mode="navigate"):
        self.tts  = tts
        self.mode = initial_mode
        self._running = True

        try:
            import speech_recognition as sr
            self._sr = sr
            threading.Thread(target=self._listen, daemon=True).start()
            print("[Voice] Wake word listener started.")
        except ImportError:
            print("[Voice] SpeechRecognition not installed — voice toggle disabled.")
            self._sr = None

    def _listen(self):
        try:
            with self._sr.Microphone() as source:
                rec = self._sr.Recognizer()
                print("[Voice] Calibrating mic...")
                rec.adjust_for_ambient_noise(source, duration=1.5)
                print("[Voice] Listening. Say 'currency' or 'navigate'.")
                while self._running:
                    try:
                        print("[Voice] Listening... ", end="", flush=True)
                        audio = rec.listen(source, timeout=4.0, phrase_time_limit=4.0)
                        text  = rec.recognize_google(audio).lower()
                        print(f"Heard: '{text}'")
                        if any(w in text for w in ("currency", "money", "rupee")):
                            if self.mode != "currency":
                                self.mode = "currency"
                                self.tts.say("Currency mode. Scanning notes.", category="_mode_")
                        elif any(w in text for w in ("navigate", "person", "normal", "object")):
                            if self.mode != "navigate":
                                self.mode = "navigate"
                                self.tts.say("Navigation mode activated.", category="_mode_")
                    except self._sr.WaitTimeoutError:
                        print("(silence)")
                    except self._sr.UnknownValueError:
                        print("(unclear)")
                    except Exception as e:
                        print(f"(error: {e})")
                        time.sleep(2)
        except Exception as e:
            print(f"[Voice] Microphone error: {e}")


# ================================================================
#  HELPERS
# ================================================================
def currency_label_to_speech(raw: str) -> str:
    if raw in CURRENCY_LABELS:
        return CURRENCY_LABELS[raw]
    # Dynamic strip: 'Rs.500_rupee' → '500 Rupees'
    cleaned = raw.replace("Rs.", "").replace("_rupee", "").replace("rupee", "").strip("_").strip()
    return f"{cleaned} Rupees" if cleaned else raw

def draw_label(img, text, x, y, color=(0, 255, 0)):
    """Filled color background behind text."""
    tw = max(len(text) * 9, 60)
    cv2.rectangle(img, (x, y - 22), (x + tw, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)


# ================================================================
#  MAIN
# ================================================================
def main():
    print("\n" + "="*55)
    print("  BLIND ASSISTANT STARTING")
    print("="*55 + "\n")

    tts = TTSSpeaker(rate=CONFIG["SPEECH_RATE"])
    tts.say("Blind assistant starting. Please wait.", category="_boot_", cooldown=0)

    # ── General YOLO ──────────────────────────────────────────
    print("Loading general YOLO model...")
    general_model = YOLO(CONFIG["YOLO_GENERAL_MODEL"])
    print("  ✓ General model ready")

    # ── Currency Model ────────────────────────────────────────
    # Prefer NCNN (fast on Pi, also works on Windows with ncnn package).
    # Fall back to .pt. If .pt also missing → disabled.
    currency_model = None
    ncnn_path = CONFIG["YOLO_CURRENCY_MODEL_NCNN"]
    pt_path   = CONFIG["YOLO_CURRENCY_MODEL_PT"]

    if os.path.isdir(ncnn_path):
        print(f"Loading currency NCNN model: {ncnn_path}")
        currency_model = YOLO(ncnn_path)
        print("  ✓ Currency NCNN ready")
    elif os.path.exists(pt_path):
        print(f"Loading currency .pt model: {pt_path}")
        currency_model = YOLO(pt_path)
        print("  ✓ Currency .pt ready")
    else:
        print("  ⚠ No currency model found — currency detection disabled.")
        print("    Place currency_yolo26n.pt or currency_yolo26n_ncnn_model/ here.")

    # Print model class names for diagnosis
    if currency_model:
        print(f"  Currency model classes: {list(currency_model.names.values())}")

    # ── Faces ────────────────────────────────────────────────
    auto_register_faces()
    face_engine = FaceEngine(CONFIG["FACE_ENCODINGS_FILE"], CONFIG["FACE_TOLERANCE"])

    # ── Camera ───────────────────────────────────────────────
    cam = CameraStream(CONFIG["CAMERA_INDEX"],
                       CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"])
    time.sleep(1.5)

    # ── Sensors ──────────────────────────────────────────────
    sensors = SensorMonitor(tts, CONFIG)

    # ── Voice Listener ────────────────────────────────────────
    vc = VoiceCommandListener(tts, initial_mode="navigate")

    tts.say("System ready.", category="_ready_", cooldown=0)
    print("System ready. Q = quit | Voice: 'currency' or 'navigate'\n")

    # ── Display setup ─────────────────────────────────────────
    WIN_NAME = "Blind Assistant - Detection View"
    # Display at 2x for clarity (but draw boxes based on raw coords * SCALE)
    SCALE    = 2
    PANEL_W  = 300
    DISP_H   = CONFIG["FRAME_HEIGHT"] * SCALE
    DISP_W   = CONFIG["FRAME_WIDTH"]  * SCALE
    TOTAL_W  = DISP_W + PANEL_W

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, TOTAL_W, DISP_H)

    frame_count   = 0
    fps_timer     = time.time()
    display_fps   = 0.0
    last_faces    = []
    last_curr_res = []

    FACE_EVERY     = CONFIG["FACE_CHECK_EVERY"]
    CURRENCY_EVERY = CONFIG["CURRENCY_CHECK_EVERY"]

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # FPS
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            display_fps = 30 / elapsed if elapsed > 0 else 0
            fps_timer   = time.time()

        active_mode = vc.mode   # "navigate" or "currency"

        # Scale up for display — ALL bounding boxes drawn on big_frame
        big_frame = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)

        person_detected = False
        gen_results     = []

        # ════════════════════════════════════════════════════════
        #  OBJECT + FACE DETECTION
        # ════════════════════════════════════════════════════════
        gen_results = general_model(
            frame,
            conf    = CONFIG["YOLO_CONF"],
            imgsz   = CONFIG["YOLO_IMGSZ"],
            verbose = False,
        )

        for r in gen_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = general_model.names[cls_id]
                conf   = float(box.conf[0])

                if label not in RELEVANT_OBJECTS:
                    continue

                # Coordinates are in original (640×480) space → scale to display
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bx1, by1 = x1*SCALE, y1*SCALE
                bx2, by2 = x2*SCALE, y2*SCALE

                cv2.rectangle(big_frame, (bx1, by1), (bx2, by2), (0, 220, 0), 2)
                draw_label(big_frame, f"{label} {conf:.0%}", bx1, by1, (0, 220, 0))

                if label == "person":
                    person_detected = True
                else:
                    # Announce with distance if sensor available
                    if GPIO_AVAILABLE and sensors._ultrasonic and sensors.dist_cm < 900:
                        dist_m = sensors.dist_cm / 100.0
                        tts.say(f"It is {label} in {dist_m:.1f} meter",
                                category=label, cooldown=CONFIG["TTS_COOLDOWN_S"])
                    else:
                        tts.say(f"It is {label}",
                                category=label, cooldown=CONFIG["TTS_COOLDOWN_S"])

        # Obstacle fallback: sensor detects close object but YOLO missed it
        if not person_detected and GPIO_AVAILABLE and sensors._ultrasonic:
            if sensors.dist_cm < 500:
                tts.say("There is an object in front of you.",
                        category="obstacle", cooldown=CONFIG["TTS_COOLDOWN_S"])

        # ── Face recognition ──────────────────────────────
        if frame_count % FACE_EVERY == 0 and person_detected:
            last_faces = face_engine.process(frame)
        elif not person_detected:
            last_faces = []

        for (name, x, y_f, w_f, h_f) in last_faces:
            color = (255, 80, 0) if name != "Unknown" else (80, 80, 255)
            bx1 = x     * SCALE;  by1 = y_f    * SCALE
            bx2 = (x+w_f)*SCALE;  by2 = (y_f+h_f)*SCALE
            cv2.rectangle(big_frame, (bx1, by1), (bx2, by2), color, 2)
            draw_label(big_frame, name, bx1, by1, color)

            if name == "Unknown":
                tts.say("It is an unknown person.",
                        category="face", cooldown=CONFIG["FACE_COOLDOWN_S"])
            else:
                tts.say(f"It is {name}.",
                        category="face", cooldown=CONFIG["FACE_COOLDOWN_S"])

        # ════════════════════════════════════════════════════════
        #  CURRENCY DETECTION
        # ════════════════════════════════════════════════════════
        if currency_model and frame_count % CURRENCY_EVERY == 0:
            last_curr_res = currency_model(
                frame,
                conf    = CONFIG["CURRENCY_CONF"],
                imgsz   = CONFIG["YOLO_IMGSZ"],
                verbose = False,
            )

            # --- Calculate Total Value ---
            total_sum = 0
            for r in last_curr_res:
                for box in r.boxes:
                    raw = currency_model.names[int(box.cls[0])]
                    # Extract the digits from the label name directly
                    num_str = ''.join(filter(str.isdigit, raw))
                    if num_str:
                        total_sum += int(num_str)
            
            if total_sum > 0:
                tts.say(f"Total amount is {total_sum} Rupees.", category="currency_total", cooldown=CONFIG["TTS_COOLDOWN_S"])
                print(f"[CURRENCY] Sum total parsed: {total_sum}")
            else:
                # If nothing is detected, wipe the ghost boxes completely!
                last_curr_res = []

        # Draw the saved boxes on the UI
        for r in last_curr_res:
            for box in r.boxes:
                raw    = currency_model.names[int(box.cls[0])]
                spoken = currency_label_to_speech(raw)
                conf   = float(box.conf[0])

                # Coordinates in original (640×480) space → scale to display
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bx1, by1 = x1*SCALE, y1*SCALE
                bx2, by2 = x2*SCALE, y2*SCALE

                cv2.rectangle(big_frame, (bx1, by1), (bx2, by2), (0, 215, 255), 3)
                draw_label(big_frame, spoken, bx1, by1, (0, 200, 255))

        if currency_model is None:
            cv2.putText(big_frame, "Currency model not loaded!", (20, DISP_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ════════════════════════════════════════════════════════
        #  PANEL — side info bar
        # ════════════════════════════════════════════════════════
        panel = np.zeros((DISP_H, PANEL_W, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        py = 0
        def ph(title, color=(100, 220, 255)):
            nonlocal py
            cv2.rectangle(panel, (0, py), (PANEL_W, py+24), (50,50,50), -1)
            cv2.putText(panel, title, (8, py+17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
            py += 24

        def pr(label, val, vc=(230,230,230)):
            nonlocal py
            cv2.putText(panel, label, (10, py+16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150,150,150), 1)
            cv2.putText(panel, str(val), (140, py+16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, vc, 1)
            py += 22

        # Header
        cv2.rectangle(panel, (0,0), (PANEL_W, 52), (40,40,40), -1)
        mode_color = (0, 215, 255) if active_mode == "currency" else (80, 255, 80)
        cv2.putText(panel, "BLIND ASSISTANT", (8, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (100,255,100), 1)
        cv2.putText(panel, f"FPS:{display_fps:.1f}  Mode:{active_mode.upper()}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1)
        cv2.line(panel, (0, 52), (PANEL_W, 52), (70,70,70), 1)
        py = 58

        # Sensors
        ph("SENSORS")
        if GPIO_AVAILABLE and sensors._ultrasonic:
            dc = (0,80,255) if sensors.dist_cm < CONFIG["OBSTACLE_WARN_CM"] else (80,255,80)
            pr("Ultrasonic", f"{sensors.dist_cm:.0f} cm", dc)
            irc = (0,80,255) if sensors.ir_triggered else (80,255,80)
            pr("IR Sensor", "TRIGGERED" if sensors.ir_triggered else "Clear", irc)
        else:
            pr("Status", "Offline (PC)", (100,100,100))
        py += 4

        # Objects
        ph("OBJECTS", (80,255,160))
        shown = 0
        for r in gen_results:
            for box in r.boxes:
                lbl = general_model.names[int(box.cls[0])]
                if lbl not in RELEVANT_OBJECTS: continue
                pr(lbl.capitalize(), f"{float(box.conf[0]):.0%}")
                shown += 1
        if shown == 0:
            pr("None", "", (70,70,70))
        py += 4

        # Currency
        ph("CURRENCY", (0,200,255))
        shown = 0
        for r in last_curr_res:
            for box in r.boxes:
                raw  = currency_model.names[int(box.cls[0])]
                spk  = currency_label_to_speech(raw)
                pr("Note", f"{spk} {float(box.conf[0]):.0%}", (0,220,255))
                shown += 1
        if shown == 0:
            pr("No notes","", (100,100,100))
        py += 4

        # Faces
        ph("FACES", (255,120,60))
        if last_faces:
            for (name, *_) in last_faces:
                fc = (255,180,80) if name != "Unknown" else (80,80,255)
                pr("Person", name, fc)
        else:
            pr("None","", (70,70,70))
        py += 4

        # Legend
        ly = DISP_H - 48
        cv2.line(panel,(0,ly),(PANEL_W,ly),(60,60,60),1)
        cv2.putText(panel,"Green=Objects", (8,ly+16),cv2.FONT_HERSHEY_SIMPLEX,0.40,(80,255,80),1)
        cv2.putText(panel,"Yellow=Currency",(8,ly+30),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,215,255),1)
        cv2.putText(panel,"Blue/Orange=Face",(8,ly+44),cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,160,80),1)

        # Stitch display
        canvas = np.hstack([big_frame, panel])

        # Mode overlay banner on video
        banner_color = (0, 200, 255) if active_mode == "currency" else (0, 200, 0)
        # cv2.putText(canvas, f"MODE: {active_mode.upper()}", (10, 36),
        #             cv2.FONT_HERSHEY_DUPLEX, 1.0, banner_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {display_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

        cv2.imshow(WIN_NAME, canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    sensors.stop()
    cv2.destroyAllWindows()
    print("Assistant stopped.")


if __name__ == "__main__":
    main()