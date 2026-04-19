# ============================================================
#  main.py  —  Blind Assistant (Debug / Simplified Version)
#
#  Runs ALL models simultaneously in parallel threads:
#    Thread 1: YOLO general object detection (every frame)
#    Thread 2: Currency detection (every frame)
#    Thread 3: Face recognition (when person detected)
#
#  NO wake words, NO mode switching.
#  All detections shown on screen + spoken via TTS.
#  Use this to confirm each model works independently.
# ============================================================

import cv2
import os
import time
import queue
import pickle
import threading
import numpy as np
from ultralytics import YOLO

# ── Optional: face_recognition (needs dlib) ─────────────────
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[INFO] face_recognition not installed — using Haar cascade only.")

# ── Optional: GPIO for sensors (Pi only) ────────────────────
try:
    from gpiozero import DistanceSensor, InputDevice
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    print("[INFO] GPIO not available (PC mode — sensors offline).")

# ================================================================
#  CONFIG
# ================================================================
CONFIG = {
    "CAMERA_INDEX":    0,
    "FRAME_WIDTH":     640,
    "FRAME_HEIGHT":    480,

    "YOLO_GENERAL_MODEL":      "yolo26n.pt",
    "YOLO_CURRENCY_MODEL_PT":  "currency_yolo26n.pt",
    "YOLO_CURRENCY_MODEL_NCNN":"currency_yolo26n_ncnn_model",

    "YOLO_GENERAL_CONF":  0.55,   # General objects confidence
    "CURRENCY_CONF":      0.30,   # Currency — lower is more sensitive
    "YOLO_IMGSZ":         640,

    "FACE_ENCODINGS_FILE": "face_encodings.pkl",
    "FACE_TOLERANCE":      0.50,

    "SPEECH_RATE":    145,
    "TTS_COOLDOWN_S": 4.0,   # 4s silence between same announcement
    "FACE_COOLDOWN_S":10.0,  # 10s silence between face announcements

    "FACE_CHECK_EVERY": 8,   # Run face recognition every N frames
}

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "face_encodings.pkl"

# ── Currency label lookup (supports both 'Rs.xxx' and plain 'xxx' format) ──
CURRENCY_LABELS = {
    "Rs.10_rupee":  "10 Rupees",  "10_rupee":  "10 Rupees",
    "Rs.20_rupee":  "20 Rupees",  "20_rupee":  "20 Rupees",
    "Rs.50_rupee":  "50 Rupees",  "50_rupee":  "50 Rupees",
    "Rs.100_rupee": "100 Rupees", "100_rupee": "100 Rupees",
    "Rs.200_rupee": "200 Rupees", "200_rupee": "200 Rupees",
    "Rs.500_rupee": "500 Rupees", "500_rupee": "500 Rupees",
    "Rs.2000_rupee":"2000 Rupees","2000_rupee":"2000 Rupees",
}

RELEVANT_OBJECTS = {
    "person", "car", "truck", "bus", "bicycle", "motorcycle",
    "chair", "bottle", "cup", "cell phone", "laptop", "tv",
    "book", "bag", "umbrella", "cat", "dog", "door",
}

# ================================================================
#  TTS ENGINE  (Windows = SAPI5 direct, Pi = pyttsx3/espeak)
# ================================================================
class TTSSpeaker:
    def __init__(self, rate=145, cooldown=4.0):
        self._queue    = queue.Queue(maxsize=6)
        self._cooldown = cooldown
        self._last     = {}
        self._rate     = rate
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str, cooldown_override: float = None):
        now = time.time()
        cd  = cooldown_override if cooldown_override is not None else self._cooldown
        # Use first word as dedup key so "car in 1.2m" and "car in 1.4m" don't clash
        key = text.split()[0] if text else text
        if now - self._last.get(key, 0) < cd:
            return
        self._last[key] = now
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        import platform
        if platform.system() == "Windows":
            import pythoncom, win32com.client
            pythoncom.CoInitialize()
            sp = win32com.client.Dispatch("SAPI.SpVoice")
            sp.Rate = -1
            while True:
                text = self._queue.get()
                try:
                    sp.Speak(text)
                except Exception as e:
                    print(f"[TTS Error] {e}")
        else:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", self._rate)
            while True:
                text = self._queue.get()
                try:
                    eng.say(text)
                    eng.runAndWait()
                except Exception as e:
                    print(f"[TTS Error] {e}")


# ================================================================
#  AUTO FACE REGISTRATION
# ================================================================
def _get_latest_mtime(folder):
    latest = 0.0
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in {'.jpg','.jpeg','.png','.bmp'}:
                t = os.path.getmtime(os.path.join(root, f))
                if t > latest: latest = t
    return latest

def auto_register_faces():
    print("[FaceReg] Checking for new faces...")
    if not FACE_RECOGNITION_AVAILABLE:
        print("[FaceReg] face_recognition not installed — skipping.")
        return
    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"[FaceReg] Created '{KNOWN_FACES_DIR}/' — add sub-folders with photos.")
        return

    photo_mtime = _get_latest_mtime(KNOWN_FACES_DIR)
    if photo_mtime == 0.0:
        print("[FaceReg] No photos found.")
        return
    pkl_mtime = os.path.getmtime(ENCODINGS_FILE) if os.path.exists(ENCODINGS_FILE) else 0.0
    if pkl_mtime >= photo_mtime:
        print("[FaceReg] Encodings up to date — skipping.")
        return

    print("[FaceReg] New photos detected — encoding...")
    encs, names = [], []
    for person in sorted(d for d in os.listdir(KNOWN_FACES_DIR)
                         if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))):
        for photo in os.listdir(os.path.join(KNOWN_FACES_DIR, person)):
            if os.path.splitext(photo)[1].lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                continue
            try:
                img  = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, person, photo))
                enc  = face_recognition.face_encodings(img)
                if enc:
                    encs.append(enc[0]); names.append(person)
            except Exception as e:
                print(f"[FaceReg]  ⚠ {photo}: {e}")

    if encs:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({"encodings": encs, "names": names}, f)
        print(f"[FaceReg] ✅ {len(encs)} face(s) saved.")


# ================================================================
#  CAMERA STREAM (threaded — always-fresh frame)
# ================================================================
class CameraStream:
    def __init__(self, index=0, width=640, height=480):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame = None
        self._lock  = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()
        print(f"Camera {index} started at {width}×{height}")

    def _reader(self):
        while True:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()
    def stop(self):
        self._cap.release()


# ================================================================
#  FACE ENGINE
# ================================================================
class FaceEngine:
    def __init__(self, enc_file="face_encodings.pkl", tolerance=0.50):
        self.known_enc   = []
        self.known_names = []
        self.tolerance   = tolerance
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if os.path.exists(enc_file):
            try:
                data = pickle.load(open(enc_file, "rb"))
                self.known_enc   = data["encodings"]
                self.known_names = data["names"]
                print(f"Loaded {len(self.known_enc)} face encoding(s).")
            except Exception as e:
                print(f"[FaceEngine] Cannot load encodings: {e}")
        else:
            print(f"[FaceEngine] No encodings found at '{enc_file}'.")

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
                    matches  = face_recognition.compare_faces(self.known_enc, enc, self.tolerance)
                    dists    = face_recognition.face_distance(self.known_enc, enc)
                    best     = int(np.argmin(dists))
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
#  PARALLEL INFERENCE ENGINE
#  Runs general + currency YOLO in worker threads so the
#  display loop never blocks waiting for model results.
# ================================================================
class InferenceEngine:
    """
    Runs general and currency YOLO in two separate background threads.
    Each model has its own dedicated frame queue — no tag routing needed.
    The display loop calls submit() each frame and get_results() to read output.
    """
    def __init__(self, general_model, currency_model):
        self.gm = general_model
        self.cm = currency_model

        self._gen_results  = []
        self._curr_results = []
        self._lock         = threading.Lock()
        self._stop_evt     = threading.Event()

        # Separate queue per model — maxsize=1 means always latest frame only
        self._gen_q  = queue.Queue(maxsize=1)
        self._cur_q  = queue.Queue(maxsize=1)

        threading.Thread(target=self._general_worker,  daemon=True, name="Gen-YOLO").start()
        if self.cm:
            threading.Thread(target=self._currency_worker, daemon=True, name="Cur-YOLO").start()

    def submit(self, frame: np.ndarray):
        """Push the latest frame to both model queues (drop old frame if busy)."""
        # Drop old frame and replace with the new one
        try:
            self._gen_q.get_nowait()
        except queue.Empty:
            pass
        self._gen_q.put_nowait(frame)

        if self.cm:
            try:
                self._cur_q.get_nowait()
            except queue.Empty:
                pass
            self._cur_q.put_nowait(frame)

    def get_results(self):
        with self._lock:
            return list(self._gen_results), list(self._curr_results)

    def _general_worker(self):
        print("[Gen-YOLO] Worker started.")
        while not self._stop_evt.is_set():
            try:
                frame = self._gen_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                results = self.gm(frame,
                                  conf=CONFIG["YOLO_GENERAL_CONF"],
                                  imgsz=CONFIG["YOLO_IMGSZ"],
                                  verbose=False)
                with self._lock:
                    self._gen_results = results
            except Exception as e:
                print(f"[Gen YOLO Error] {e}")

    def _currency_worker(self):
        print("[Cur-YOLO] Worker started.")
        while not self._stop_evt.is_set():
            try:
                frame = self._cur_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                results = self.cm(frame,
                                  conf=CONFIG["CURRENCY_CONF"],
                                  imgsz=CONFIG["YOLO_IMGSZ"],
                                  verbose=False)
                with self._lock:
                    self._curr_results = results
            except Exception as e:
                print(f"[Currency YOLO Error] {e}")

    def stop(self):
        self._stop_evt.set()



# ================================================================
#  HELPERS
# ================================================================
def draw_label(img, text, x, y, color=(0,255,0)):
    tw = len(text) * 9
    cv2.rectangle(img, (x, y-22), (x+tw, y), color, -1)
    cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

def currency_spoken(raw_lbl: str) -> str:
    if raw_lbl in CURRENCY_LABELS:
        return CURRENCY_LABELS[raw_lbl]
    cleaned = raw_lbl.replace("Rs.","").replace("_rupee","").strip("_")
    return f"{cleaned} Rupees" if cleaned else raw_lbl


# ================================================================
#  MAIN
# ================================================================
def main():
    print("\n" + "="*55)
    print("  BLIND ASSISTANT  (Debug / All-Models Version)")
    print("="*55 + "\n")

    tts = TTSSpeaker(rate=CONFIG["SPEECH_RATE"], cooldown=CONFIG["TTS_COOLDOWN_S"])
    tts.say("Blind assistant starting. Please wait.")

    # ── General model ────────────────────────────────────────
    print("Loading general YOLO model...")
    general_model = YOLO(CONFIG["YOLO_GENERAL_MODEL"])
    print("  OK")

    # ── Currency model ──────────────────────────────────────
    # Try NCNN first (faster, works on both Pi and Windows when ncnn is installed).
    # Fall back to .pt if NCNN folder is missing.
    ncnn_path    = CONFIG["YOLO_CURRENCY_MODEL_NCNN"]
    pt_path      = CONFIG["YOLO_CURRENCY_MODEL_PT"]
    currency_model = None

    if os.path.isdir(ncnn_path):
        print(f"Loading currency NCNN model: {ncnn_path}")
        currency_model = YOLO(ncnn_path)
    elif os.path.exists(pt_path):
        print(f"Loading currency PyTorch model: {pt_path}")
        currency_model = YOLO(pt_path)
    else:
        print("⚠  Currency model not found — currency detection disabled.")

    # ── Face setup ──────────────────────────────────────────
    auto_register_faces()
    face_engine = FaceEngine(CONFIG["FACE_ENCODINGS_FILE"], CONFIG["FACE_TOLERANCE"])

    # ── Camera ───────────────────────────────────────────────
    cam = CameraStream(CONFIG["CAMERA_INDEX"],
                       CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"])
    time.sleep(1.5)

    # ── Parallel inference engine ────────────────────────────
    engine = InferenceEngine(general_model, currency_model)

    tts.say("System ready.")
    print("System ready. Press Q to quit.\n")

    # ── Display setup ────────────────────────────────────────
    WIN      = "Blind Assistant — All Models"
    SCALE    = 2
    PANEL_W  = 380
    DISP_H   = CONFIG["FRAME_HEIGHT"] * SCALE   # 960
    DISP_W   = CONFIG["FRAME_WIDTH"]  * SCALE   # 1280
    TOTAL_W  = DISP_W + PANEL_W

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, TOTAL_W, DISP_H)

    frame_count = 0
    fps_timer   = time.time()
    display_fps = 0.0
    last_faces  = []
    FACE_EVERY  = CONFIG["FACE_CHECK_EVERY"]

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # Push frame to inference workers
        engine.submit(frame)

        # FPS calc
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            display_fps = 30 / elapsed if elapsed > 0 else 0
            fps_timer   = time.time()

        # Get latest results (non-blocking)
        gen_results, curr_results = engine.get_results()

        # ── Scale frame up for display ────────────────────────
        big = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)

        # ── Draw general object boxes ─────────────────────────
        person_detected = False
        for r in gen_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = general_model.names[cls_id]
                conf   = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                if label not in RELEVANT_OBJECTS:
                    continue

                # Scale to big frame
                bx1,by1 = x1*SCALE, y1*SCALE
                bx2,by2 = x2*SCALE, y2*SCALE
                cv2.rectangle(big, (bx1,by1), (bx2,by2), (0,220,0), 2)
                cv2.putText(big, f"{label} {conf:.0%}",
                            (bx1, max(by1-8,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,255,0), 2)

                if label == "person":
                    person_detected = True
                else:
                    tts.say(f"It is {label}", cooldown_override=CONFIG["TTS_COOLDOWN_S"])

        # ── Draw currency boxes ───────────────────────────────
        for r in curr_results:
            for box in r.boxes:
                raw   = currency_model.names[int(box.cls[0])]
                spoken= currency_spoken(raw)
                conf  = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                bx1,by1 = x1*SCALE, y1*SCALE
                bx2,by2 = x2*SCALE, y2*SCALE
                cv2.rectangle(big, (bx1,by1), (bx2,by2), (0,215,255), 3)
                cv2.putText(big, spoken,
                            (bx1, max(by1-10,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,215,255), 2)
                tts.say(f"It is {spoken}", cooldown_override=CONFIG["TTS_COOLDOWN_S"])
                print(f"[CURRENCY] {spoken}  ({conf:.0%})")

        # ── Face recognition (runs every N frames if person seen) ─
        if frame_count % FACE_EVERY == 0 and person_detected:
            last_faces = face_engine.process(frame)
        elif not person_detected:
            last_faces = []

        for (name, x, y_f, w_f, h_f) in last_faces:
            color = (255,80,0) if name != "Unknown" else (80,80,255)
            bx1,by1 = x*SCALE, y_f*SCALE
            bx2,by2 = (x+w_f)*SCALE, (y_f+h_f)*SCALE
            cv2.rectangle(big, (bx1,by1), (bx2,by2), color, 2)
            cv2.putText(big, name, (bx1, max(by1-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            if name == "Unknown":
                tts.say("It is an unknown person.", cooldown_override=CONFIG["FACE_COOLDOWN_S"])
            else:
                tts.say(f"It is {name}.", cooldown_override=CONFIG["FACE_COOLDOWN_S"])

        # ── Side panel ────────────────────────────────────────
        panel = np.zeros((DISP_H, PANEL_W, 3), dtype=np.uint8)
        panel[:] = (28, 28, 28)

        # Header
        cv2.rectangle(panel, (0,0), (PANEL_W, 50), (45,45,45), -1)
        cv2.putText(panel, "BLIND ASSISTANT", (10,22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (100,255,100), 1)
        cv2.putText(panel, f"FPS: {display_fps:.1f}   [All Models ON]", (10,43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,160,160), 1)
        cv2.line(panel, (0,50), (PANEL_W,50), (70,70,70), 1)

        y = 65
        def sec(title, color=(120,220,255)):
            nonlocal y
            cv2.rectangle(panel,(0,y),(PANEL_W,y+26),(50,50,50),-1)
            cv2.putText(panel,f"  {title}",(8,y+18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,color,1)
            y+=30
        def row(lbl, val, vc=(255,255,255)):
            nonlocal y
            cv2.putText(panel,lbl,(12,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.50,(160,160,160),1)
            cv2.putText(panel,str(val),(160,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.55,vc,1)
            y+=26

        # Objects
        sec("OBJECTS", (80,255,160))
        obj_rows = 0
        for r in gen_results:
            for box in r.boxes:
                lbl  = general_model.names[int(box.cls[0])]
                if lbl not in RELEVANT_OBJECTS: continue
                conf = float(box.conf[0])
                row(lbl.capitalize(), f"{conf:.0%}")
                obj_rows += 1
        if obj_rows == 0:
            row("None","", (80,80,80))
        y += 4

        # Currency
        sec("CURRENCY", (0,200,255))
        curr_rows = 0
        for r in curr_results:
            for box in r.boxes:
                raw  = currency_model.names[int(box.cls[0])] if currency_model else "?"
                spk  = currency_spoken(raw)
                conf = float(box.conf[0])
                row("Note", f"{spk}  {conf:.0%}", (0,220,255))
                curr_rows += 1
        if curr_rows == 0:
            row("None","", (80,80,80))
        y += 4

        # Faces
        sec("FACES", (255,120,60))
        face_rows = 0
        for (name, *_) in last_faces:
            fc = (255,180,80) if name != "Unknown" else (80,80,255)
            row("Person", name, fc)
            face_rows += 1
        if face_rows == 0:
            row("None","", (80,80,80))

        # Status bar
        sy = DISP_H - 55
        cv2.line(panel, (0,sy), (PANEL_W,sy), (60,60,60), 1)
        status = "Sensors: Offline (PC)" if not GPIO_AVAILABLE else "Sensors: Online"
        cv2.putText(panel, status, (10, sy+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120,120,120), 1)
        cv2.putText(panel, "Press Q to quit", (10, sy+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100,100,100), 1)

        # Stitch together
        canvas = np.hstack([big, panel])
        cv2.imshow(WIN, canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    engine.stop()
    cam.stop()
    cv2.destroyAllWindows()
    print("Assistant stopped.")


if __name__ == "__main__":
    main()