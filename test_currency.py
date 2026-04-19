"""
test_currency.py  --  Standalone currency model test
Run: python test_currency.py
Press Q to quit.

This shows EXACTLY what the model sees per-frame,
including raw class names, so you can verify class-label mapping.
"""

import os, time
import cv2
import numpy as np
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────
#PT_MODEL  = "currency_yolo26n.pt"
PT_MODEL = "new_curr_best.pt"
CONF      = 0.45          # deliberately LOW so we see near-matches too
IMGSZ     = 640
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480

# ── Load model ──────────────────────────────────────────────────────
if not os.path.exists(PT_MODEL):
    print(f"ERROR: {PT_MODEL} not found in current directory!")
    print("Make sure you are running from c:\\Rajendran\\blind")
    exit(1)

print(f"Loading {PT_MODEL}...")
model = YOLO(PT_MODEL)
print(f"\n=== Model class names ({len(model.names)}) ===")
for idx, name in model.names.items():
    print(f"  [{idx}] {name}")
print("=" * 40)
print(f"\nCamera opening... conf threshold = {CONF}")

# cap = cv2.VideoCapture(CAM_INDEX)
cap = cv2.VideoCapture("hel.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
time.sleep(1)

frame_n = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame_n += 1

    # Run model every frame for testing
    results = model(frame, conf=CONF, imgsz=IMGSZ, verbose=False)

    found_any = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"[Frame {frame_n:04d}]  Detected: '{label}'  conf={conf:.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(frame, f"{label} {conf:.0%}",
                        (x1, max(y1-8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
            found_any = True

    if not found_any and frame_n % 30 == 0:
        print(f"[Frame {frame_n:04d}]  (nothing detected)")

    cv2.putText(frame, f"MODE: CURRENCY TEST  conf>={CONF:.0%}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Currency Test  (Q = quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nTest complete.")
