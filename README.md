# Blind Assistant — Setup & Usage Guide

## Project Files

| File | Purpose |
|------|---------|
| `train_colab.py` | Run in Google Colab to train the currency model |
| `face_register.py` | Register known faces into `face_encodings.pkl` |
| `main2.py` | Main app to run on Raspberry Pi 4 |

---

## PHASE 1 — Train Currency Model (Google Colab)

### Step 1 — Open Colab
1. Go to https://colab.research.google.com
2. New notebook → **Runtime → Change runtime type → T4 GPU**

### Step 2 — Install
```python
!pip install ultralytics gdown tqdm
```

### Step 3 — Upload Dataset
1. Go to https://data.mendeley.com/datasets/48ympv8jjf/1
2. Click **Download All** → save the `.zip` to your PC
3. In Colab: **Files panel (left) → Upload** → select the zip
4. In a Colab cell, run:
```python
from train_colab import *
extract_dataset("your_zip_filename.zip")  # use exact filename
inspect_raw_data()  # check folder names
```

### Step 4 — Prepare + Train
```python
# Check the CLASS_MAP in train_colab.py and update folder names if needed
prepare_yolo_data()
create_yaml()
best_model = start_training()
export_for_rpi(best_model)
```

### Step 5 — Download Model
```python
from google.colab import files
files.download('runs/detect/indian_currency_yolo26/weights/best.pt')
```
Rename the file to `currency_yolo26n.pt` and copy it to `c:\Rajendran\blind\`

---

## PHASE 2 — Register Faces (on your PC)

### Step 1 — Create folders
```
c:\Rajendran\blind\known_faces\
    Rajendran\
        photo1.jpg
        photo2.jpg
    Mother\
        photo1.jpg
```

### Step 2 — Install & Run
```bash
pip install face-recognition
python face_register.py
```
This creates `face_encodings.pkl` in the same folder.

---

## PHASE 3 — Deploy to Raspberry Pi 4

### Step 1 — Copy files to the Pi
```
main2.py
currency_yolo26n.pt
face_encodings.pkl
```

### Step 2 — Install on Raspberry Pi
```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3-pip cmake libboost-python-dev libopenblas-dev espeak portaudio19-dev

# Python packages
pip install ultralytics opencv-python pyttsx3 gpiozero face-recognition


sudo apt-get install -y espeak cmake libboost-python-dev libopenblas-dev
pip install ultralytics opencv-python pyttsx3 gpiozero face-recognition
```

### Step 3 — Run
```bash
python main2.py
```
Press **Q** to quit.

---

## Hardware Pin Reference (BCM Numbering)

| Component | Pin | Notes |
|-----------|-----|-------|
| Ultrasonic Trigger | GPIO 4 | |
| Ultrasonic Echo | GPIO 17 | Use 1kΩ+2kΩ voltage divider |
| IR Sensor (Digital) | GPIO 24 | Change in CONFIG if different |
| Camera | CSI port | Or USB webcam index 0 |

---

## What Each Thread Does

```
Thread 1 — Camera Stream    → Keeps buffer fresh (no frame lag)
Thread 2 — Vision Loop      → YOLO26n + Currency model (main thread)
Thread 3 — Sensor Monitor   → Ultrasonic + IR polling at 5 Hz
Thread 4 — TTS Speaker      → Async speech queue (never blocks vision)
```

---

## Expected Performance on Raspberry Pi 4

| Component | Speed |
|-----------|-------|
| YOLO26n (320×320) | ~10–15 FPS |
| Face Recognition | Every 8 frames |
| Currency Model | Every frame |
| Sensor Polling | 5 Hz (0.2s) |
| TTS Cooldown | 2.5 sec per message |
