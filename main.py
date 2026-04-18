import cv2
import threading
import pyttsx3
from gpiozero import DistanceSensor
from ultralytics import YOLO
# Import the currency model functions from the cloned repo
# from currency_model import CurrencyClassifier 

# 1. Initialize Text-to-Speech
engine = pyttsx3.init()
def speak(text):
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

# 2. Initialize Sensors (Ultrasonic: Trigger=4, Echo=17)
# ultrasonic = DistanceSensor(echo=17, trigger=4)

# 3. Load YOLOv26 and Face Detection
# YOLO26n (Nano) is highly recommended for RPi 4 for real-time speed.
model = YOLO('yolo26n.pt') 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def sensor_monitor():
#     """Background thread to monitor distance"""
#     while True:
#         # dist = ultrasonic.distance * 100 # Convert to cm
#         if dist < 50: # If obstacle is closer than 50cm
#             speak("Warning: Obstacle very close")

def main_vision_loop():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run YOLOv26 Detection
        results = model(frame, conf=0.5)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                # Logic for Face Detection within frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    speak("A person is in front of you")

                # Logic for Money Identification
                if label == "money" or label == "paper":
                    # Call the predict function from your GitHub repo
                    # result = currency_model.predict(frame)
                    # speak(f"Detected {result} Rupees")
                    pass
                
                # Logic for distance + object (combining YOLO + Sensor)
                # dist = round(ultrasonic.distance, 1)
                # speak(f"{label} is at {dist} meters")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start Threads
# t1 = threading.Thread(target=sensor_monitor)
# t1.daemon = True
# t1.start()

main_vision_loop()