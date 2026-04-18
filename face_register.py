# ============================================================
#  face_register.py  —  Register Faces for the Blind Assistant
#
#  Run this ONCE on a PC or Raspberry Pi with photos of people
#  you want the system to recognize.
#
#  HOW TO USE:
#  1. Create a folder called 'known_faces/'
#  2. Inside it, create one sub-folder per person, named after them:
#       known_faces/
#           Rajendran/
#               photo1.jpg
#               photo2.jpg
#           Mother/
#               photo1.jpg
#  3. Run: python face_register.py
#  4. This produces 'face_encodings.pkl' — copy it to Raspberry Pi.
#
# ============================================================

import os
import pickle
import cv2

# face_recognition library uses dlib under the hood.
# Install on PC:  pip install face-recognition
# Install on RPi: sudo apt-get install -y cmake libboost-python-dev
#                 pip install face-recognition
import face_recognition

KNOWN_FACES_DIR    = "known_faces"
ENCODINGS_OUTPUT   = "face_encodings.pkl"

def register_faces():
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created '{KNOWN_FACES_DIR}/' folder.")
        print("Add sub-folders named after each person, with their photos inside.")
        print("Example: known_faces/Rajendran/photo1.jpg")
        return

    known_encodings = []
    known_names     = []
    
    persons = [d for d in os.listdir(KNOWN_FACES_DIR)
               if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]

    if not persons:
        print("No person folders found in 'known_faces/'. Add some photos first.")
        return

    print(f"Found {len(persons)} person(s): {', '.join(persons)}")

    for person_name in persons:
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
        image_files   = [
            f for f in os.listdir(person_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_files:
            print(f"  ⚠ No images found for '{person_name}'. Skipping.")
            continue

        print(f"\n  Processing '{person_name}' ({len(image_files)} image(s))...")
        encodings_added = 0

        for img_file in image_files:
            img_path = os.path.join(person_folder, img_file)
            image    = face_recognition.load_image_file(img_path)
            encs     = face_recognition.face_encodings(image)

            if len(encs) == 0:
                print(f"    ⚠ No face found in '{img_file}'. Skip.")
                continue
            if len(encs) > 1:
                print(f"    ⚠ Multiple faces in '{img_file}'. Using the first one.")

            known_encodings.append(encs[0])
            known_names.append(person_name)
            encodings_added += 1

        print(f"  ✓ '{person_name}' registered with {encodings_added} encoding(s).")

    if not known_encodings:
        print("\nNo valid face encodings generated. Check your photos.")
        return

    # Save encodings
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_OUTPUT, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✅ Saved {len(known_encodings)} encoding(s) to '{ENCODINGS_OUTPUT}'")
    print(f"   Copy this file to your Raspberry Pi in 'c:/Rajendran/blind/'")

if __name__ == "__main__":
    register_faces()
