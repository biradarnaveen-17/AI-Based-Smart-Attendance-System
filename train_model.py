import face_recognition
import cv2
import numpy as np
import os
import pickle

print("Loading dataset and encoding faces...")

known_face_encodings = []
known_face_names = []
base_dir = "dataset" # Folder containing individual student folders

# Loop through each person's folder in the dataset directory
for person_name in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person_name)
    if os.path.isdir(person_dir):
        print(f"Processing images for {person_name}...")
        for image_name in os.listdir(person_dir):
            if image_name.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}. Skipping.")
                    continue
                # Convert BGR (OpenCV default) to RGB (face_recognition default)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Find all the faces and their encodings in the current image
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                for face_encoding in face_encodings:
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

print(f"Finished encoding {len(known_face_encodings)} faces from {len(os.listdir(base_dir))} individuals.")

# Save the encodings and names for later use
with open('encodings.pkl', 'wb') as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

print("Encodings saved to encodings.pkl")