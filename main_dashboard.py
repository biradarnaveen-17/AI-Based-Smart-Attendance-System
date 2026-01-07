import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import sqlite3
import time

# --- SETTINGS ---
CONFIDENCE_THRESHOLD = 65  
DB_NAME = "attendance_system.db"

# --- PATHS ---
DATASET_DIR = Path("dataset")
TRAINER_FILE = "trainer.yml"
STUDENT_MAP_FILE = "student_map.csv"
ATTENDANCE_FILE = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Ensure directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
if not os.path.exists(STUDENT_MAP_FILE):
    with open(STUDENT_MAP_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name"])

class AttendanceDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System (Multi-Angle Capable)")
        self.root.geometry("950x700")
        self.root.configure(bg="#2c3e50")

        self.already_marked = set()
        self.init_db()

        # HEADER
        tk.Label(root, text="SMART ATTENDANCE DASHBOARD", font=("Verdana", 20, "bold"), bg="#34495e", fg="white", pady=20).pack(fill=tk.X)

        # BUTTONS FRAME
        frame = tk.Frame(root, bg="#2c3e50")
        frame.pack(expand=True)
        btn_style = {"font": ("Arial", 12), "fg": "white", "width": 35, "height": 2}

        tk.Button(frame, text="1. Register Student (Add Photos)", bg="#27ae60", command=self.register_student, **btn_style).pack(pady=8)
        tk.Button(frame, text="2. Train Model", bg="#e67e22", command=self.train_model, **btn_style).pack(pady=8)
        tk.Button(frame, text="3. Start Camera Attendance", bg="#2980b9", command=self.start_attendance_your_code, **btn_style).pack(pady=8)
        tk.Button(frame, text="4. Manual Attendance Entry", bg="#c0392b", command=self.manual_attendance_window, **btn_style).pack(pady=8)
        tk.Button(frame, text="🔄 Start New Class Session", bg="#f39c12", command=self.start_new_class, **btn_style).pack(pady=20)
        tk.Button(frame, text="📂 Open Attendance Sheet", bg="#8e44ad", command=self.open_csv, **btn_style).pack(pady=8)

        self.status_label = tk.Label(root, text="System Ready", bg="#2c3e50", fg="#ecf0f1", font=("Arial", 10))
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def init_db(self):
        try:
            self.conn = sqlite3.connect(DB_NAME)
            self.cursor = self.conn.cursor()
            self.cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)")
            self.cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, name TEXT, time TEXT, date TEXT, method TEXT)")
            self.conn.commit()
        except Exception as err:
            messagebox.showerror("Database Error", f"Error creating database: {err}")

    def start_new_class(self):
        self.already_marked.clear()
        self.log_separator_csv()
        messagebox.showinfo("New Class", "Session Reset!")

    def log_separator_csv(self):
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["---", "NEW CLASS STARTED", datetime.now().strftime('%H:%M:%S'), "---", "---"])

    # --- 1. REGISTER (Updated with Conflict Check & Append Mode) ---
    def register_student(self):
        s_id = simpledialog.askinteger("Input", "Enter Student ID (Number):", parent=self.root)
        if not s_id: return
        s_name = simpledialog.askstring("Input", "Enter Student Name:", parent=self.root)
        if not s_name: return
        
        # 1. CHECK CONFLICTS
        existing_name = self.get_student_name_by_id(s_id)
        
        if existing_name:
            if existing_name.lower() != s_name.lower():
                messagebox.showerror("ID Conflict", 
                                     f"ERROR: ID {s_id} is already registered to '{existing_name}'.\n\n"
                                     f"You cannot register this ID as '{s_name}'.\n"
                                     "Use a different ID or enter the correct name to add photos.")
                return
            else:
                if not messagebox.askyesno("Add Photos?", f"ID {s_id} ({existing_name}) already exists.\nDo you want to add MORE photos to this student?"):
                    return
        
        # 2. DETERMINE START COUNT (Don't overwrite!)
        existing_files = [f for f in os.listdir(DATASET_DIR) if f.startswith(f"User.{s_id}.")]
        max_count = 0
        if existing_files:
            # Extract numbers from User.1.45.jpg -> 45
            for f in existing_files:
                try:
                    parts = f.split('.')
                    if len(parts) >= 3:
                        num = int(parts[2])
                        if num > max_count: max_count = num
                except: pass
        
        start_count = max_count + 1
        target_count = start_count + 50 # Capture 50 NEW photos

        messagebox.showinfo("Instructions", 
                            f"Starting capture from image #{start_count}.\n"
                            "Look at the camera and SLOWLY turn your head:\n"
                            "- Left, Right, Up, Down\n"
                            "We need different angles!")

        self.save_student_name(s_id, s_name)
        try:
            self.cursor.execute("INSERT OR IGNORE INTO students (id, name) VALUES (?, ?)", (s_id, s_name))
            self.conn.commit()
        except: pass
        
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(min_detection_confidence=0.5)
        cap = cv2.VideoCapture(0)
        
        current_pic = start_count
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    padding = 20
                    x, y = max(0, x - padding), max(0, y - padding)
                    w, h = w + (padding*2), h + (padding*2)
                    if x+w > iw or y+h > ih: continue

                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0:
                        filename = DATASET_DIR / f"User.{s_id}.{current_pic}.jpg"
                        cv2.imwrite(str(filename), cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Saved: {current_pic}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        current_pic += 1
                        cv2.waitKey(150) # Delay for angles

            cv2.imshow("Registering - TURN HEAD SLOWLY", frame)
            
            if current_pic >= target_count or cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        messagebox.showinfo("Success", f"Added 50 new photos for {s_name}!\nTotal photos: {current_pic - 1}\n\nDon't forget to click 'Train Model'.")

    def get_student_name_by_id(self, s_id):
        # Check CSV first
        if os.path.exists(STUDENT_MAP_FILE):
            with open(STUDENT_MAP_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and int(row[0]) == s_id:
                        return row[1]
        return None

    def save_student_name(self, s_id, s_name):
        data = {}
        if os.path.exists(STUDENT_MAP_FILE):
            with open(STUDENT_MAP_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row: data[int(row[0])] = row[1]
        
        # Only update if new (or reinforcing same name)
        data[s_id] = s_name
        
        with open(STUDENT_MAP_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name"])
            for key, val in data.items():
                writer.writerow([key, val])

    # --- 2. TRAIN ---
    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            try:
                pil_img = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_img, 'uint8')
                id = int(os.path.split(image_path)[-1].split(".")[1])
                face_samples.append(img_numpy)
                ids.append(id)
            except: pass
        
        if not ids:
            messagebox.showerror("Error", "No images found.")
            return

        recognizer.train(face_samples, np.array(ids))
        recognizer.write(TRAINER_FILE)
        messagebox.showinfo("Success", f"Model trained on {len(np.unique(ids))} students!")

    # --- 3. ATTENDANCE ---
    def start_attendance_your_code(self):
        if not os.path.exists(TRAINER_FILE):
            messagebox.showerror("Error", "Trainer file missing!")
            return

        names = {}
        if os.path.exists(STUDENT_MAP_FILE):
            with open(STUDENT_MAP_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row: names[int(row[0])] = row[1]

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    padding = 10
                    x, y = max(0, x - padding), max(0, y - padding)
                    w, h = w + (padding*2), h + (padding*2)
                    if x+w > iw or y+h > ih: continue

                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0:
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        try:
                            id, confidence = recognizer.predict(gray_face)

                            if confidence < CONFIDENCE_THRESHOLD:
                                name = names.get(id, f"ID:{id}")
                                color = (0, 255, 0)
                                conf_display = f"{round(100 - confidence)}%"
                                if id not in self.already_marked:
                                    self.mark_database(id, name, "Auto-Camera")
                                    self.already_marked.add(id)
                                    self.status_label.config(text=f"Marked: {name}")
                            else:
                                name = "Unknown"
                                color = (0, 0, 255)
                                conf_display = "Low Match"

                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, f"{name}", (x+5, y-5), font, 1, (255, 255, 255), 2)
                            cv2.putText(frame, f"Dist:{round(confidence)}", (x+5, y+h+20), font, 0.6, (0, 255, 255), 1)

                        except Exception: pass

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def manual_attendance_window(self):
        manual_win = tk.Toplevel(self.root)
        manual_win.title("Manual Entry")
        manual_win.geometry("300x200")
        tk.Label(manual_win, text="Enter Student ID:", font=("Arial", 12)).pack(pady=10)
        entry_id = tk.Entry(manual_win, font=("Arial", 12))
        entry_id.pack(pady=5)
        def submit_manual():
            sid_str = entry_id.get()
            if not sid_str.isdigit(): return
            sid = int(sid_str)
            if sid in self.already_marked: 
                messagebox.showwarning("Warning", "Already marked!")
                return
            name = self.get_student_name_by_id(sid)
            if name:
                self.mark_database(sid, name, "Manual-Entry")
                self.already_marked.add(sid)
                manual_win.destroy()
                messagebox.showinfo("Success", f"Marked {name}")
            else:
                messagebox.showerror("Error", "ID Not Found")
        tk.Button(manual_win, text="Mark Present", bg="#27ae60", fg="white", command=submit_manual).pack(pady=20)

    def mark_database(self, s_id, name, method):
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        date_str = now.strftime('%Y-%m-%d')
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([s_id, name, time_str, date_str, method])
        try:
            self.cursor.execute("INSERT INTO attendance (student_id, name, time, date, method) VALUES (?, ?, ?, ?, ?)", (s_id, name, time_str, date_str, method))
            self.conn.commit()
        except: pass

    def open_csv(self):
        if os.path.exists(ATTENDANCE_FILE): os.startfile(ATTENDANCE_FILE)
        else: messagebox.showinfo("Info", "No attendance file found today.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceDashboard(root)
    root.mainloop()