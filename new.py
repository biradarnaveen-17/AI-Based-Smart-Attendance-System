import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from PIL import Image
from datetime import datetime, date
from pathlib import Path
import sqlite3
import threading 
import socket
from flask import Flask, render_template

# --- ABSOLUTE PATH SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "attendance_system.db"
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_FILE = BASE_DIR / "trainer.yml"
ATTENDANCE_FILE = BASE_DIR / f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

# --- WEB SERVER ---
app_flask = Flask(__name__)

@app_flask.route("/attendance")
def attendance_today():
    today_str = date.today().strftime("%Y-%m-%d")
    students = []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (today_str,))
        rows = cursor.fetchall()
        for row in rows:
            students.append({"name": row[0], "time": row[1]})
        conn.close()
    except: pass
    return render_template("student_list.html", students=students)

class AttendanceSystem(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Based Smart Attendance System")
        self.geometry("1100x850")
        self.configure(fg_color="#1a1c1e")
        
        self.session_marked = set()
        self.init_db()
        
        # --- SIDEBAR (CSV REMOVED) ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#111214")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        ctk.CTkLabel(self.sidebar, text="", font=("Verdana", 20, "bold"), text_color="#3b8ed0").pack(pady=40)

        ctk.CTkButton(self.sidebar, text="Register Student", fg_color="#27ae60", height=45, command=self.register_student).pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="Manage Records", fg_color="#34495e", height=45, command=self.manage_records).pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="Train AI Model", fg_color="#e67e22", height=45, command=self.train_model).pack(pady=10, padx=20)
        ctk.CTkButton(self.sidebar, text="🌐 Start Web View", fg_color="#2980b9", height=45, command=self.show_web_link).pack(pady=30, padx=20)

        # --- MAIN PANEL ---
        self.main_frame = ctk.CTkFrame(self, fg_color="#1a1c1e")
        self.main_frame.pack(expand=True, fill="both")
        
        ctk.CTkLabel(self.main_frame, text="ATTENDANCE DASHBOARD", font=("Verdana", 32, "bold")).pack(pady=(80, 40))
        
        ctk.CTkButton(self.main_frame, text="START CAMERA SCAN", width=500, height=80, font=("Arial", 20, "bold"), command=self.start_camera).pack(pady=15)
        ctk.CTkButton(self.main_frame, text="MANUAL ID ENTRY", width=500, height=50, command=self.manual_entry).pack(pady=10)
        ctk.CTkButton(self.main_frame, text="START NEW SESSION", width=500, height=50, fg_color="#f39c12", command=self.start_new_session).pack(pady=40)
        
        self.status_bar = ctk.CTkLabel(self, text="System Ready | Database Connected", font=("Arial", 14), fg_color="#111214")
        self.status_bar.pack(side=tk.BOTTOM, fill="x")

    def init_db(self):
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT, reg_date TEXT)")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, name TEXT, time TEXT, date TEXT)")
        self.conn.commit()

    def mark_pres(self, s_id, name):
        """Core function to save attendance to DB and CSV"""
        now = datetime.now()
        tm, dt = now.strftime('%H:%M:%S'), now.strftime('%Y-%m-%d')
        
        # Save to Database
        self.cursor.execute("INSERT INTO attendance (student_id, name, time, date) VALUES (?, ?, ?, ?)", (s_id, name, tm, dt))
        self.conn.commit()
        
        # Save to CSV
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([s_id, name, tm, dt])
        
        self.session_marked.add(s_id)
        self.status_bar.configure(text=f"Last Marked: {name} ({s_id}) at {tm}")

    def manual_entry(self):
        """Fixed Manual Entry Logic"""
        s_id = ctk.CTkInputDialog(text="Enter Student ID:", title="Manual Entry").get_input()
        if s_id and s_id.isdigit():
            sid_int = int(s_id)
            
            # Check if already marked in this session
            if sid_int in self.session_marked:
                messagebox.showinfo("Note", "Attendance already marked for this ID in current session.")
                return

            # Check if student exists in DB
            self.cursor.execute("SELECT name FROM students WHERE id=?", (sid_int,))
            res = self.cursor.fetchone()
            
            if res:
                self.mark_pres(sid_int, res[0])
                messagebox.showinfo("Success", f"Attendance recorded for: {res[0]}")
            else:
                messagebox.showerror("Error", f"No student found with ID: {s_id}")

    def start_new_session(self):
        self.session_marked.clear()
        tm = datetime.now().strftime('%H:%M:%S')
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            csv.writer(f).writerow(["---", "NEW SESSION STARTED", tm, "---"])
        messagebox.showinfo("Session", "New session started. You can now re-mark students.")

    def register_student(self):
        s_id = ctk.CTkInputDialog(text="Enter unique ID:", title="Register").get_input()
        s_name = ctk.CTkInputDialog(text="Enter full Name:", title="Register").get_input()
        if not s_id or not s_name: return

        self.cursor.execute("INSERT OR REPLACE INTO students (id, name, reg_date) VALUES (?, ?, ?)", 
                        (int(s_id), s_name, date.today().strftime('%Y-%m-%d')))
        self.conn.commit()

        cap = cv2.VideoCapture(1)
        detector = mp.solutions.face_detection.FaceDetection()
        total_captured = 0
        
        for phase in ["NO MASK", "WITH MASK"]:
            messagebox.showinfo("Register", f"Phase: {phase}\nPress 'C' to start taking 25 photos.")
            p_count = 0
            while p_count < 25:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"{phase}: Press 'C'", (20, 40), 1, 1.5, (0, 255, 255), 2)
                cv2.imshow("Registering...", frame)
                if cv2.waitKey(1) == ord('c'):
                    while p_count < 25:
                        ret, frame = cap.read()
                        frame = cv2.flip(frame, 1)
                        res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if res.detections:
                            for det in res.detections:
                                bbox = det.location_data.relative_bounding_box
                                ih, iw, _ = frame.shape
                                x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
                                face = cv2.cvtColor(frame[max(0,y):y+h, max(0,x):x+w], cv2.COLOR_BGR2GRAY)
                                if face.size > 0:
                                    cv2.imwrite(str(DATASET_DIR / f"User.{s_id}.{total_captured}.jpg"), face)
                                    p_count += 1
                                    total_captured += 1
                                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Saved: {p_count}/25", (x, y-10), 1, 1.2, (0, 255, 0), 2)
                                    cv2.imshow("Registering...", frame)
                                    cv2.waitKey(250)
                        if cv2.waitKey(1) == ord('q'): break
            if cv2.waitKey(1) == ord('q'): break
        cap.release(); cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"Registered {s_name} (ID: {s_id})")

    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = [], []
        image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
        if not image_paths: return messagebox.showerror("Error", "No images found!")
        for path in image_paths:
            faces.append(np.array(Image.open(path).convert('L'), 'uint8'))
            ids.append(int(os.path.split(path)[-1].split(".")[1]))
        recognizer.train(faces, np.array(ids))
        recognizer.write(str(TRAINER_FILE))
        self.cursor.execute("SELECT COUNT(*) FROM students")
        count = self.cursor.fetchone()[0]
        messagebox.showinfo("AI Trainer", f"Successfully trained {count} students!")

    def manage_records(self):
        win = ctk.CTkToplevel(self)
        win.title("Student Records")
        win.geometry("700x600")
        win.attributes("-topmost", True)
        scroll = ctk.CTkScrollableFrame(win, width=650, height=500)
        scroll.pack(pady=20, padx=20)

        def load_list():
            for w in scroll.winfo_children(): w.destroy()
            self.cursor.execute("SELECT * FROM students")
            for s_id, name, r_date in self.cursor.fetchall():
                row = ctk.CTkFrame(scroll)
                row.pack(fill="x", pady=5)
                ctk.CTkLabel(row, text=f"ID: {s_id} | {name}", width=250, anchor="w").pack(side="left", padx=10)
                ctk.CTkButton(row, text="Edit", width=70, command=lambda i=s_id: edit_std(i)).pack(side="right", padx=5)
                ctk.CTkButton(row, text="Del", width=70, fg_color="#c0392b", command=lambda i=s_id: del_std(i)).pack(side="right", padx=5)

        def edit_std(i):
            new_n = ctk.CTkInputDialog(text="New Name:", title="Edit").get_input()
            if new_n:
                self.cursor.execute("UPDATE students SET name=? WHERE id=?", (new_n, i))
                self.cursor.execute("UPDATE attendance SET name=? WHERE student_id=?", (new_n, i))
                self.conn.commit()
                load_list()

        def del_std(i):
            if messagebox.askyesno("Confirm", "Delete records and photos?"):
                self.cursor.execute("DELETE FROM students WHERE id=?", (i,))
                self.cursor.execute("DELETE FROM attendance WHERE student_id=?", (i,))
                self.conn.commit()
                for f in os.listdir(DATASET_DIR):
                    if f.startswith(f"User.{i}."): os.remove(DATASET_DIR / f)
                load_list()
        load_list()

    def start_camera(self):
        if not os.path.exists(TRAINER_FILE): return messagebox.showerror("Error", "Train model first!")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(TRAINER_FILE))
        self.cursor.execute("SELECT id, name FROM students")
        names_map = {row[0]: row[1] for row in self.cursor.fetchall()}
        cap = cv2.VideoCapture(1)
        detector = mp.solutions.face_detection.FaceDetection()
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.detections:
                for det in res.detections:
                    bbox = det.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
                    roi = cv2.cvtColor(frame[max(0,y):y+h, max(0,x):x+w], cv2.COLOR_BGR2GRAY)
                    if roi.size > 0:
                        s_id, conf = recognizer.predict(roi)
                        name = names_map.get(s_id, "Unknown") if conf < 60 else "Unknown"
                        if name != "Unknown" and s_id not in self.session_marked:
                            self.mark_pres(s_id, name)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"{name}", (x, y-10), 1, 1.5, (0, 255, 0), 2)
            cv2.imshow("Scanner", frame)
            if cv2.waitKey(1) == ord('q'): break
        cap.release(); cv2.destroyAllWindows()

    def show_web_link(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]; s.close()
        except: ip = "127.0.0.1"
        link = f"http://{ip}:5001/attendance"
        threading.Thread(target=lambda: app_flask.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False), daemon=True).start()
        win = ctk.CTkToplevel(self); win.title("Web Server Active"); win.geometry("400x200"); win.attributes("-topmost", True)
        ctk.CTkLabel(win, text="Phone URL:").pack(pady=20)
        entry = ctk.CTkEntry(win, width=350); entry.insert(0, link); entry.configure(state="readonly"); entry.pack(pady=10)

if __name__ == "__main__":
    app = AttendanceSystem()
    app.mainloop()