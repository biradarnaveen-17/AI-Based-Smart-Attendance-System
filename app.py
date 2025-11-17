import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import subprocess
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import pickle
import time
from datetime import datetime
import os
import threading
import queue 
import shutil 
import json 
import webbrowser 
import sqlite3

# --- NEW: Excel/Pandas Imports ---
try:
    import pandas as pd
except ImportError:
    messagebox.showerror("Missing Libraries", "Required library 'pandas' not found.\n\nPlease run: pip install pandas")
    exit()

try:
    import openpyxl
except ImportError:
    messagebox.showerror("Missing Libraries", "Required library 'openpyxl' not found.\n\nPlease run: pip install openpyxl")
    exit()


# --- Configuration ---
RECOGNITION_THRESHOLD = 0.55 
PROCESS_EVERY_N_FRAMES = 5 

# --- DB Configuration ---
DB_NAME = 'attendance.db' 


class VideoProcessor(threading.Thread):
    def __init__(self, video_capture, known_face_encodings, known_face_names, recognition_threshold, results_queue):
        threading.Thread.__init__(self)
        self.video_capture = video_capture
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.recognition_threshold = recognition_threshold
        self.results_queue = results_queue
        self.running = True
        
        self.frame_counter = 0
        self.last_known_info = [] 
        self.last_known_count = 0
        
        self.last_frame_time = time.time()
        self.fps = 0

    def run(self):
        while self.running:
            start_time = time.time()
            
            if not self.video_capture or not self.video_capture.isOpened():
                self.running = False
                break
                
            ret, frame = self.video_capture.read()
            if not ret:
                print("VideoProcessor: Failed to grab frame.")
                time.sleep(0.01)
                continue

            self.frame_counter += 1
            
            recognized_info = self.last_known_info
            recognized_count = self.last_known_count
            is_fresh_data = False 

            if self.frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                is_fresh_data = True 
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                recognized_info = [] 
                current_frame_recognized_students = set()

                for i, face_encoding in enumerate(face_encodings):
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = None

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        best_match_distance = face_distances[best_match_index]

                        if best_match_distance <= self.recognition_threshold:
                            name = self.known_face_names[best_match_index]
                            current_frame_recognized_students.add(name)
                            confidence = (1 - best_match_distance / self.recognition_threshold) * 100
                            if confidence < 0: confidence = 0

                    top, right, bottom, left = face_locations[i]
                    scaled_top, scaled_right, scaled_bottom, scaled_left = top * 2, right * 2, bottom * 2, left * 2

                    recognized_info.append({
                        "name": name,
                        "confidence": confidence,
                        "location": (scaled_top, scaled_right, scaled_bottom, scaled_left) 
                    })
                
                recognized_count = len(current_frame_recognized_students)
                
                self.last_known_info = recognized_info
                self.last_known_count = recognized_count
            
            try:
                self.results_queue.put_nowait({
                    "frame": frame, 
                    "recognized_info": self.last_known_info, 
                    "recognized_count": self.last_known_count,
                    "is_fresh_data": is_fresh_data 
                })
            except queue.Full:
                pass 

            end_time = time.time()
            if (end_time - start_time) > 0:
                self.fps = 1 / (end_time - start_time)

    def stop(self):
        self.running = False


class AttendanceSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Attendance System (Integrated)")
        self.root.geometry("1280x720") 
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.training_results_queue = queue.Queue(maxsize=1)

        # --- DB INTEGRATION (SQLite) ---
        self.db_conn = self.connect_to_database()
        if self.db_conn:
            self.setup_database()
        else:
            messagebox.showerror("Database Error", "Failed to create local SQLite database.")
            self.root.destroy()
            return
        # --- END DB INTEGRATION ---
        
        self.known_face_encodings = []
        self.known_face_names = []
        self_load_encodings_called = False
        try:
            self.load_encodings()
            self_load_encodings_called = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load encodings on startup: {e}\nMake sure 'encodings.pkl' exists or register a new student.")
            
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open webcam. Please ensure it is connected and not in use.")
            self.root.destroy()
            return

        self.processing_results_queue = queue.Queue(maxsize=2)

        self.video_processor = VideoProcessor(
            self.video_capture,
            self.known_face_encodings,
            self.known_face_names,
            RECOGNITION_THRESHOLD,
            self.processing_results_queue
        )
        self.video_processor.daemon = True 
        self.video_processor.start()

        self.students_marked_today = set()
        self.load_todays_attendance()

        self.attendance_records_data = [] 

        self.create_widgets()
        self.update_video_feed() 
        self.check_training_results()
        
        if self_load_encodings_called:
            self.update_attendance_display(show_popup=False) 
        
        self.reg_capture = None
        self.reg_window = None
        self.reg_video_label = None
        self.reg_cam_running = False
        self.current_reg_frame = None
        self.snapshot_count = 0

    def connect_to_database(self):
        """Connects to the SQLite database file."""
        try:
            conn = sqlite3.connect(DB_NAME)
            print(f"Connected to database '{DB_NAME}'")
            return conn
        except sqlite3.Error as e:
            print(f"Error connecting to SQLite: {e}")
            return None

    def setup_database(self):
        """Creates the attendance table if it doesn't exist."""
        try:
            cursor = self.db_conn.cursor()
            # Changed SQL syntax for SQLite
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence TEXT,
                UNIQUE(name, date, time)
            )
            """)
            self.db_conn.commit()
            print("Attendance table is ready.")
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to create table: {e}")

    def load_todays_attendance(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        print(f"Loading students already marked for {today_str}...")
        try:
            # No need to check connection, sqlite3 auto-handles
            cursor = self.db_conn.cursor()
            # Changed SQL placeholder from %s to ?
            query = "SELECT DISTINCT name FROM attendance WHERE date = ?"
            cursor.execute(query, (today_str,))
            rows = cursor.fetchall()
            cursor.close()
            
            for row in rows:
                self.students_marked_today.add(row[0])
            
            print(f"Loaded {len(self.students_marked_today)} students: {self.students_marked_today}")
                
        except sqlite3.Error as e:
             print(f"Error loading today's attendance: {e}")

    def load_encodings(self):
        try:
            with open('encodings.pkl', 'rb') as f:
                data = pickle.load(f)
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_encodings)} known faces for recognition.")
            
            if hasattr(self, 'video_processor'):
                self.video_processor.known_face_encodings = data['encodings']
                self.video_processor.known_face_names = data['names']
                
        except FileNotFoundError:
            print("encodings.pkl not found. Please register a student to create it.")
            self.known_face_encodings = []
            self.known_face_names = []
            if not os.path.exists("dataset"):
                os.makedirs("dataset") 
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load encodings: {e}")
            if hasattr(self, 'root'):
                self.root.destroy()
            return

    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg="#282C34", padx=10, pady=10)
        header_frame.pack(fill="x")
        header_label = tk.Label(header_frame, text="AI Attendance System (Integrated)", font=("Arial", 20, "bold"), fg="white", bg="#282C34")
        header_label.pack(pady=5)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        live_camera_frame = tk.Frame(self.notebook, bg="lightgray")
        self.notebook.add(live_camera_frame, text="Live Camera")

        live_camera_frame.grid_rowconfigure(1, weight=1) 
        live_camera_frame.grid_columnconfigure(0, weight=1)

        stats_frame = tk.Frame(live_camera_frame, bg="lightgray")
        stats_frame.grid(row=0, column=0, sticky="ew", pady=5, padx=10) 

        self.recognized_count_label = tk.Label(stats_frame, text="Recognized: 0", font=("Arial", 12), bg="lightgray")
        self.recognized_count_label.pack(side="left")

        self.fps_label = tk.Label(stats_frame, text="FPS: 0", font=("Arial", 12), bg="lightgray")
        self.fps_label.pack(side="right")

        self.video_label = tk.Label(live_camera_frame, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=5) 

        video_control_frame = tk.Frame(live_camera_frame, bg="lightgray")
        video_control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5) 

        btn_register = tk.Button(video_control_frame, text="+ Register Student", command=self.register_student, bg="#61AFEF", fg="white", font=("Arial", 10, "bold"))
        btn_register.pack(side="left", padx=5) 
        btn_manual = tk.Button(video_control_frame, text="Manual Mark", command=self.manual_mark, bg="#E5C07B", fg="white", font=("Arial", 10, "bold"))
        btn_manual.pack(side="left", padx=5)
        btn_refresh = tk.Button(video_control_frame, text="Refresh", command=lambda: self.update_attendance_display(show_popup=True), bg="#98C379", fg="white", font=("Arial", 10, "bold"))
        btn_refresh.pack(side="left", padx=5)
        
        # --- MODIFIED: Button for Excel ---
        btn_export = tk.Button(video_control_frame, text="Export (Excel)", command=self.export_to_excel, bg="#C678DD", fg="white", font=("Arial", 10, "bold"))
        btn_export.pack(side="left", padx=5)
        
        btn_web = tk.Button(video_control_frame, text="View Daily List (Web)", command=self.open_web_status, bg="#E06C75", fg="white", font=("Arial", 10, "bold"))
        btn_web.pack(side="right", padx=5) 

        attendance_records_frame = tk.Frame(self.notebook, bg="lightgray")
        # ... (rest of create_widgets) ...
        self.notebook.add(attendance_records_frame, text="Attendance Records")

        self.attendance_tree = ttk.Treeview(attendance_records_frame, columns=("Name", "Date", "Time", "Confidence"), show="headings")
        self.attendance_tree.heading("Name", text="Name")
        self.attendance_tree.heading("Date", text="Date")
        self.attendance_tree.heading("Time", text="Time")
        self.attendance_tree.heading("Confidence", text="Confidence")
        self.attendance_tree.column("Name", width=150, anchor="center")
        self.attendance_tree.column("Date", width=100, anchor="center")
        self.attendance_tree.column("Time", width=100, anchor="center")
        self.attendance_tree.column("Confidence", width=100, anchor="center")
        self.attendance_tree.pack(expand=True, fill="both", padx=10, pady=10)

        tree_scrollbar_y = ttk.Scrollbar(attendance_records_frame, orient="vertical", command=self.attendance_tree.yview)
        tree_scrollbar_y.pack(side="right", fill="y")
        self.attendance_tree.config(yscrollcommand=tree_scrollbar_y.set)

    def open_web_status(self):
        webbrowser.open_new_tab("http://127.0.0.1:5001/attendance")

    def update_live_status(self, name):
        status_data = {"name": name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            with open('live_status.json', 'w') as f:
                json.dump(status_data, f)
        except Exception as e:
            print(f"Error writing live status: {e}")

    def update_live_status_threaded(self, name):
        threading.Thread(target=self.update_live_status, args=(name,), daemon=True).start()

    def mark_attendance(self, name, confidence):
        if name != "Unknown" and name not in self.students_marked_today:
            self.students_marked_today.add(name)
            
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M:%S") 
            conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"
            
            time_display_str = timestamp.strftime("%I:%M:%S %p")
            record = (name, date_str, time_display_str, conf_str)
            
            self.attendance_tree.insert("", "end", values=record)
            
            print(f"Marking {name} as present for {date_str}.")
            threading.Thread(target=self.save_attendance_to_db, args=(name, date_str, time_str, conf_str), daemon=True).start()
            
            self.update_live_status_threaded(name)

    def save_attendance_to_db(self, name, date_str, time_str, conf_str):
        """Saves a single attendance record to the SQLite database."""
        try:
            # Note: For threading, it's safer to create a new connection
            # for each write operation. SQLite can be sensitive.
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            # Changed SQL placeholder from %s to ?
            query = "INSERT INTO attendance (name, date, time, confidence) VALUES (?, ?, ?, ?)"
            values = (name, date_str, time_str, conf_str)
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
        except sqlite3.Error as e:
            print(f"Error saving attendance to DB: {e}")

    def load_attendance_from_db(self):
        """Loads all attendance records from the database for the GUI list."""
        self.attendance_records_data = []
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT name, date, time, confidence FROM attendance ORDER BY date DESC, time DESC")
            rows = cursor.fetchall()
            cursor.close()
            
            for row in rows:
                name, date_str, time_str, confidence = row
                # Convert time string (HH:MM:SS) to display format
                try:
                    time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                    time_display_str = time_obj.strftime("%I:%M:%S %p")
                except ValueError:
                    time_display_str = time_str # Fallback
                
                self.attendance_records_data.append((name, date_str, time_display_str, confidence))
                
        except sqlite3.Error as e:
             print(f"Error loading attendance from DB: {e}")

    def update_attendance_display(self, show_popup=False):
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        self.load_attendance_from_db()
        
        for record in self.attendance_records_data:
            self.attendance_tree.insert("", "end", values=record)
        if show_popup:
            messagebox.showinfo("Refresh Complete", f"Loaded {len(self.attendance_records_data)} records from database.", parent=self.root)

    def update_video_feed(self):
        try:
            results = self.processing_results_queue.get_nowait()
            frame = results["frame"] 
            recognized_info = results["recognized_info"]
            recognized_count = results["recognized_count"]
            is_fresh_data = results["is_fresh_data"] 
            
            display_frame = frame.copy() 
            
            for info in recognized_info:
                name = info["name"]
                confidence = info["confidence"]
                top, right, bottom, left = info["location"] 
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                label = f"{name}"
                if confidence is not None:
                    label += f" ({confidence:.1f}%)"
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
                
                if is_fresh_data:
                    self.mark_attendance(name, confidence)
            
            if is_fresh_data:
                self.recognized_count_label.config(text=f"Recognized: {recognized_count}")
                self.fps_label.config(text=f"FPS: {int(self.video_processor.fps)}")
            
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)

            target_width = self.video_label.winfo_width()
            target_height = self.video_label.winfo_height()

            if target_width > 1 and target_height > 1:
                img_aspect = img.width / img.height
                label_aspect = target_width / target_height

                if img_aspect > label_aspect:
                    new_width = target_width
                    new_height = int(target_width / img_aspect)
                else:
                    new_height = target_height
                    new_width = int(target_height * img_aspect)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)
            
            self._last_displayed_image = img_tk 

        except queue.Empty:
            if hasattr(self, '_last_displayed_image'):
                 self.video_label.config(image=self._last_displayed_image)
            
        self.root.after(33, self.update_video_feed)

    def register_student(self):
        self.name_window = tk.Toplevel(self.root)
        self.name_window.title("Register Student - Step 1")
        self.name_window.geometry("350x150")
        self.name_window.transient(self.root)
        self.name_window.grab_set()
        self.name_window.focus_force()
        tk.Label(self.name_window, text="Enter Student's Name:", font=("Arial", 12)).pack(pady=10)
        name_entry = tk.Entry(self.name_window, width=30, font=("Arial", 12))
        name_entry.pack(pady=5, padx=20)
        def on_submit_name():
            student_name = name_entry.get().strip()
            if not student_name:
                messagebox.showwarning("Invalid Name", "Name cannot be empty.", parent=self.name_window)
                return
            invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
            if any(char in student_name for char in invalid_chars):
                messagebox.showwarning("Invalid Name", "Name contains invalid characters.", parent=self.name_window)
                return
            dataset_path = "dataset"
            student_path = os.path.join(dataset_path, student_name)
            if os.path.exists(student_path):
                messagebox.showwarning("Name Exists", "A student with this name already exists.", parent=self.name_window)
                return
            try:
                os.makedirs(student_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create directory: {e}", parent=self.name_window)
                return
            self.name_window.grab_release()
            self.name_window.destroy()
            self.ask_photo_source(student_name, student_path)
        submit_button = tk.Button(self.name_window, text="Next", command=on_submit_name, bg="#61AFEF", fg="white", font=("Arial", 10, "bold"))
        submit_button.pack(pady=20)
        self.name_window.bind('<Return>', lambda event: on_submit_name())

    def ask_photo_source(self, student_name, student_path):
        self.source_window = tk.Toplevel(self.root)
        self.source_window.title("Register Student - Step 2")
        self.source_window.geometry("350x150")
        self.source_window.transient(self.root)
        self.source_window.grab_set()
        self.source_window.focus_force()
        tk.Label(self.source_window, text=f"How do you want to add photos for {student_name}?", font=("Arial", 12)).pack(pady=10)
        button_frame = tk.Frame(self.source_window)
        button_frame.pack(pady=20)
        def on_upload():
            self.source_window.grab_release()
            self.source_window.destroy()
            self.upload_photos(student_name, student_path)
        def on_camera():
            self.source_window.grab_release()
            self.source_window.destroy()
            self.open_registration_camera(student_name, student_path)
        upload_button = tk.Button(button_frame, text="Upload from Files", command=on_upload, bg="#98C379", fg="white", font=("Arial", 10, "bold"))
        upload_button.pack(side="left", padx=10)
        camera_button = tk.Button(button_frame, text="Open Live Camera", command=on_camera, bg="#E5C07B", fg="white", font=("Arial", 10, "bold"))
        camera_button.pack(side="left", padx=10)

    def upload_photos(self, student_name, student_path):
        files = filedialog.askopenfilenames(
            parent=self.root,
            title=f"Select photos for {student_name}",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        if not files:
            messagebox.showinfo("No Files", "No files were selected.", parent=self.root)
            return
        copied_count = 0
        for file in files:
            try:
                filename = os.path.basename(file)
                new_path = os.path.join(student_path, filename)
                shutil.copy(file, new_path)
                copied_count += 1
            except Exception as e:
                print(f"Failed to copy {file}: {e}")
        
        if copied_count > 0:
            self.process_and_update_encodings_in_memory(student_name, student_path, copied_count)

    def open_registration_camera(self, student_name, student_path):
        self.reg_window = tk.Toplevel(self.root)
        self.reg_window.title(f"Register: {student_name}")
        self.reg_window.transient(self.root)
        self.reg_window.grab_set()
        
        print("Stopping main video processor...")
        self.video_processor.stop() 
        self.video_processor.join(timeout=1.0) 
        
        if self.video_capture.isOpened():
            self.video_capture.release()
            print("Main video capture released for registration.")
        
        print("Opening registration camera...")
        self.reg_capture = cv2.VideoCapture(0)
        if not self.reg_capture.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera. Please ensure it is not in use by another app.", parent=self.root)
            self.reg_window.destroy()
            self.reg_capture = None
            self.start_main_video_processor()
            return
            
        self.reg_video_label = tk.Label(self.reg_window, bg="black")
        self.reg_video_label.pack()
        self.snapshot_count = 0
        status_text = f"Photos Saved: {self.snapshot_count}"
        self.reg_status_label = tk.Label(self.reg_window, text=status_text, font=("Arial", 12))
        self.reg_status_label.pack(pady=5)
        btn_frame = tk.Frame(self.reg_window)
        btn_frame.pack(pady=10)
        snapshot_button = tk.Button(btn_frame, text="Take Snapshot", command=lambda: self.save_snapshot(student_path), bg="#61AFEF", fg="white", font=("Arial", 10, "bold"))
        snapshot_button.pack(side="left", padx=10)
        
        done_button = tk.Button(btn_frame, text="Done", command=lambda: self.close_registration_camera(student_name, student_path), bg="#98C379", fg="white", font=("Arial", 10, "bold"))
        done_button.pack(side="left", padx=10)

        self.reg_cam_running = True
        self.update_reg_feed()
        self.reg_window.protocol("WM_DELETE_WINDOW", lambda: self.close_registration_camera(student_name, student_path))

    def update_reg_feed(self):
        if not self.reg_cam_running:
            return
        
        if not self.reg_capture or not self.reg_capture.isOpened():
             print("Registration capture is not open.")
             self.close_registration_camera() 
             return

        ret, frame = self.reg_capture.read()
        if ret:
            self.current_reg_frame = frame 
            display_frame = frame.copy()
            rgb_frame_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame_small)
            
            for top, right, bottom, left in face_locations:
                top *= 2; right *= 2; bottom *= 2; left *= 2
                color = (0, 255, 0) if len(face_locations) == 1 else (0, 0, 255) 
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            
            img.thumbnail((640, 480), Image.LANCZOS) 
            img_tk = ImageTk.PhotoImage(image=img)
            self.reg_video_label.img_tk = img_tk
            self.reg_video_label.config(image=img_tk)
            self.reg_window.after(10, self.update_reg_feed)
        else:
            print("Failed to grab frame from reg_capture")
            self.reg_window.after(10, self.update_reg_feed)

    def save_snapshot(self, student_path):
        if self.current_reg_frame is None:
            messagebox.showwarning("No Image", "Camera frame not available.", parent=self.reg_window)
            return
        
        rgb_frame_small = cv2.cvtColor(cv2.resize(self.current_reg_frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame_small)
        
        if len(face_locations) == 0:
            messagebox.showwarning("No Face", "No face detected. Please face the camera.", parent=self.reg_window)
            return
        elif len(face_locations) > 1:
            messagebox.showwarning("Multiple Faces", "Too many faces. Please ensure only one person is in frame.", parent=self.reg_window)
            return
        
        try:
            self.snapshot_count += 1
            filename = f"capture_{self.snapshot_count}.jpg"
            filepath = os.path.join(student_path, filename)
            cv2.imwrite(filepath, self.current_reg_frame)
            self.reg_status_label.config(text=f"Photos Saved: {self.snapshot_count}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save snapshot: {e}", parent=self.reg_window)
            self.snapshot_count -= 1 

    def close_registration_camera(self, student_name=None, student_path=None):
        self.reg_cam_running = False
        
        if self.reg_capture and self.reg_capture.isOpened():
            self.reg_capture.release()
            print("Registration camera released.")
        self.reg_capture = None
        
        if self.reg_window:
            self.reg_window.grab_release()
            self.reg_window.destroy()
            self.reg_window = None
            
        self.start_main_video_processor()
        
        if self.snapshot_count > 0 and student_name and student_path:
            self.process_and_update_encodings_in_memory(student_name, student_path, self.snapshot_count)

    def start_main_video_processor(self):
        """Helper function to safely start the main video processor."""
        if not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(0)
            print("Main video capture re-initialized.")

        if hasattr(self, 'video_processor') and self.video_processor.is_alive():
             print("Main video processor is already running.")
             return

        self.video_processor = VideoProcessor(self.video_capture, self.known_face_encodings, self.known_face_names, RECOGNITION_THRESHOLD, self.processing_results_queue)
        self.video_processor.daemon = True
        self.video_processor.start()
        print("Main video processor restarted.")

    def process_and_update_encodings_in_memory(self, student_name, student_path, photo_count):
        print(f"Starting background encoding for {student_name}...")
        
        self.root.config(cursor="watch")
        self.processing_popup = tk.Toplevel(self.root)
        self.processing_popup.title("Processing...")
        self.processing_popup.geometry("300x100")
        tk.Label(self.processing_popup, text=f"Encoding {photo_count} photos for {student_name}...\nThis may take a moment.").pack(pady=20)
        self.processing_popup.transient(self.root)
        self.processing_popup.grab_set()
        
        threading.Thread(target=self._background_encoding_task, 
                         args=(student_name, student_path), 
                         daemon=True).start()

    def _background_encoding_task(self, student_name, student_path):
        """THIS FUNCTION RUNS ON A BACKGROUND THREAD."""
        new_encodings_count = 0
        try:
            current_known_encodings = []
            current_known_names = []
            try:
                with open('encodings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    current_known_encodings = data['encodings']
                    current_known_names = data['names']
            except FileNotFoundError:
                pass 
            
            temp_encodings = []
            temp_names = []
            for i, name in enumerate(current_known_names):
                if name != student_name:
                    temp_encodings.append(current_known_encodings[i])
                    temp_names.append(current_known_names[i])

            for image_name in os.listdir(student_path):
                if image_name.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(student_path, image_name)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings_list = face_recognition.face_encodings(image)
                        
                        if face_encodings_list:
                            face_encoding = face_encodings_list[0]
                            temp_encodings.append(face_encoding)
                            temp_names.append(student_name)
                            new_encodings_count += 1
                        else:
                            print(f"Warning: No face detected in {image_path}. Skipping.")
                    except IndexError:
                        print(f"Warning: No face detected in {image_path}. Skipping.")
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            
            with open('encodings.pkl', 'wb') as f:
                pickle.dump({'encodings': temp_encodings, 'names': temp_names}, f)
            
            self.training_results_queue.put({"student_name": student_name, "count": new_encodings_count})
            
        except Exception as e:
            self.training_results_queue.put({"student_name": student_name, "count": 0, "error": str(e)})

    def check_training_results(self):
        """Checks the queue for results from the background thread."""
        try:
            result = self.training_results_queue.get_nowait()
            
            self.root.config(cursor="")
            if hasattr(self, 'processing_popup') and self.processing_popup:
                try:
                    self.processing_popup.grab_release()
                    self.processing_popup.destroy()
                    self.processing_popup = None
                except tk.TclError:
                    pass # Window already destroyed

            if "error" in result:
                messagebox.showerror("Encoding Error", f"Failed to encode faces for {result['student_name']}: {result['error']}", parent=self.root)
                return

            student_name = result["student_name"]
            new_encodings_count = result["count"]

            if new_encodings_count > 0:
                self.load_encodings() # Reload all encodings
                
                messagebox.showinfo("Update Complete", 
                                    f"Added {new_encodings_count} new images for {student_name}.\nThey can be recognized immediately.",
                                    parent=self.root)
            else:
                 messagebox.showwarning("Update Failed", f"No faces were found in the photos for {student_name}.", parent=self.root)

        except queue.Empty:
            pass 
        
        self.root.after(1000, self.check_training_results)
    
    def manual_mark(self):
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Manual Mark Attendance")
        manual_window.geometry("300x150")
        manual_window.transient(self.root)  
        manual_window.grab_set()          
        manual_window.focus_force()        
        tk.Label(manual_window, text="Enter Student's Name:", font=("Arial", 12)).pack(pady=10)
        name_entry = tk.Entry(manual_window, width=30, font=("Arial", 12))
        name_entry.pack(pady=5, padx=20)
        def submit_manual_mark():
            student_name = name_entry.get().strip()
            if student_name:
                not_marked = student_name not in self.students_marked_today
                
                self.mark_attendance(student_name, confidence=None)

                if not not_marked:
                     # This logic adds a *visual* duplicate to the list if already marked
                     # The core self.mark_attendance() prevents a DB duplicate
                     timestamp = datetime.now()
                     date_str = timestamp.strftime("%Y-%m-%d")
                     time_display_str = timestamp.strftime("%I:%M:%S %p")
                     record = (student_name, date_str, time_display_str, "N/A (Manual)")
                     self.attendance_tree.insert("", "end", values=record)
                
                messagebox.showinfo("Success", f"Manually marked attendance for {student_name}.", parent=manual_window)
                manual_window.grab_release() 
                manual_window.destroy()
            else:
                messagebox.showwarning("Input Error", "Please enter a student name.", parent=manual_window)
        tk.Button(manual_window, text="Mark Present", command=submit_manual_mark).pack(pady=10)

    # --- NEW: Excel Export Function ---
    def export_to_excel(self):
        """Exports all attendance records to an Excel file."""
        print("Exporting to Excel...")

        # 1. Get the save location
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Excel Export",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")]
        )
        
        if not filename:
            print("Export cancelled.")
            return

        # 2. Get all data from the database
        all_records = []
        try:
            conn = self.connect_to_database() # Get a fresh connection
            if not conn:
                 messagebox.showerror("Database Error", "Could not connect to database for export.")
                 return
                 
            cursor = conn.cursor()
            cursor.execute("SELECT name, date, time, confidence FROM attendance ORDER BY date DESC, time DESC")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                messagebox.showinfo("Export", "No attendance data found in the database.", parent=self.root)
                return

            # Format the data for Pandas
            for row in rows:
                name, date_str, time_str, confidence = row
                try:
                    time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                    time_display_str = time_obj.strftime("%I:%M:%S %p")
                except ValueError:
                    time_display_str = time_str # Fallback
                
                all_records.append({
                    "Name": name,
                    "Date": date_str,
                    "Time": time_display_str,
                    "Confidence": confidence
                })
        
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to fetch data for export: {e}", parent=self.root)
            return

        # 3. Write to Excel using Pandas
        try:
            df = pd.DataFrame(all_records)
            
            # Make columns wider
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Attendance')
                worksheet = writer.sheets['Attendance']
                worksheet.column_dimensions['A'].width = 25  # Name
                worksheet.column_dimensions['B'].width = 12  # Date
                worksheet.column_dimensions['C'].width = 15  # Time
                worksheet.column_dimensions['D'].width = 12  # Confidence

            messagebox.showinfo("Export Successful", f"Successfully exported {len(all_records)} records to:\n{filename}", parent=self.root)

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write Excel file: {e}", parent=self.root)

    # --- END of Excel Function ---

    def on_closing(self):
        print("Saving updated encodings to encodings.pkl...")
        try:
            with open('encodings.pkl', 'wb') as f:
                pickle.dump({'encodings': self.known_face_encodings, 'names': self.known_face_names}, f)
            print("Encodings saved successfully.")
        except Exception as e:
            print(f"Error saving encodings: {e}")

        print("Closing application...")
        self.video_processor.stop()
        self.video_processor.join(timeout=2.0)
        
        self.reg_cam_running = False
        if self.reg_capture and self.reg_capture.isOpened():
            self.reg_capture.release()
            print("Registration camera released.")
        
        if self.video_capture.isOpened():
            self.video_capture.release()
            print("Main video capture released.")
            
        if self.db_conn:
            self.db_conn.close()
            print("Database connection closed.")
        
        cv2.destroyAllWindows()
        self.root.destroy()
        print("Application closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()
