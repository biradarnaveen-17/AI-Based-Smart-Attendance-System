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
import mysql.connector
from mysql.connector import Error
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Configuration ---
RECOGNITION_THRESHOLD = 0.55 
PROCESS_EVERY_N_FRAMES = 5 

# --- DB Configuration ---
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = '' # Your XAMPP password, if you have one
DB_NAME = 'ai_attendance_system' 

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
                
                # --- Multi-face detection at 0.5x size ---
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

                    # Rescale locations back to the full frame size (fx=0.5 means multiply by 2)
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
                    "frame": frame, # Full size frame
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

        self.db_conn = self.connect_to_database()
        if self.db_conn:
            self.setup_database()
        else:
            messagebox.showerror("Database Error", "Failed to connect to MySQL database. Please ensure XAMPP is running.")
            self.root.destroy()
            return
        
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
        
        # self.reg_capture = None # <-- CHANGE 1: This line is removed.
        self.reg_window = None
        self.reg_video_label = None
        self.reg_cam_running = False
        self.current_reg_frame = None
        self.snapshot_count = 0

    def connect_to_database(self):
        try:
            conn = mysql.connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
            conn.database = DB_NAME
            print(f"Connected to database '{DB_NAME}'")
            return conn
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None

    def setup_database(self):
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                confidence VARCHAR(10),
                UNIQUE KEY unique_entry (name, date, time)
            )
            """)
            print("Attendance table is ready.")
        except Error as e:
            messagebox.showerror("Database Error", f"Failed to create table: {e}")

    def load_todays_attendance(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        print(f"Loading students already marked for {today_str}...")
        try:
            if not self.db_conn.is_connected():
                self.db_conn = self.connect_to_database()
                
            cursor = self.db_conn.cursor()
            query = "SELECT DISTINCT name FROM attendance WHERE date = %s"
            cursor.execute(query, (today_str,))
            rows = cursor.fetchall()
            cursor.close()
            
            for row in rows:
                self.students_marked_today.add(row[0])
            
            print(f"Loaded {len(self.students_marked_today)} students: {self.students_marked_today}")
                
        except Error as e:
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

        # Use grid for live_camera_frame content
        live_camera_frame.grid_rowconfigure(0, weight=0) # For stats_frame
        live_camera_frame.grid_rowconfigure(1, weight=1) # For video_label
        live_camera_frame.grid_rowconfigure(2, weight=0) # For video_control_frame
        live_camera_frame.grid_columnconfigure(0, weight=1)

        stats_frame = tk.Frame(live_camera_frame, bg="lightgray")
        stats_frame.grid(row=0, column=0, sticky="ew", pady=5, padx=10) # Placed in grid

        self.recognized_count_label = tk.Label(stats_frame, text="Recognized: 0", font=("Arial", 12), bg="lightgray")
        self.recognized_count_label.pack(side="left")

        self.fps_label = tk.Label(stats_frame, text="FPS: 0", font=("Arial", 12), bg="lightgray")
        self.fps_label.pack(side="right")

        self.video_label = tk.Label(live_camera_frame, bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=5) # Placed in grid and configured to expand

        # Frame for buttons below the video feed, inside live_camera_frame
        video_control_frame = tk.Frame(live_camera_frame, bg="lightgray")
        video_control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5) # Placed in grid

        btn_register = tk.Button(video_control_frame, text="+ Register Student", command=self.register_student, bg="#61AFEF", fg="white", font=("Arial", 10, "bold"))
        btn_register.pack(side="left", padx=5) # Reduced padx for more buttons
        btn_manual = tk.Button(video_control_frame, text="Manual Mark", command=self.manual_mark, bg="#E5C07B", fg="white", font=("Arial", 10, "bold"))
        btn_manual.pack(side="left", padx=5)
        btn_refresh = tk.Button(video_control_frame, text="Refresh", command=lambda: self.update_attendance_display(show_popup=True), bg="#98C379", fg="white", font=("Arial", 10, "bold"))
        btn_refresh.pack(side="left", padx=5)
        
        btn_export = tk.Button(video_control_frame, text="Export (XML)", command=self.export_to_xml, bg="#C678DD", fg="white", font=("Arial", 10, "bold"))
        btn_export.pack(side="left", padx=5)
        
        btn_web = tk.Button(video_control_frame, text="View Daily List (Web)", command=self.open_web_status, bg="#E06C75", fg="white", font=("Arial", 10, "bold"))
        btn_web.pack(side="right", padx=5) # Keep this on the right

        attendance_records_frame = tk.Frame(self.notebook, bg="lightgray")
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

        # Removed the button_frame that was at self.root level and packing side="bottom"
        # All buttons are now within video_control_frame inside live_camera_frame

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
        """Saves a single attendance record to the MySQL database."""
        try:
            if not self.db_conn.is_connected():
                self.db_conn = self.connect_to_database()
            
            cursor = self.db_conn.cursor()
            query = "INSERT INTO attendance (name, date, time, confidence) VALUES (%s, %s, %s, %s)"
            values = (name, date_str, time_str, conf_str)
            cursor.execute(query, values)
            self.db_conn.commit()
            cursor.close()
        except Error as e:
            print(f"Error saving attendance to DB: {e}")
            if e.errno == 2006: 
                print("Reconnecting to database...")
                self.db_conn = self.connect_to_database()

    def load_attendance_from_db(self):
        """Loads all attendance records from the database for the GUI list."""
        self.attendance_records_data = []
        try:
            if not self.db_conn.is_connected():
                self.db_conn = self.connect_to_database()
                
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT name, date, time, confidence FROM attendance ORDER BY date DESC, time DESC")
            rows = cursor.fetchall()
            cursor.close()
            
            for row in rows:
                name, date, time_obj, confidence = row
                date_str = date.strftime("%Y-%m-%d")
                time_str = (datetime.min + time_obj).strftime("%I:%M:%S %p") 
                self.attendance_records_data.append((name, date_str, time_str, confidence))
                
        except Error as e:
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
            frame = results["frame"] # Full-size BGR frame
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
                cv2.rectangle(display_frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                
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
        
        self.video_processor.stop() 
        self.video_processor.join(timeout=1.0) 
        if self.video_processor.is_alive():
            print("Warning: Main video processor still alive during registration.")
        
        # <-- CHANGE 2: This block that created self.reg_capture is removed.
        # We will now use self.video_capture, which is already open.
            
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
            
        # <-- CHANGE 3: Using self.video_capture instead of self.reg_capture
        ret, frame = self.video_capture.read()
        
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
            # Updated print statement for clarity
            print("Failed to grab frame from self.video_capture in reg_feed")
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
        
        # <-- CHANGE 4: This block that released self.reg_capture is removed.
        # We no longer release the camera here, as the main processor will take over.
        
        if self.reg_window:
            self.reg_window.grab_release()
            self.reg_window.destroy()
            self.reg_window = None
            
        # We restart the main video processor, which will reuse self.video_capture
        self.video_processor = VideoProcessor(self.video_capture, self.known_face_encodings, self.known_face_names, RECOGNITION_THRESHOLD, self.processing_results_queue)
        self.video_processor.daemon = True
        self.video_processor.start()
        print("Main video processor restarted.")
        
        if self.snapshot_count > 0 and student_name and student_path:
            self.process_and_update_encodings_in_memory(student_name, student_path, self.snapshot_count)

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
            # Re-read encodings to avoid duplicates if this student was partially processed before
            current_known_encodings = []
            current_known_names = []
            try:
                with open('encodings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    current_known_encodings = data['encodings']
                    current_known_names = data['names']
            except FileNotFoundError:
                pass # No encodings yet, start fresh
            
            # Filter out existing encodings for this student to re-encode (useful if photos are changed)
            temp_encodings = []
            temp_names = []
            for i, name in enumerate(current_known_names):
                if name != student_name:
                    temp_encodings.append(current_known_encodings[i])
                    temp_names.append(current_known_names[i])

            # Now add new encodings for the student
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
            
            # Save the updated list back to the pkl file
            with open('encodings.pkl', 'wb') as f:
                pickle.dump({'encodings': temp_encodings, 'names': temp_names}, f)
            
            self.training_results_queue.put({"student_name": student_name, "count": new_encodings_count, "path": student_path})
            
        except Exception as e:
            self.training_results_queue.put({"student_name": student_name, "count": 0, "error": str(e)})

    def check_training_results(self):
        """Checks the queue for results from the background thread."""
        try:
            result = self.training_results_queue.get_nowait()
            
            self.root.config(cursor="")
            if hasattr(self, 'processing_popup') and self.processing_popup:
                self.processing_popup.grab_release()
                self.processing_popup.destroy()
                self.processing_popup = None

            if "error" in result:
                messagebox.showerror("Encoding Error", f"Failed to encode faces for {result['student_name']}: {result['error']}", parent=self.root)
                return

            student_name = result["student_name"]
            student_path = result["path"]
            new_encodings_count = result["count"]

            if new_encodings_count > 0:
                self.load_encodings() # Reload all encodings to ensure the video processor gets the most up-to-date list
                
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

    def export_to_xml(self):
        """Exports today's unique attendance list to an XML file."""
        print("Exporting to XML...")
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save XML Export",
            defaultextension=".xml",
            filetypes=[("XML Files", "*.xml")]
        )
        
        if not filename:
            print("Export cancelled.")
            return
            
        todays_students = []
        try:
            if not self.db_conn.is_connected():
                self.db_conn = self.connect_to_database()
            
            cursor = self.db_conn.cursor()
            query = """
            SELECT DISTINCT name, MIN(time) as first_seen
            FROM attendance
            WHERE date = %s
            GROUP BY name
            ORDER BY name
            """
            cursor.execute(query, (today_str,))
            rows = cursor.fetchall()
            cursor.close()
            
            for row in rows:
                name, time_obj = row
                todays_students.append({
                    "name": name,
                    "time": (datetime.min + time_obj).strftime("%I:%M:%S %p")
                })
        
        except Error as e:
            messagebox.showerror("Database Error", f"Failed to fetch data for export: {e}")
            return
            
        if not todays_students:
            messagebox.showinfo("Export", "No students have been marked present today.", parent=self.root)
            return
            
        root_element = ET.Element("AttendanceList")
        root_element.set("date", today_str)
        
        for student in todays_students:
            student_element = ET.SubElement(root_element, "Student")
            
            name_element = ET.SubElement(student_element, "Name")
            name_element.text = student["name"]
            
            time_element = ET.SubElement(student_element, "FirstSeen")
            time_element.text = student["time"]
            
        try:
            xml_string = ET.tostring(root_element, encoding='utf-8')
            
            dom = minidom.parseString(xml_string)
            pretty_xml_string = dom.toprettyxml(indent="  ")
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(pretty_xml_string)
                
            messagebox.showinfo("Export Successful", f"Successfully exported {len(todays_students)} students to:\n{filename}", parent=self.root)
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to write XML file: {e}", parent=self.root)


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
        if self.video_processor.is_alive():
            print("Warning: Main video processor did not terminate in time.")
        
        self.reg_cam_running = False
        
        # We also remove the check for self.reg_capture here, as it no longer exists.
        
        if self.reg_window:
            self.reg_window.destroy()
            
        if self.video_capture.isOpened():
            self.video_capture.release()
            print("Main video capture released.")
            
        if self.db_conn and self.db_conn.is_connected():
            self.db_conn.close()
            print("Database connection closed.")
        
        cv2.destroyAllWindows()
        self.root.destroy()
        print("Application closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()
