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

# --- Configuration ---
RECOGNITION_THRESHOLD = 0.55 
ATTENDANCE_INTERVAL_SECONDS = 60 
PROCESS_EVERY_N_FRAMES = 5 
# --- End Configuration ---

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
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
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
                    scaled_top, scaled_right, scaled_bottom, scaled_left = top * 4, right * 4, bottom * 4, left * 4

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
                    "recognized_info": recognized_info, 
                    "recognized_count": recognized_count,
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
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.training_thread = None
        self.training_queue = queue.Queue() 
        
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
            
        self.display_width = 640  
        self.display_height = 480 

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

        self.attendance_log = {} 
        self.attendance_records_data = []

        self.create_widgets()
        self.update_video_feed() 
        self.root.after(100, self.check_training_result) 
        
        if self_load_encodings_called:
            self.update_attendance_display(show_popup=False) 
        
        self.reg_capture = None
        self.reg_window = None
        self.reg_video_label = None
        self.reg_cam_running = False
        self.current_reg_frame = None
        self.snapshot_count = 0


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
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#282C34", padx=10, pady=10)
        header_frame.pack(fill="x")
        header_label = tk.Label(header_frame, text="AI Attendance System (Integrated)", font=("Arial", 20, "bold"), fg="white", bg="#282C34")
        header_label.pack(pady=5)

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Live Camera Tab ---
        live_camera_frame = tk.Frame(self.notebook, bg="lightgray")
        self.notebook.add(live_camera_frame, text="Live Camera")

        stats_frame = tk.Frame(live_camera_frame, bg="lightgray")
        stats_frame.pack(fill="x", pady=5)

        self.recognized_count_label = tk.Label(stats_frame, text="Recognized: 0", font=("Arial", 12), bg="lightgray")
        self.recognized_count_label.pack(side="left", padx=10)

        self.fps_label = tk.Label(stats_frame, text="FPS: 0", font=("Arial", 12), bg="lightgray")
        self.fps_label.pack(side="right", padx=10)

        self.video_label = tk.Label(live_camera_frame, bg="black")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=5)
        self.video_label.bind("<Configure>", self.on_video_label_resize) 

        # --- Attendance Records Tab ---
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

        # --- Bottom Button Bar ---
        button_frame = tk.Frame(self.root, bg="#282C34", padx=10, pady=10)
        button_frame.pack(fill="x", side="bottom")

        btn_register = tk.Button(button_frame, text="+ Register Student", command=self.register_student, bg="#61AFEF", fg="white", font=("Arial", 10, "bold"))
        btn_register.pack(side="left", padx=10, pady=5)
        btn_manual = tk.Button(button_frame, text="Manual Mark", command=self.manual_mark, bg="#E5C07B", fg="white", font=("Arial", 10, "bold"))
        btn_manual.pack(side="left", padx=10, pady=5)
        btn_refresh = tk.Button(button_frame, text="Refresh", command=lambda: self.update_attendance_display(show_popup=True), bg="#98C379", fg="white", font=("Arial", 10, "bold"))
        btn_refresh.pack(side="left", padx=10, pady=5)
        
        btn_web = tk.Button(button_frame, text="View Daily List (Web)", command=self.open_web_status, bg="#E06C75", fg="white", font=("Arial", 10, "bold"))
        btn_web.pack(side="right", padx=10, pady=5)


    def open_web_status(self):
        """
        Opens the daily attendance list page in the default web browser.
        """
        webbrowser.open_new_tab("http://127.0.0.1:5001/attendance")


    def on_video_label_resize(self, event):
        self.display_width = event.width
        self.display_height = event.height

    def update_live_status(self, name):
        """
        Updates a local JSON file with the name of the most recently seen student.
        """
        status_data = {
            "name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open('live_status.json', 'w') as f:
                json.dump(status_data, f)
        except Exception as e:
            print(f"Error writing live status: {e}")

    def mark_attendance(self, name, confidence):
        current_time = time.time()
        
        if name != "Unknown":
            self.update_live_status(name)
        
        if name != "Unknown" and (name not in self.attendance_log or (current_time - self.attendance_log[name]) > ATTENDANCE_INTERVAL_SECONDS):
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            
            # --- THIS IS THE 24-HOUR TIME VERSION ---
            time_str = timestamp.strftime("%H:%M:%S")
            # --- END ---
            
            conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"

            record = (name, date_str, time_str, conf_str)
            self.attendance_records_data.append(record)
            self.attendance_log[name] = current_time 

            self.attendance_tree.insert("", "end", values=record)
            
            threading.Thread(target=self.save_attendance_to_csv, args=(record,), daemon=True).start()

    def save_attendance_to_csv(self, record):
        try:
            file_exists = os.path.isfile("attendance.csv")
            with open("attendance.csv", "a", newline='') as f: 
                if not file_exists:
                    f.write("Name,Date,Time,Confidence\n")
                f.write(f"{record[0]},{record[1]},{record[2]},{record[3]}\n")
        except Exception as e:
            print(f"Error saving attendance to CSV in thread: {e}")

    def load_attendance_from_csv(self):
        self.attendance_records_data = []
        try:
            if os.path.exists("attendance.csv"):
                with open("attendance.csv", "r") as f:
                    header = f.readline() 
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 4:
                            self.attendance_records_data.append(tuple(parts))
        except Exception as e:
             print(f"Error loading attendance from CSV: {e}")

    def update_attendance_display(self, show_popup=False):
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        self.load_attendance_from_csv()
        
        for record in self.attendance_records_data:
            self.attendance_tree.insert("", "end", values=record)
            
        if show_popup:
            messagebox.showinfo("Refresh Complete", 
                                f"Loaded {len(self.attendance_records_data)} records from attendance.csv.",
                                parent=self.root) 

    def update_video_feed(self):
        try:
            results = self.processing_results_queue.get_nowait()
            frame = results["frame"]
            recognized_info = results["recognized_info"]
            recognized_count = results["recognized_count"]
            is_fresh_data = results["is_fresh_data"] 

            for info in recognized_info:
                name = info["name"]
                confidence = info["confidence"]
                top, right, bottom, left = info["location"]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                label = f"{name}"
                if confidence is not None:
                    label += f" ({confidence:.1f}%)"

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                if is_fresh_data:
                    self.mark_attendance(name, confidence)

            if is_fresh_data:
                self.recognized_count_label.config(text=f"Recognized: {recognized_count}")
                self.fps_label.config(text=f"FPS: {int(self.video_processor.fps)}")
                
            self._last_displayed_frame = frame

        except queue.Empty:
            frame = None
            if hasattr(self, '_last_displayed_frame'):
                 frame = self._last_displayed_frame
            else:
                frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        if frame is not None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            
            target_width = self.display_width
            target_height = self.display_height
            if target_width < 1 or target_height < 1:
                target_width, target_height = 640, 480
            
            img.thumbnail((target_width, target_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)

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

        messagebox.showinfo("Upload Complete", f"Successfully copied {copied_count} photos.", parent=self.root)
        
        if copied_count > 0:
            self.ask_retrain_model()

    def open_registration_camera(self, student_name, student_path):
        self.reg_window = tk.Toplevel(self.root)
        self.reg_window.title(f"Register: {student_name}")
        self.reg_window.transient(self.root)
        self.reg_window.grab_set()
        
        self.video_processor.running = False 
        
        self.reg_capture = cv2.VideoCapture(0)
        if not self.reg_capture.isOpened():
            messagebox.showerror("Camera Error", 
                                 "Could not open camera. Please close and re-open the app.", 
                                 parent=self.root)
            self.reg_window.destroy()
            self.reg_capture = None
            self.video_processor.running = True 
            if not self.video_processor.is_alive():
                self.video_processor = VideoProcessor(
                    self.video_capture, self.known_face_encodings, self.known_face_names, RECOGNITION_THRESHOLD, self.processing_results_queue)
                self.video_processor.daemon = True
                self.video_processor.start()
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

        done_button = tk.Button(btn_frame, text="Done", command=self.close_registration_camera, bg="#98C379", fg="white", font=("Arial", 10, "bold"))
        done_button.pack(side="left", padx=10)

        self.reg_cam_running = True
        self.update_reg_feed()
        
        self.reg_window.protocol("WM_DELETE_WINDOW", self.close_registration_camera)

    def update_reg_feed(self):
        if not self.reg_cam_running:
            return
            
        ret, frame = self.reg_capture.read()
        if ret:
            self.current_reg_frame = frame
            
            rgb_frame_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame_small)
            
            for top, right, bottom, left in face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                color = (0, 255, 0) if len(face_locations) == 1 else (0, 0, 255) 
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)


            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        rgb_frame = cv2.cvtColor(self.current_reg_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
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

    def close_registration_camera(self):
        self.reg_cam_running = False
        
        if self.reg_capture:
            self.reg_capture.release()
            print("Registration camera released.")
            
        self.reg_capture = None
        
        if self.reg_window:
            self.reg_window.grab_release()
            self.reg_window.destroy()
            self.reg_window = None
            
        self.video_processor.running = True
        if not self.video_processor.is_alive():
            self.video_processor = VideoProcessor(
                self.video_capture,
                self.known_face_encodings,
                self.known_face_names,
                RECOGNITION_THRESHOLD,
                self.processing_results_queue
            )
            self.video_processor.daemon = True
            self.video_processor.start()

        if self.snapshot_count > 0:
            self.ask_retrain_model()

    def ask_retrain_model(self):
        answer = messagebox.askyesno(
            "Training Required",
            "Photos have been added. Would you like to re-train the AI model now?\n\nThis process will run in the background but may take time.",
            parent=self.root
        )
        
        if answer:
            self.run_training_script_threaded()

    def run_training_script_threaded(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training in Progress", "The AI model is already training.", parent=self.root)
            return

        self.wait_window = tk.Toplevel(self.root)
        self.wait_window.title("Training in Progress...")
        self.wait_window.geometry("350x120")
        self.wait_window.transient(self.root)
        self.wait_window.grab_set()
        
        tk.Label(self.wait_window, text="AI Model Training Running", font=("Arial", 14, "bold"), fg="#61AFEF").pack(pady=10)
        self.training_status_label = tk.Label(self.wait_window, text="Please wait.", font=("Arial", 12))
        self.training_status_label.pack(pady=5)
        
        self.root.update_idletasks()
        
        self.training_thread = threading.Thread(target=self.background_training_task, daemon=True)
        self.training_thread.start()


    def background_training_task(self):
        try:
            result = subprocess.run(
                ['python', 'train_model.py'],
                capture_output=True, text=True, check=True
            )
            self.training_queue.put({"status": "success", "stdout": result.stdout})
            
        except subprocess.CalledProcessError as e:
            self.training_queue.put({"status": "failure", "error": e.stderr})
        except FileNotFoundError:
            self.training_queue.put({"status": "failure", "error": "Error: 'python' command not found."})
        except Exception as e:
            self.training_queue.put({"status": "failure", "error": f"An unexpected error occurred: {e}"})

    def check_training_result(self):
        try:
            result = self.training_queue.get_nowait()
            
            if hasattr(self, 'wait_window') and self.wait_window:
                self.wait_window.grab_release()
                self.wait_window.destroy()
                self.wait_window = None
            
            if result['status'] == 'success':
                self.load_encodings()
                messagebox.showinfo(
                    "Training Complete",
                    f"Successfully trained and reloaded model.\n\nLoaded {len(self.known_face_encodings)} faces.",
                    parent=self.root
                )
            else:
                messagebox.showerror(
                    "Training Failed",
                    f"The training script failed.\n\nError:\n{result['error']}",
                    parent=self.root
                )
            self.training_thread = None 

        except queue.Empty:
            if self.training_thread and self.training_thread.is_alive() and hasattr(self, 'training_status_label'):
                current_text = self.training_status_label.cget("text")
                new_text = current_text + "."
                if len(new_text) > 15: 
                    new_text = "Please wait."
                self.training_status_label.config(text=new_text)
            pass
        
        self.root.after(500, self.check_training_result) 


    def manual_mark(self):
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Manual Mark Attendance")
        manual_window.geometry("300x150")

        manual_window.transient(self.root)  
        manual_window.grab_set()             
        manual_window.focus_force()          

        tk.Label(manual_window, text="Enter Student Name:").pack(pady=10)
        name_entry = tk.Entry(manual_window)
        name_entry.pack(pady=5)

        def submit_manual_mark():
            student_name = name_entry.get().strip()
            if student_name:
                self.mark_attendance(student_name, confidence=None)
                messagebox.showinfo("Success", 
                                    f"Manually marked attendance for {student_name}.",
                                    parent=manual_window)
                manual_window.grab_release() 
                manual_window.destroy()
            else:
                messagebox.showwarning("Input Error", 
                                      "Please enter a student name.",
                                      parent=manual_window)

        tk.Button(manual_window, text="Mark Present", command=submit_manual_mark).pack(pady=10)

    def on_closing(self):
        print("Closing application...")
        self.video_processor.stop()
        print("Waiting for video processor to join...")
        self.video_processor.join()
        print("Video processor joined.")
        
        self.reg_cam_running = False
        if self.reg_capture:
            self.reg_capture.release()
            print("Registration camera released.")
        if self.reg_window:
            self.reg_window.destroy()
            
        if self.video_capture.isOpened():
            self.video_capture.release()
            print("Video capture released.")
            
        cv2.destroyAllWindows()
        self.root.destroy()
        print("Application closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()