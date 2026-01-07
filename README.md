# 📌 AI Based Smart Attendance System

## 📖 Project Overview
Taking attendance manually in classrooms is time-consuming, error-prone, and allows proxy attendance.  
The **AI Based Smart Attendance System** automates this process using **real-time face recognition**, ensuring accuracy, speed, and contactless operation.

The system detects student faces using a webcam and automatically records attendance with date and time, eliminating roll calls and biometric contact.

---

## 🎯 Objectives
- Automate attendance using face recognition
- Prevent proxy attendance
- Save classroom time
- Maintain digital attendance records
- Provide an easy-to-use interface for teachers
- Run on regular laptops without expensive hardware

---

## 🧠 Key Features
- Real-time face detection and recognition  
- Automatic attendance marking  
- Contactless and hygienic system  
- User-friendly Tkinter GUI  
- SQLite database for attendance storage  
- CSV export of attendance records  
- Flask-based web viewer for mobile access  
- Works efficiently on low-end systems  

---

## 🛠️ Technologies Used

| Category | Tools |
|-------|------|
| Programming Language | Python |
| Computer Vision | OpenCV, MediaPipe |
| Face Recognition | LBPH Algorithm |
| GUI | Tkinter |
| Database | SQLite |
| Web Viewer | Flask |
| Hardware | Laptop / PC with Webcam |

---

## 🧩 System Architecture
![System Architecture](screenshots/system_architecture.png)

**Architecture Components:**
- GUI Dashboard (Teacher Interface)
- Backend Processing (Face Detection & Recognition)
- Local Database (Attendance & Student Data)

---

## 🔄 Workflow
![Workflow Diagram](screenshots/workflow.png)

1. Student registration and image capture  
2. Face model training  
3. Live webcam detection  
4. Face recognition  
5. Attendance logging  
6. Database update  
7. GUI display  

---

## 🧪 Methodology
- MediaPipe for fast face detection  
- LBPH algorithm for lightweight face recognition  
- SQLite for local data storage  
- Multi-threading for smooth GUI and video processing  
- Flask server for real-time mobile viewing  

---

## 🖥️ Application Screenshots

### 🏠 Main Dashboard
![Dashboard](screenshots/dashboard.png)

### 👤 Student Registration
![Registration](screenshots/student_registration.png)

### 🎥 Live Attendance Marking
![Live Attendance](screenshots/live_attendance.png)

### 📊 Attendance Records
![Attendance Records](screenshots/attendance_records.png)

---

## 📈 Results & Performance
- Recognition Accuracy: **92% – 95%**
- Speed: **12–18 FPS (single face)**, **8–12 FPS (multiple faces)**
- One attendance entry per student per day
- Proxy attendance completely eliminated during testing

---

## ✅ Advantages
- Saves classroom time
- Improves accuracy of attendance records
- Prevents proxy attendance
- Cost-effective and portable
- No physical contact required

---

## 🚀 Future Enhancements
- Liveness detection (anti-spoofing)
- Cloud database synchronization
- Mobile application
- ERP/LMS integration
- Emotion recognition
- Low-light optimization

---

## 📂 Project Structure
