# === WEB ATTENDANCE VIEWER FOR MOBILE & PC ===
# Reads from the DAILY CSV FILE to support "Session Resets"

from flask import Flask, render_template
import csv
import os
from datetime import date, datetime

app = Flask(__name__)

@app.route("/")
def home():
    return "<h2>Attendance Viewer Running</h2><br>Go to <a href='/attendance'>/attendance</a>"

@app.route("/attendance")
def attendance_today():
    # 1. Construct the filename for TODAY (e.g., Attendance_2025-01-02.csv)
    today_str = date.today().strftime("%Y-%m-%d")
    filename = f"Attendance_{today_str}.csv"
    
    students = []
    
    # 2. Check if file exists
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # 3. Find the LAST "New Class" marker
                start_index = 0
                for i, row in enumerate(rows):
                    # Check if this row is the separator line
                    if len(row) > 1 and "NEW CLASS STARTED" in row[1]:
                        start_index = i + 1  # Start reading AFTER this line
                
                # 4. Collect students from that point onwards
                # CSV Structure: [ID, Name, Time, Date, Method]
                for row in rows[start_index:]:
                    # Ensure it's a valid student row (has ID, not a separator)
                    if len(row) > 2 and row[0] != "---":
                        name = row[1]
                        time_raw = row[2]
                        
                        # Convert 24hr time to AM/PM for display
                        try:
                            t = datetime.strptime(time_raw, "%H:%M:%S")
                            time_str = t.strftime("%I:%M:%S %p")
                        except:
                            time_str = time_raw

                        # Deduplicate: Only add if not already in the list
                        # (So if Naveen is marked 3 times, show him once)
                        if not any(s['name'] == name for s in students):
                            students.append({
                                "name": name,
                                "time": time_str
                            })
                            
        except Exception as e:
            print(f"Error reading CSV: {e}")

    # 5. Send to HTML
    return render_template("student_list.html", students=students)
    

if __name__ == "__main__":
    print("\n🚀 Web Attendance Viewer Started")
    print("👉 Session Logic: Synced with CSV 'New Class' marker.")
    print("\n📲 To check on Mobile (same WiFi):")
    print("   http://YOUR-LAPTOP-IP:5001/attendance\n")

    app.run(host="0.0.0.0", debug=True, port=5001)