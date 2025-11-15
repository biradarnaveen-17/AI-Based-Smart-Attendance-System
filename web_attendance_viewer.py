# web_attendance_viewer.py
from flask import Flask, render_template, send_from_directory
import pandas as pd
import os
from datetime import date

app = Flask(__name__)

# Register the attendance_proofs directory to serve static images
@app.route('/attendance_proofs/<filename>')
def serve_proof_image(filename):
    return send_from_directory('attendance_proofs', filename)

@app.route('/')
def home():
    return "Welcome to the Student Attendance Viewer. Go to <a href='/attendance'>/attendance</a> to see the list."

@app.route('/attendance')
def student_attendance_list():
    present_students = []
    
    today_str = date.today().strftime("%Y-%m-%d")

    try:
        if not os.path.exists("attendance.csv"):
            return render_template('student_list.html', students=[], error="Attendance record file not found.")

        df = pd.read_csv('attendance.csv')
        
        if df.empty:
            return render_template('student_list.html', students=[], error="No attendance records yet for today.")

        # --- FIX TO PREVENT CRASH ---
        # We drop duplicates by 'Name' because 'ID' is not in the CSV
        present_df = df[df['Date'] == today_str].drop_duplicates(subset=['Name'])

        # --- CODE THAT PRODUCES THE SCREENSHOT ---
        # We send 'id' as 'N/A' and the 'timestamp' as the full time
        for index, row in present_df.iterrows():
            present_students.append({
                'name': row.get('Name', 'N/A'),
                'id': 'N/A', # This is where "N/A" comes from
                'timestamp': f"{row.get('Time', 'N/A')}" # This is where "14:36:12" comes from
            })
            
    except FileNotFoundError:
        return render_template('student_list.html', students=[], error="Attendance records are not available.")
    except KeyError as e:
        # This will catch if 'Date' or 'Name' is missing
        return render_template('student_list.html', students=[], error=f"Error reading attendance data: Missing column {e}. Please ensure attendance.csv is correctly formatted.")
    except Exception as e:
        return render_template('student_list.html', students=[], error=f"An error occurred: {e}")

    return render_template('student_list.html', students=present_students)

if __name__ == '__main__':
    # Run on 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', debug=True, port=5001)