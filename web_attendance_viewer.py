# web_attendance_viewer.py
from flask import Flask, render_template, send_from_directory
import os
from datetime import date, datetime
import sqlite3  # <-- CHANGED: No more mysql
from sqlite3 import Error

app = Flask(__name__)

# --- DB Configuration ---
# This is now just a file name! No server needed.
DB_NAME = 'attendance.db'

def get_db_connection():
    """Establishes a connection to the database."""
    try:
        # conn.row_factory makes the results act like dictionaries
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row  # <-- ADDED: To get column names
        return conn
    except Error as e:
        print(f"Error connecting to SQLite: {e}")
        return None

@app.route('/attendance_proofs/<filename>')
def serve_proof_image(filename):
    # This route is unchanged
    return send_from_directory('attendance_proofs', filename)

@app.route('/')
def home():
    return "Welcome to the Student Attendance Viewer. Go to <a href='/attendance'>/attendance</a> to see the list."

@app.route('/attendance')
def student_attendance_list():
    present_students = []
    today_str = date.today().strftime("%Y-%m-%d")
    
    conn = get_db_connection()
    if not conn:
        return render_template('student_list.html', students=[], error="Could not connect to the attendance database.")

    try:
        cursor = conn.cursor()
        
        # Query for unique students present today, with their first seen time
        # CHANGED: Placeholder is now '?' instead of '%s'
        query = """
        SELECT name, MIN(time) as first_seen
        FROM attendance
        WHERE date = ?
        GROUP BY name
        ORDER BY name
        """
        cursor.execute(query, (today_str,))
        rows = cursor.fetchall()
        
        for row in rows:
            # CHANGED: The database now stores time as a text string (e.g., "14:30:05")
            # We need to parse this string into a time object to re-format it.
            try:
                time_obj = datetime.strptime(row['first_seen'], "%H:%M:%S").time()
                time_str = time_obj.strftime("%I:%M:%S %p")
            except ValueError:
                time_str = row['first_seen'] # Fallback if time format is wrong

            present_students.append({
                'name': row['name'],
                'time': time_str
            })
            
    except Error as e:
        print(f"Error querying database: {e}")
        return render_template('student_list.html', students=[], error=f"An error occurred: {e}")
    finally:
        if conn:
            conn.close() # CHANGED: How sqlite connections are closed

    return render_template('student_list.html', students=present_students, error=None)

if __name__ == '__main__':
    # Run on 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', debug=True, port=5001)
