# web_attendance_viewer.py
from flask import Flask, render_template, send_from_directory
import pandas as pd
import os
from datetime import date, datetime # <--- THIS IS THE FIX
import mysql.connector # Added for database
from mysql.connector import Error

app = Flask(__name__)

# --- DB Configuration ---
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = '' # Your XAMPP password, if you have one
DB_NAME = 'ai_attendance_system'

def get_db_connection():
    """Establishes a connection to the database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

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
    
    conn = get_db_connection()
    if not conn:
        return render_template('student_list.html', students=[], error="Could not connect to the attendance database.")

    try:
        cursor = conn.cursor(dictionary=True) # Use dictionary cursor
        
        # Query for unique students present today, with their first seen time
        query = """
        SELECT name, MIN(time) as first_seen
        FROM attendance
        WHERE date = %s
        GROUP BY name
        ORDER BY name
        """
        cursor.execute(query, (today_str,))
        rows = cursor.fetchall()
        
        for row in rows:
            # Convert time object to a readable string
            time_str = (datetime.min + row['first_seen']).strftime("%I:%M:%S %p")
            present_students.append({
                'name': row['name'],
                'time': time_str
            })
            
    except Error as e:
        print(f"Error querying database: {e}")
        return render_template('student_list.html', students=[], error=f"An error occurred: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

    return render_template('student_list.html', students=present_students, error=None)

if __name__ == '__main__':
    # Run on 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', debug=True, port=5001)
