import sqlite3
import os

DB_NAME = 'attendance.db'

def connect_to_database():
    """Connects to the SQLite database file."""
    try:
        conn = sqlite3.connect(DB_NAME)
        print(f"Connected to database '{DB_NAME}'")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to SQLite: {e}")
        return None

def setup_database(conn):
    """Creates the attendance table if it doesn't exist."""
    try:
        cursor = conn.cursor()
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
        conn.commit()
        print("Attendance table is ready.")
    except sqlite3.Error as e:
        print(f"Failed to create table: {e}")

if __name__ == "__main__":
    if os.path.exists(DB_NAME):
        print(f"Database file '{DB_NAME}' already exists.")
    else:
        print(f"Creating database file '{DB_NAME}'...")
        
    db_conn = connect_to_database()
    
    if db_conn:
        setup_database(db_conn)
        db_conn.close()
        print("Database setup complete.")
    else:
        print("Failed to initialize database.")