import sqlite3
import os

def create_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect("logs/visitors.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            image_path TEXT,
            event_type TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(person_id, image_path, event_type):
    conn = sqlite3.connect("logs/visitors.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO events (person_id, image_path, event_type, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    """, (person_id, image_path, event_type))
    conn.commit()
    conn.close()
