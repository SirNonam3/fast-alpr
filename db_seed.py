import sqlite3
from datetime import datetime, timedelta

DB_FILE = "gate_logs.db"

# Connect (creates file if not exists)
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    camera TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
""")

# Insert some test data
now = datetime.now()
sample_data = [
    ("34ABC123", "Entrance", (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")),
    ("35XYZ789", "Exit", (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")),
    ("06MERT06", "Entrance", now.strftime("%Y-%m-%d %H:%M:%S")),
    ("16TEST16", "Exit", now.strftime("%Y-%m-%d %H:%M:%S")),
]

cursor.executemany("INSERT INTO logs (plate, camera, timestamp) VALUES (?, ?, ?)", sample_data)

conn.commit()
conn.close()

print("✅ Example data inserted into gate_logs.db")
