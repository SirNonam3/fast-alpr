import sqlite3

DB_FILE = "gate_logs.db"

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create authorized people table
cursor.execute("""
CREATE TABLE IF NOT EXISTS authorized_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    plate TEXT NOT NULL UNIQUE
)
""")

# Create logs table
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    name TEXT,
    camera TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
""")

# Insert some example authorized people
authorized = [
    ("Mert Akten", "34ABC123"),
    
]

cursor.executemany("INSERT OR IGNORE INTO authorized_people (name, plate) VALUES (?, ?)", authorized)

conn.commit()
conn.close()

print("✅ Authorized people table created and filled with example data")
