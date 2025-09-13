import sqlite3
from datetime import datetime, timedelta

DB_FILE = "gate_logs.db"

# Example data
sample_logs = [
    ("34ABC123", "Ahmet Yılmaz", "North Gate", datetime.now() - timedelta(days=2)),
    ("06XYZ789", "Merve Çamlıdağ", "South Gate", datetime.now() - timedelta(days=1, hours=5)),
    ("35LMN456", "Gökalp Karayılan", "Main Entrance", datetime.now() - timedelta(hours=12)),
    ("07JKL321", "Nurten Onut", "North Gate", datetime.now() - timedelta(hours=3)),
    ("01QWE987", "Mert Akten", "South Gate", datetime.now()),
]

# Insert into DB
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

for plate, name, camera, ts in sample_logs:
    cursor.execute(
        "INSERT INTO logs (plate, name, camera, timestamp) VALUES (?, ?, ?, ?)",
        (plate, name, camera, ts.strftime("%Y-%m-%d %H:%M:%S")),
    )

conn.commit()
conn.close()

print("✅ Example data inserted into gate_logs.db")
