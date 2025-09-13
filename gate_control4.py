import cv2
import sqlite3
import datetime
from fast_alpr import ALPR

DB_FILE = "gate_logs.db"

# Connect to DB and make sure tables exist
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS authorized_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    plate TEXT NOT NULL UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    name TEXT,
    camera TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
""")

conn.commit()
conn.close()


def log_event(plate, camera):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Look up if this plate is in authorized_people
    cursor.execute("SELECT name FROM authorized_people WHERE plate = ?", (plate,))
    row = cursor.fetchone()
    name = row[0] if row else "Unknown"

    # Save event
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO logs (plate, name, camera, timestamp) VALUES (?, ?, ?, ?)",
                   (plate, name, camera, timestamp))

    conn.commit()
    conn.close()

    print(f"📌 Logged: {plate} - {name} ({camera}) @ {timestamp}")


# Initialize ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Cameras (Entrance + Exit)
cameras = {
    "Entrance": "rtsp://admin:admin@192.168.1.35:554/stream1",
    "Exit": "rtsp://admin:admin@192.168.1.35:554/stream2"
}

caps = {name: cv2.VideoCapture(url) for name, url in cameras.items()}
frame_count = 0

while True:
    for cam_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 30 != 0:  # process every 15th frame
            continue

        results = alpr.predict(frame)

        for res in results:
            plate = res.ocr.text.strip()
            confidence = res.ocr.confidence

            if confidence > 0.9:
                print(f"Detected on {cam_name}: {plate} (conf: {confidence:.2f})")
                log_event(plate, cam_name)

        # Optional live preview
        annotated = alpr.draw_predictions(frame)
        cv2.imshow(f"ALPR Gate System - {cam_name}", annotated)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
