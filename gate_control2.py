import cv2
import threading
import sqlite3
import datetime
from fast_alpr import ALPR

DB_FILE = "gate_logs.db"

# --- Database Setup ---
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

# Example authorized people (insert if missing)
authorized = [
    ("Mert Akten", "34ABC123"),
    ("Merve Çamlıdağ", "35XYZ789"),
    ("Gökalp Karayılan", "06MERT06"),
    ("Nurten Onut", "16TEST16"),
]
cursor.executemany("INSERT OR IGNORE INTO authorized_people (name, plate) VALUES (?, ?)", authorized)

conn.commit()
conn.close()


# --- Logger ---
def log_event(plate, camera):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM authorized_people WHERE plate = ?", (plate,))
    row = cursor.fetchone()
    if row:
        name = row[0]
        authorized = True
    else:
        name = "Unknown"
        authorized = False

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (plate, name, camera, timestamp) VALUES (?, ?, ?, ?)",
        (plate, name, camera, timestamp),
    )

    conn.commit()
    conn.close()

    print(f"📌 Logged: {plate} - {name} ({camera}) @ {timestamp}")
    return authorized, name


# --- Gate Control Stub ---
def open_gate(camera):
    print(f"🚪 Gate OPEN triggered at {camera}!")


# --- ALPR Setup ---
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

exit_flag = False  # global flag to stop threads


# --- Camera Processing Function ---
def process_camera(rtsp_url, window_name):
    global exit_flag
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    frame_count = 0

    while cap.isOpened() and not exit_flag:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 60 == 0:  # process every 30th frame
            results = alpr.predict(frame)

            for res in results:
                plate = res.ocr.text.strip()
                confidence = res.ocr.confidence

                print(f"[{window_name}] Detected: {plate} (conf: {confidence:.2f})")

                if confidence > 0.9:
                    authorized, name = log_event(plate, window_name)
                    if authorized:
                        open_gate(window_name)

        annotated = alpr.draw_predictions(frame)
        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit_flag = True
            break

    cap.release()


# --- RTSP Feeds ---
entrance_cam = "rtsp://admin:admin@192.168.1.35:554/stream"
exit_cam = "rtsp://admin:admin@192.168.1.35:554/stream2"

# Run both cameras in parallel threads
t1 = threading.Thread(target=process_camera, args=(entrance_cam, "Entrance Camera"))
t2 = threading.Thread(target=process_camera, args=(exit_cam, "Exit Camera"))

t1.start()
t2.start()

t1.join()
t2.join()

cv2.destroyAllWindows()
