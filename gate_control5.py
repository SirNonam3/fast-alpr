import cv2
import threading
import queue
import sqlite3
import datetime
from fast_alpr import ALPR
import time

DB_FILE = "gate_logs.db"
exit_flag = False
COOLDOWN = 15  # seconds
last_seen = {}  # {plate: last_detection_time}

# --- Database setup ---
def init_db():
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
    authorized = [
        ("Mert Akten", "MN397"),
        ("Merve Çamlıdağ", "35XYZ789"),
        ("Gökalp Karayılan", "06MERT06"),
        ("Nurten Onut", "16TEST16"),
    ]
    cursor.executemany("INSERT OR IGNORE INTO authorized_people (name, plate) VALUES (?, ?)", authorized)
    conn.commit()
    conn.close()
init_db()

# --- Logger ---
def log_event(plate, camera):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM authorized_people WHERE plate = ?", (plate,))
    row = cursor.fetchone()
    name = row[0] if row else "Unknown"
    authorized = bool(row)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (plate, name, camera, timestamp) VALUES (?, ?, ?, ?)",
        (plate, name, camera, timestamp),
    )
    conn.commit()
    conn.close()
    print(f"📌 Logged: {plate} - {name} ({camera}) @ {timestamp}")
    return authorized, name

# --- Gate control ---
def open_gate(camera):
    print(f"🚪 Gate OPEN triggered at {camera}!")

# --- ALPR setup ---
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# --- Frame Queues ---
frame_raw_queues = { "Entrance": queue.Queue(maxsize=1), "Exit": queue.Queue(maxsize=1) }
frame_disp_queues = { "Entrance": queue.Queue(maxsize=1), "Exit": queue.Queue(maxsize=1) }

# --- Capture thread ---
def capture_camera(rtsp_url, cam_name):
    global exit_flag
    cap = cv2.VideoCapture(rtsp_url)
    while not exit_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        if not frame_raw_queues[cam_name].empty():
            try:
                frame_raw_queues[cam_name].get_nowait()
            except queue.Empty:
                pass
        frame_raw_queues[cam_name].put(frame)
    cap.release()

# --- Processing thread ---
def process_camera(cam_name):
    global exit_flag
    while not exit_flag:
        if not frame_raw_queues[cam_name].empty():
            frame = frame_raw_queues[cam_name].get()
            results = alpr.predict(frame)
            now = time.time()
            for res in results:
                plate = res.ocr.text.strip()
                confidence = res.ocr.confidence
                if confidence > 0.9:
                    # Check cooldown
                    if plate in last_seen and now - last_seen[plate] < COOLDOWN:
                        continue
                    last_seen[plate] = now
                    authorized, _ = log_event(plate, cam_name)
                    if authorized:
                        open_gate(cam_name)
            annotated = alpr.draw_predictions(frame)
            if not frame_disp_queues[cam_name].empty():
                try:
                    frame_disp_queues[cam_name].get_nowait()
                except queue.Empty:
                    pass
            frame_disp_queues[cam_name].put(annotated)
        else:
            time.sleep(0.01)

# --- RTSP cameras ---
cameras = {
    "Entrance": "rtsp://admin:admin@192.168.1.35:554/stream",
    "Exit": "rtsp://admin:admin@192.168.1.35:554/stream2"
}

# --- Start threads ---
threads = []
for name, url in cameras.items():
    t1 = threading.Thread(target=capture_camera, args=(url, name), daemon=True)
    t2 = threading.Thread(target=process_camera, args=(name,), daemon=True)
    t1.start()
    t2.start()
    threads.extend([t1, t2])

# --- Main display loop (main thread) ---
while not exit_flag:
    for name in cameras.keys():
        if not frame_disp_queues[name].empty():
            frame = frame_disp_queues[name].get()
            cv2.imshow(f"ALPR - {name}", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit_flag = True
        break

cv2.destroyAllWindows()
