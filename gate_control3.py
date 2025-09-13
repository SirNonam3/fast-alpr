import cv2
import threading
import time
import sqlite3
from fast_alpr import ALPR

# Plates allowed to open the gate
AUTHORIZED_PLATES = {"5AU5341", "34ABC123", "06XYZ789"}

# Initialize ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Global flag to stop threads
exit_flag = False

# --- Database setup ---
conn = sqlite3.connect("gate_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT,
    camera TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def log_event(plate, camera):
    cursor.execute("INSERT INTO logs (plate, camera) VALUES (?, ?)", (plate, camera))
    conn.commit()
    print(f"📋 Logged: {plate} from {camera}")

def trigger_gate():
    print("⚡ Gate relay triggered (opening gate)...")
    # relay = relaylib.Relay(port="COM3")
    # relay.on(1)
    # time.sleep(1)
    # relay.off(1)

def process_camera(rtsp_url, window_name):
    global exit_flag
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0

    while cap.isOpened() and not exit_flag:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 15 != 0:
            continue

        results = alpr.predict(frame)

        for res in results:
            plate = res.ocr.text.strip()
            confidence = res.ocr.confidence

            print(f"[{window_name}] Detected: {plate} (conf: {confidence:.2f})")

            if plate in AUTHORIZED_PLATES and confidence > 0.90:
                print(f"✅ [{window_name}] Gate Opened for {plate}")
                trigger_gate()
                log_event(plate, window_name)

        annotated = alpr.draw_predictions(frame)
        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit_flag = True
            break

    cap.release()

# RTSP camera feeds
entrance_cam = "rtsp://admin:admin@192.168.1.35:554/stream"
exit_cam = "rtsp://admin:admin@192.168.1.35:554/stream"

# Run both cameras in parallel
t1 = threading.Thread(target=process_camera, args=(entrance_cam, "Entrance Camera"))
t2 = threading.Thread(target=process_camera, args=(exit_cam, "Exit Camera"))

t1.start()
t2.start()

t1.join()
t2.join()

cv2.destroyAllWindows()
conn.close()
